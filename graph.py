"""The same harness as harness.py, rebuilt with LangGraph, so you can see the seven parts under a framework's names.

harness.py shows every part as plain Python. Real teams often use a framework instead, and then the parts hide
behind its vocabulary: a loop becomes a cycle in a graph, an approval becomes interrupt(), memory becomes a
checkpointer. Without seeing both side by side it is easy to believe a framework adds safety by itself. It does
not: the gate, the step cap and the "final is a claim" check still have to be written, just in other words.

It reuses tools.py for the to-do tools and harness.py for RunResult, TraceEntry and handover_note, and returns
the same RunResult as harness.run(), so try_it.py and agent.py can switch engines with one flag (--engine
langgraph). The to-do State stays outside the graph; the graph's own state holds only the conversation.

What to look at, and where each part lives
    GState            the graph's state: an append-only history (part 3, Context) and a steps counter that IS the
                      step cap (part 1, Loop), plus the pending call, the denial count and how the run ended.
    NODE decide       part 1, Loop = harness steps 1 and 3: ask the model once per step, never past max_steps.
    NODE gate         part 5, Gates = harness step 6: interrupt() pauses the graph, saves it, and resumes with the
                      human's answer. LangGraph reruns this node from the top on resume, so nothing with a side
                      effect may sit before interrupt(). An unknown tool is gated here (the harness refuses it).
    NODE act          part 2, Tools = harness steps 5 and 7: check the input, run the tool once, record the result.
    NODE check        part 6, Verify = harness step 2: "final" is a claim, so verify(state) looks at the real list.
    after_decide(), after_gate()
                      conditional edges: the loop's exits, as the graph's choice of the next node.
    InMemorySaver     part 4, Memory: the checkpointer that holds the paused run, keyed by thread_id. It lives only
                      as long as the process; agent.py saves tasks.json once after a langgraph run.
    run_graph()       the driver: stream() the updates into the trace (part 7, Trace), answer each pause with
                      approve(), resume with Command(resume=...), and add the hand-over note (Recovery).
    Not here, on purpose: the repeat check, the history window and the checkpoint hook.
"""
import json
import operator
from typing import Annotated, Any, Callable, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from harness import RunResult, Tool, TraceEntry, handover_note


class GState(TypedDict):
    """What the graph saves at every step: the conversation and the counters, never the to-do list itself."""
    history: Annotated[list[str], operator.add]     # append-only: a node returns new lines, LangGraph adds them
    log: Annotated[list[list[Any]], operator.add]   # [step, kind, detail] entries; stream() turns them into the trace
    steps: int                                      # model calls so far: the step cap reads this, nothing else
    pending: dict                                   # the call the model asked for, waiting for the gate or act
    denials: int
    status: str                                     # empty while running; set by the node that ends the run
    answer: str


def run_graph(goal: str, *, model: Callable, tools: list[Tool], state: Any, max_steps: int = 10,
              max_result_chars: int = 300, max_denials: int = 3, approve: Callable | None = None,
              verify: Callable | None = None, thread_id: str = "t1") -> RunResult:
    """Run the goal through a LangGraph graph; same inputs and same RunResult as harness.run()."""
    by_name = {tool.name: tool for tool in tools}

    def cut(text: str) -> str:
        """Context: the same cut as the harness, so one huge result cannot flood the model or the trace."""
        return text if len(text) <= max_result_chars else text[:max_result_chars] + " [truncated]"

    def call_text(pending: dict) -> str:
        """'<tool> <JSON args>' with sorted keys, exactly as harness.py writes it."""
        return f"{pending['name']} {json.dumps(pending['args'], sort_keys=True)}"

    # NODE decide [part 1, Loop] = harness steps 1 and 3
    def decide(s: GState) -> dict:
        """Ask the model for one decision, unless the step cap is reached."""
        step = s["steps"] + 1
        if s["steps"] >= max_steps:            # the cap lives in the state, so a pause and resume cannot reset it
            return {"status": "step-limit", "answer": "stopped: step limit reached",
                    "log": [[s["steps"], "stop", "step limit reached"]]}
        try:
            decision = model(list(s["history"]))
            if not isinstance(decision, dict) or not ("final" in decision or "tool" in decision):
                raise ValueError("malformed model output")
        except Exception as err:               # a broken model ends the run, it never crashes it
            return {"steps": step, "status": "model-error", "answer": "stopped: the model call failed",
                    "log": [[step, "stop", cut(f"model error: {err}")]]}
        if "final" in decision:                # final wins over a tool named in the same reply
            answer = str(decision["final"])
            return {"steps": step, "answer": answer, "pending": {"final": True},
                    "log": [[step, "decide", cut(f"final: {answer}")]]}
        args = decision["args"] if isinstance(decision.get("args"), dict) else {}
        pending = {"name": str(decision["tool"]), "args": args}
        return {"steps": step, "pending": pending, "history": [f"assistant wants {call_text(pending)}"],
                "log": [[step, "decide", cut(call_text(pending))]]}

    # EDGE after decide [part 1, Loop]: the exits and the choice of the next node
    def after_decide(s: GState) -> str:
        """Pick the next node: stop, check a final answer, ask the human, or act."""
        if s["status"]:
            return END
        if s["pending"].get("final"):
            return "check"
        tool = by_name.get(s["pending"]["name"])
        if tool is not None:
            try:
                if tool.validate(s["pending"]["args"]):
                    return "act"               # bad input is refused in act; never put a broken call to a human
            except Exception:
                return "act"
        if tool is None or tool.risk == "irreversible":
            return "gate"                      # an unknown tool is gated: nobody knows what it would do
        return "act"

    # NODE gate [part 5, Gates] = harness step 6
    def gate(s: GState) -> dict:
        """Pause for a human; only a real yes lets the call through. Nothing before interrupt() changes anything."""
        pending, step = s["pending"], s["steps"]
        tool = by_name.get(pending["name"])
        try:
            words = tool.describe(pending["args"], state) if tool and tool.describe else call_text(pending)
        except Exception:
            words = None                       # a broken description is a refusal, decided by the driver
        allowed = interrupt({"name": pending["name"], "args": pending["args"], "words": words})
        if allowed:
            return {}
        denials = s["denials"] + 1
        update = {"denials": denials, "log": [[step, "denied", cut(call_text(pending))]],
                  "history": [f"denied: {call_text(pending)} was not approved. Do not retry it."]}
        if denials >= max_denials:             # stop pestering the human
            update.update(status="denied-limit", answer="stopped: too many actions were denied")
            update["log"] = update["log"] + [[step, "stop", "too many denied actions"]]
        return update

    # EDGE after gate [part 5, Gates]: yes -> act, no -> back to decide, cap reached -> stop
    def after_gate(s: GState) -> str:
        """Pick the next node after the human answered."""
        if s["status"]:
            return END
        return "decide" if s["log"] and s["log"][-1][1] == "denied" else "act"

    # NODE act [part 2, Tools] = harness steps 5 and 7
    def act(s: GState) -> dict:
        """Check the input and run the tool once; any problem goes back to the model as an error line."""
        pending, step = s["pending"], s["steps"]
        tool = by_name.get(pending["name"])
        try:
            problem = f"unknown tool {pending['name']}" if tool is None else tool.validate(pending["args"])
        except Exception as err:
            problem = f"input check crashed: {err}"
        if problem:
            return {"history": [cut(f"error: {problem}")], "log": [[step, "error", cut(f"error: {problem}")]]}
        try:
            result = tool.run(pending["args"], state)
        except Exception as err:               # never retried: a repeated write is worse than one failure
            return {"history": [cut(f"error: {err}")], "log": [[step, "error", cut(f"error: {err}")]]}
        return {"history": [f"tool: {cut(result)}"], "log": [[step, "tool", cut(result)]]}

    # NODE check [part 6, Verify] = harness step 2
    def check(s: GState) -> dict:
        """Look at the real list before calling the run done; a check that crashes has not passed."""
        problem = None
        if verify is not None:
            try:
                problem = verify(state)
            except Exception as err:
                problem = f"the check itself crashed: {err}"
        if problem:
            return {"status": "failed-verification", "answer": f"not done: {problem}",
                    "log": [[s["steps"], "stop", cut(f"verification failed: {problem}")]]}
        return {"status": "done-with-denials" if s["denials"] else "done"}

    builder = StateGraph(GState)
    builder.add_node("decide", decide)
    builder.add_node("gate", gate)
    builder.add_node("act", act)
    builder.add_node("check", check)
    builder.add_edge(START, "decide")
    builder.add_conditional_edges("decide", after_decide, ["check", "gate", "act", END])
    builder.add_conditional_edges("gate", after_gate, ["act", "decide", END])
    builder.add_edge("act", "decide")          # the cycle: the loop of harness.run()
    builder.add_edge("check", END)
    graph = builder.compile(checkpointer=InMemorySaver())   # memory: a paused run waits here, keyed by thread_id

    # recursion_limit is only a backstop; the steps counter is the real cap (each step visits at most 3 nodes)
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": max_steps * 6 + 10}
    trace: list[TraceEntry] = []
    payload: Any = {"history": [f"user: {goal}"], "log": [], "steps": 0, "pending": {}, "denials": 0,
                    "status": "", "answer": ""}
    try:
        while True:
            question = None
            for update in graph.stream(payload, config, stream_mode="updates"):
                if "__interrupt__" in update:
                    question = update["__interrupt__"][0].value
                    continue
                for node_update in update.values():          # trace (part 7): one entry per logged event
                    for step, kind, detail in (node_update or {}).get("log", []):
                        trace.append(TraceEntry(step, kind, detail))
            if question is None:
                break
            step = graph.get_state(config).values["steps"]
            trace.append(TraceEntry(step, "pause", cut(f"paused at the gate, state saved: {question['words']}")))
            allowed = False                                  # default deny: no approver, or one that crashes, means no
            try:
                if approve is not None and question["words"] is not None:
                    allowed = bool(approve(question["name"], question["args"], question["words"]))
            except Exception:
                allowed = False
            payload = Command(resume=allowed)
    except GraphRecursionError:                              # the backstop: should never fire before the steps cap
        trace.append(TraceEntry(max_steps, "stop", "step limit reached (recursion backstop)"))
        return RunResult("step-limit", "stopped: step limit reached", 0, trace,
                         handover_note("step-limit", goal, trace))

    values = graph.get_state(config).values
    status = values["status"] or "step-limit"
    return RunResult(status, values["answer"], values["denials"], trace, handover_note(status, goal, trace))
