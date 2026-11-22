"""The harness: the loop that sits between a model and your to-do list, and decides what actually happens.

A model can only suggest. Left alone, a loop around a model will call tools that do not exist, pass nonsense
arguments, crash on the first failing call, or keep going forever, and nobody can tell afterwards what it did.
This file is the ordinary code that stops all of that: it asks the model for one decision at a time, looks the
tool up, checks the input, runs it, tells the model what happened, and writes a numbered trace of every step.

It knows nothing about tasks. tools.py defines the to-do tools and imports Tool from here; this file treats the
state as an opaque object that only the tools touch. The model is any function that takes the history lines and
returns {"tool": name, "args": {...}} or {"final": text}, so tests can pass a scripted fake and llm.py a real one.

What to look at, and where each harness part lives
    Tool         (Tools) the contract for one action: its name, the parameters the model may send, how risky it
                 is, how to check the input, and how to run it.
    TraceEntry   (Trace) one numbered line of the record of what happened.
    RunResult    the answer plus a status decided by code, never by the model's own claim.
    run()        (Loop) the decide, act, observe, stop cycle, one model call per step, capped at max_steps.
                 Step 1 asks the model, step 2 handles "final", step 3 records the call, step 4 is the repeat
                 check, step 5 looks the tool up and checks its input, step 6 is the gate, step 7 runs the tool,
                 step 8 saves progress.
    step 8       (Memory) after every successful tool that changed something, checkpoint(state) saves the list.
                 Reads save nothing. If saving fails the run stops "checkpoint-failed": losing work silently is
                 worse than stopping and saying so.
    step 2       (Verify) "final" is a claim, not a fact: verify(state) looks at the real list, and a problem, or a
                 check that crashes, ends the run "failed-verification" instead of "done".
    handover_note(), finish()
                 (Recovery) every exit goes through finish(), so every early stop carries a plain note with the
                 goal, the status and the last step, for the person who has to pick the work up.
    step 7       (Tools, Recovery) a failing tool is reported once and never retried: a repeated write is worse
                 than one honest failure.
    loop exits   (Loop) three hard limits that live in code, because "please don't loop" in a prompt is only a
                 request: the step cap (max_steps model calls), the repeat check in step 4 (the same call 3 times
                 within the last 6), and the denial cap in step 6 (max_denials refusals in total).
    step 6       (Gates) only an irreversible tool asks a human, and the question shows the real effect. No
                 approver, or an approver or describe() that crashes, means no: the default is deny.
    cut()        (Context) no tool result or error, and no trace line, may be longer than max_result_chars, so
                 one huge task cannot flood the model or bury the rest of the trace.
    view()       (Context) what the model sees each turn: the goal, any refusals that slid out of the window, then
                 the last max_history lines. A refusal is never forgotten, or the model would simply ask again.
    _args_text() sorted JSON for arguments, so the same call always reads the same in history and trace.
"""
import json
from dataclasses import dataclass
from typing import Any, Callable, Literal

Risk = Literal["read", "write", "irreversible"]
Status = Literal["done", "done-with-denials", "failed-verification", "step-limit", "loop-detected",
                 "denied-limit", "model-error", "checkpoint-failed"]
TraceKind = Literal["decide", "tool", "denied", "error", "stop", "pause"]

Model = Callable[[list[str]], dict[str, Any]]


@dataclass
class Tool:
    """One action the model may ask for, with the rules the harness enforces around it."""
    name: str
    description: str                                     # shown to the model in its tool menu
    params: dict[str, str]                               # parameter name -> type word, also shown in the menu
    risk: Risk                                           # read by the harness, never by the model: a prompt cannot lower it
    validate: Callable[[dict[str, Any]], "str | None"]   # a problem sentence, or None when the input is fine
    run: Callable[[dict[str, Any], Any], str]            # (args, state) -> a short result; raises on failure
    describe: "Callable[[dict[str, Any], Any], str] | None" = None   # plain words for an approval question (Phase 2)


@dataclass
class TraceEntry:
    """One numbered line of the run's record."""
    step: int
    kind: TraceKind
    detail: str


@dataclass
class RunResult:
    """What a run hands back: how it ended, the answer, how many actions were refused, and the trace."""
    status: Status
    answer: str
    denials: int
    trace: list[TraceEntry]
    handover: str = ""


def handover_note(status: str, goal: str, trace: list[TraceEntry]) -> str:
    """Recovery: plain words a person can pick the work up from; empty when the run finished normally."""
    if status in ("done", "done-with-denials"):
        return ""
    last = trace[-1] if trace else None
    return (f"Stopped with status '{status}'. Goal: '{goal}'. Steps taken: {last.step if last else 0}. "
            f"Last step: {last.detail if last else 'none'}. "
            "A person should check the task list and decide whether to retry.")


def _args_text(args: dict[str, Any]) -> str:
    """Arguments as JSON with sorted keys, so key order never makes one call look like two."""
    return json.dumps(args, sort_keys=True)


def run(
    goal: str,
    *,
    model: Model,
    tools: list[Tool],
    state: Any,
    max_steps: int = 10,
    max_result_chars: int = 300,
    max_denials: int = 3,
    max_history: "int | None" = None,
    approve: "Callable[[str, dict[str, Any], str], bool] | None" = None,
    verify: "Callable[[Any], str | None] | None" = None,
    checkpoint: "Callable[[Any], None] | None" = None,
) -> RunResult:
    """Work towards the goal one model call at a time until the model says "final" or a limit is hit."""
    history: list[str] = [f"user: {goal}"]        # the goal is always the first line the model reads
    trace: list[TraceEntry] = []
    denials = 0
    asked: list[str] = []                         # every call the model asked for, valid or not, for the repeat check
    by_name = {tool.name: tool for tool in tools}

    def cut(text: str) -> str:
        """Context: shorten a result or trace line, with a visible marker so nobody mistakes it for the whole thing."""
        return text if len(text) <= max_result_chars else text[:max_result_chars] + " [truncated]"

    def log(step: int, kind: TraceKind, detail: str) -> None:
        """Trace: add one numbered line, cut to the same size as any result."""
        trace.append(TraceEntry(step, kind, cut(detail)))

    def view() -> list[str]:
        """Context: the lines the model is shown this turn (a fresh list, so the model cannot edit the real history)."""
        if max_history is None or len(history) <= max_history + 1:
            return list(history)
        tail = history[-max_history:] if max_history > 0 else []     # history[-0:] would be the whole list
        slid_out = history[1:len(history) - len(tail)]
        kept_denials = [line for line in slid_out if line.startswith("denied:")]
        return [history[0], *kept_denials, *tail]

    def finish(status: Status, answer: str) -> RunResult:
        """Recovery: the one way out of a run, so no early stop can forget its hand-over note."""
        return RunResult(status, answer, denials, trace, handover_note(status, goal, trace))

    for step in range(1, max_steps + 1):
        # ---- 1. ask the model (part 1, Loop): a broken or confused model ends the run, it never crashes it ----
        try:
            decision = model(view())
            if not isinstance(decision, dict) or not ("final" in decision or "tool" in decision):
                raise ValueError("malformed model output")
        except Exception as err:
            log(step, "stop", f"model error: {err}")
            return finish("model-error", "stopped: the model call failed")

        # ---- 2. "final" (part 1, Loop; part 6, Verify): the model says it is done. If a reply also names a
        #         tool, final wins. The claim is then checked against the real state, never taken on trust ----
        if "final" in decision:
            answer = str(decision["final"])
            log(step, "decide", f"final: {answer}")
            problem = None
            if verify is not None:
                try:
                    problem = verify(state)
                except Exception as err:          # a check that cannot run has not passed
                    problem = f"the check itself crashed: {err}"
            if problem:
                log(step, "stop", f"verification failed: {problem}")
                return finish("failed-verification", f"not done: {problem}")
            return finish("done-with-denials" if denials else "done", answer)

        # ---- 3. record the call (part 7, Trace): every request is traced before anything checks it ----
        name = str(decision["tool"])
        args = decision["args"] if isinstance(decision.get("args"), dict) else {}   # junk args mean "no arguments"
        call = f"{name} {_args_text(args)}"
        log(step, "decide", call)

        # ---- 4. repeat check (part 1, Loop): a model asking the same thing again is stuck, so stop it. It runs
        #         before the gate, so a refused delete asked three times ends here, not at the denial cap ----
        asked.append(call)                    # sorted-key JSON, so key order cannot disguise a repeat
        if asked[-6:].count(call) >= 3:
            log(step, "stop", "same call 3 times in the last 6 steps")
            return finish("loop-detected", "stopped: repeating the same action")
        history.append(f"assistant wants {call}")

        # ---- 5. look up and check (part 2, Tools): a wrong name or bad input goes back to the model as data ----
        tool = by_name.get(name)
        if tool is None:
            problem = f"unknown tool {name}"
        else:
            try:
                problem = tool.validate(args)
            except Exception as err:          # a buggy check must not take the whole run down
                problem = f"input check crashed: {err}"
        if problem:
            log(step, "error", f"error: {problem}")
            history.append(cut(f"error: {problem}"))
            continue

        # ---- 6. the gate (part 5, Gates): risk lives in code, so no prompt and no planted note can lower it ----
        if tool.risk == "irreversible":
            allowed = False                     # default deny: stays False unless a human really says yes
            try:
                words = tool.describe(args, state) if tool.describe else call   # the real effect, not just an id
                if approve is not None:
                    allowed = bool(approve(name, args, words))
            except Exception:
                allowed = False                 # a broken question or a broken approver is never a yes
            if not allowed:
                denials += 1
                log(step, "denied", call)
                # Not cut: the model must always read the whole refusal, including "Do not retry it."
                history.append(f"denied: {call} was not approved. Do not retry it.")
                if denials >= max_denials:      # denial cap (part 1, Loop): stop pestering the human
                    log(step, "stop", "too many denied actions")
                    return finish("denied-limit", "stopped: too many actions were denied")
                continue

        # ---- 7. run the tool (part 2, Tools): a failure is reported, never retried, because repeating a write
        #         that half-worked is worse than one honest error ----
        try:
            result = tool.run(args, state)
        except Exception as err:
            log(step, "error", f"error: {err}")
            history.append(cut(f"error: {err}"))
            continue
        log(step, "tool", result)
        history.append(f"tool: {cut(result)}")      # "tool:" marks a result as data the model reads, never as an order

        # ---- 8. checkpoint (part 4, Memory): save after anything that changed the list, so a crash or a new
        #         terminal loses nothing. A failed save stops the run: carrying on would pile up unsaved work ----
        if tool.risk != "read" and checkpoint is not None:
            try:
                checkpoint(state)
            except Exception as err:
                log(step, "stop", f"checkpoint failed: {err}")
                return finish("checkpoint-failed", "stopped: progress could not be saved")

    log(max_steps, "stop", "step limit reached")
    return finish("step-limit", "stopped: step limit reached")
