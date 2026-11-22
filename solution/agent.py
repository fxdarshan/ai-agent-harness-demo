"""The Desk Assistant: the command-line app that joins every part, and the one file with no rules of its own.

harness.py is the engine, tools.py is the to-do app, store.py is the saved file and llm.py is the model. Nothing runs
until something joins them and reads your command line, and that is this file. It stays thin on purpose: a safety rule
written here would be invisible to the tests, so every rule stays in the harness where the tests protect it.

What to look at, and why each piece is here (line numbers are for this file as shipped)
    build_parser (line 40)
        Every flag in one place, so you can see everything a run can be told to do.
    make_approver (line 58)
        Turns --approve into the harness's approval callback, and shows the human the real effect of what they are
        being asked to approve.
    main (line 71)
        Loads memory first: a saved file wins, and the seed is only the very first start (78). Unless --quiet,
        it wraps the model, tools and approver so a live log prints while the run goes (103), and hands the trace
        export the real system prompt and a session id for this task list (107). Then it picks the engine (109),
        wires the checkpoint hook that saves after every change (119), runs, and prints the trace (125).
    Exit codes (line 134; setup problems at 86 and 95)
        0 finished, 1 stopped early, 2 a setup problem, so a script can tell the three apart.
    Reading .env (line 138)
        Only the command line does it, so tests that call main() stay predictable.

Examples:
    python3 agent.py "Add 'File taxes', then show my list" --approve yes
    python3 agent.py "Delete the dentist task" --approve ask
    python3 agent.py "Tidy up my to-do list" --seed data/seeds/trap.json --approve no
"""
import argparse
import os
import sys

import config
from harness import run
from store import load_state, save_state
from tools import ALL_TOOLS, make_oldest_verifier, new_state

ARROWS = {"decide": "model  ->", "tool": "tool   <-", "denied": "gate   X ", "error": "error  ! ", "stop": "stop   # ", "pause": "gate   ? "}


def build_parser() -> argparse.ArgumentParser:
    """All command-line flags in one place."""
    p = argparse.ArgumentParser(description="Desk Assistant: a to-do agent with a harness around it.")
    p.add_argument("goal", help="what you want done, in plain words")
    p.add_argument("--approve", choices=["ask", "yes", "no"], default="ask", help="who answers approval questions (default: ask you)")
    p.add_argument("--state", default="tasks.json", help="where the task list is saved (default: tasks.json)")
    p.add_argument("--seed", help="a starting task list (a JSON file); used only when the state file does not exist yet")
    p.add_argument("--engine", choices=["plain", "langgraph"], default="plain")
    p.add_argument("--export", choices=["console", "langfuse"], help="also draw the run as a tree (extras/trace_export.py)")
    p.add_argument("--quiet", action="store_true", help="turn off the live log of every model reply, tool call and approval")
    p.add_argument("--naive-prompt", action="store_true", help="drop the prompt's warning about tool results (injection drill)")
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--max-history", type=int, default=None, help="show the model only the last N history lines")
    p.add_argument("--max-result-chars", type=int, default=300)
    p.add_argument("--verify-oldest", type=int, metavar="N", help="after 'final', check that exactly the N oldest tasks are gone")
    return p


def make_approver(mode: str, input_fn, out):
    """Turn --approve into the harness's approve(name, args, words) callback."""
    def approve(_name, _args, words):
        """Answer one approval question."""
        if mode == "yes":
            return True
        if mode == "no":
            return False
        out(f"APPROVAL NEEDED: {words}. Cannot be undone. Did you ask for this, or did text in a task? No = the agent is told and will not retry; 3 refusals stop the run.")
        return input_fn("Allow this? [y/N] ").strip().lower().startswith("y")
    return approve


def main(argv=None, *, model=None, input_fn=input, out=print) -> int:
    """Run one goal. `model` and `input_fn` can be replaced in tests. Returns the exit code."""
    args = build_parser().parse_args(argv)

    # Memory: a saved file wins. The seed is only the first-ever starting point.
    try:
        if os.path.exists(args.state):
            state = load_state(args.state)
        elif args.seed:
            if not os.path.exists(args.seed):
                raise ValueError(f"seed file not found: {args.seed}")
            state = load_state(args.seed)
        else:
            state = new_state()
    except ValueError as err:
        out(f"configuration error: {err}")
        return 2

    system_prompt = None                         # only a real model has one; the trace export shows it on every model call
    if model is None:
        import llm
        try:
            model = llm.make_model(ALL_TOOLS, naive=args.naive_prompt)
        except RuntimeError as err:
            out(f"configuration error: {err}")
            return 2
        system_prompt = llm.build_system_prompt(ALL_TOOLS, naive=args.naive_prompt)

    limits = dict(max_steps=args.max_steps, max_result_chars=args.max_result_chars)
    common = dict(model=model, tools=ALL_TOOLS, state=state, approve=make_approver(args.approve, input_fn, out),
                  verify=make_oldest_verifier(state, args.verify_oldest) if args.verify_oldest else None)
    tracer = None
    if args.export or not args.quiet:            # the live log and --export both wrap the model, tools and approver
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "extras"))
        from trace_export import ListSink, Tracer, choose_sink, run_traced, state_session_id
        # Runs on the same state file share one Langfuse session, so the viewer can replay how that list changed.
        tracer = Tracer(choose_sink(args.export) if args.export else ListSink(), log=None if args.quiet else out,
                        system_prompt=system_prompt, session_id=state_session_id(args.state))
    if args.engine == "langgraph":
        if args.max_history is not None:
            out("note: --max-history is not used by the langgraph engine")
        from graph import run_graph
        if tracer:
            result = run_traced(args.goal, tracer, engine=run_graph, **common, **limits)
        else:
            result = run_graph(args.goal, **common, **limits)
        save_state(state, args.state)               # this engine has no checkpoint hook, so save once at the end
    else:
        engine_args = dict(common, max_history=args.max_history, checkpoint=lambda s: save_state(s, args.state))
        if tracer:
            result = run_traced(args.goal, tracer, **engine_args, **limits)
        else:
            result = run(args.goal, **engine_args, **limits)

    for entry in result.trace:
        out(f"step {entry.step:<2} {ARROWS[entry.kind]} {entry.detail.replace(chr(10), ' | ')}")
    out(f"status: {result.status}")
    out(f"answer: {result.answer}")
    if result.handover:
        out(f"hand-over note: {result.handover}")
    out(f"tasks left: {len(state.tasks)}")
    for task in state.tasks:
        out(f"  {task.id}: {task.text[:80]}")
    return 0 if result.status in ("done", "done-with-denials") else 1


if __name__ == "__main__":
    config.load_env()          # only the command line reads .env; main() itself reads the environment, so tests stay predictable
    sys.exit(main())
