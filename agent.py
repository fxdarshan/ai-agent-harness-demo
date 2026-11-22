"""The Desk Assistant: the command-line app that joins every part into one program you can run.

harness.py is the engine, tools.py the to-do app, store.py the saved file, llm.py the model, and graph.py the
same engine in LangGraph. None of them runs on its own: something has to read your command line, load the list,
build the model, ask you the approval questions, run the goal and print what happened. That is this file.

It stays thin on purpose. A safety rule written here would be invisible to the harness tests and would not apply
to try_it.py or the scenario runner, so every rule (the gate, the limits, verification, saving) lives in
harness.py, tools.py and store.py, and this file only wires them together.

What to look at
    build_parser()    every flag in one place, so you can see everything a run can be told to do.
    make_approver()   turns --approve into the harness's approval callback. "ask" shows you the real effect
                      (the task text) and only "y" counts as yes; anything else, or no answer, is a no.
    load_starting_state()
                      memory: a saved file wins, and --seed is used only for the very first run. A damaged file
                      or a missing seed is a setup problem, never a silent fresh start.
    main()            builds the model, picks the engine, wires the checkpoint hook (save after every change),
                      runs the goal through the live log or the --export viewer, and prints the trace.
    save_seed_if_unsaved()
                      a run that changed nothing never triggers the checkpoint, so a list started from --seed
                      would be forgotten and the next run would start empty. This saves that starting list once.
    print_result()    the end-of-run format the scenario runner reads: one line per trace entry, then the status.
    Exit codes        0 finished (done or done-with-denials), 1 stopped early, 2 a setup problem, so a script can
                      tell the three apart.
    __main__          the only place that reads .env, so tests that call main() never depend on your keys.
"""
import argparse
import os
import sys

import config
from harness import RunResult, run
from store import load_state, save_state
from tools import ALL_TOOLS, State, make_oldest_verifier, new_state

ARROWS = {"decide": "model  ->", "tool": "tool   <-", "denied": "gate   X ", "error": "error  ! ",
          "stop": "stop   # ", "pause": "gate   ? "}


def build_parser() -> argparse.ArgumentParser:
    """All command-line flags, with the defaults from the spec."""
    p = argparse.ArgumentParser(description="Desk Assistant: a to-do agent with a harness around it.")
    p.add_argument("goal", help="what you want done, in plain words")
    p.add_argument("--approve", choices=["ask", "yes", "no"], default="ask", help="who answers approval questions")
    p.add_argument("--state", default="tasks.json", help="where the task list is saved")
    p.add_argument("--seed", help="a starting list, used only when the state file does not exist yet")
    p.add_argument("--engine", choices=["plain", "langgraph"], default="plain")
    p.add_argument("--export", choices=["console", "langfuse"], help="also draw the run as a tree")
    p.add_argument("--quiet", action="store_true", help="no live log while the run goes")
    p.add_argument("--naive-prompt", action="store_true", help="drop the prompt's warning about tool results (drill)")
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--max-history", type=int, default=None, help="show the model only the last N history lines")
    p.add_argument("--max-result-chars", type=int, default=300)
    p.add_argument("--verify-oldest", type=int, metavar="N", help="after 'final', check exactly the N oldest are gone")
    return p


def make_approver(mode: str, input_fn, out):
    """Turn --approve into the harness's approve(name, args, words) callback."""
    def approve(_name, _args, words):
        """Answer one approval question; only an explicit 'y' is a yes."""
        if mode != "ask":
            return mode == "yes"
        out(f"APPROVAL NEEDED: {words}. Cannot be undone. Did you ask for this, or did text in a task? "
            "No = the agent is told and will not retry; 3 refusals stop the run.")
        try:
            return input_fn("Allow this? [y/N] ").strip().lower().startswith("y")
        except EOFError:                       # no human at the keyboard: no answer means no
            return False
    return approve


def load_starting_state(state_path: str, seed_path: "str | None") -> State:
    """Memory: the saved list if there is one, else the seed, else an empty list. Raises ValueError on a bad file."""
    if os.path.exists(state_path):
        return load_state(state_path)          # the saved list wins: a seed must never overwrite real work
    if seed_path:
        if not os.path.exists(seed_path):
            raise ValueError(f"seed file not found: {seed_path}")
        return load_state(seed_path)
    return new_state()


def save_seed_if_unsaved(result: RunResult, state: State, state_path: str, seed_path: "str | None") -> None:
    """Memory: keep a seeded list that no change in this run has saved yet, so the next run still remembers it."""
    if not seed_path or os.path.exists(state_path):
        return                                 # no seed, or the checkpoint already saved: nothing to add
    if result.status == "checkpoint-failed":
        return                                 # that save already failed once; writes are never retried
    save_state(state, state_path)              # nothing changed, so this is exactly the seed, as a saved file


def print_result(result: RunResult, state: State, out) -> None:
    """The end-of-run summary in the exact format of the spec (the scenario runner reads it)."""
    for entry in result.trace:
        out(f"step {entry.step:<2} {ARROWS[entry.kind]} {entry.detail.replace(chr(10), ' | ')}")
    out(f"status: {result.status}")
    out(f"answer: {result.answer}")
    if result.handover:
        out(f"hand-over note: {result.handover}")
    out(f"tasks left: {len(state.tasks)}")
    for task in state.tasks:
        out(f"  {task.id}: {task.text}")


def main(argv=None, *, model=None, input_fn=input, out=print) -> int:
    """Run one goal from the command line. `model`, `input_fn` and `out` can be replaced in tests. Returns the exit code."""
    args = build_parser().parse_args(argv)
    try:
        state = load_starting_state(args.state, args.seed)
    except ValueError as err:
        out(f"configuration error: {err}")
        return 2

    system_prompt = None                       # known only for the real model; the trace export shows it
    if model is None:
        import llm
        try:
            model = llm.make_model(ALL_TOOLS, naive=args.naive_prompt)
        except RuntimeError as err:            # no key, for example: a plain sentence, not a traceback
            out(f"configuration error: {err}")
            return 2
        system_prompt = llm.build_system_prompt(ALL_TOOLS, naive=args.naive_prompt)

    options = dict(model=model, tools=ALL_TOOLS, state=state, approve=make_approver(args.approve, input_fn, out),
                   max_steps=args.max_steps, max_result_chars=args.max_result_chars)
    if args.verify_oldest is not None:
        options["verify"] = make_oldest_verifier(state, args.verify_oldest)
    if args.engine == "langgraph":
        from graph import run_graph            # imported only here, so the plain engine needs no LangGraph install
        engine = run_graph
        if args.max_history is not None:
            out("note: --max-history is not used by the langgraph engine")
    else:
        engine = run
        options.update(max_history=args.max_history, checkpoint=lambda s: save_state(s, args.state))

    if args.quiet and not args.export:
        result = engine(args.goal, **options)
    else:                                      # the live log and --export both wrap the run; the harness is unchanged
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "extras"))
        from trace_export import ListSink, Tracer, choose_sink, run_traced, state_session_id
        tracer = Tracer(choose_sink(args.export) if args.export else ListSink(), log=None if args.quiet else out,
                        system_prompt=system_prompt, session_id=state_session_id(args.state))
        result = run_traced(args.goal, tracer, engine=engine, **options)

    try:
        if args.engine == "langgraph":         # this engine has no checkpoint hook, so save once at the end
            save_state(state, args.state)
        else:
            save_seed_if_unsaved(result, state, args.state, args.seed)
    except OSError as err:                     # a lost save is said out loud, never swallowed
        out(f"error: the task list could not be saved to {args.state}: {err}")
        print_result(result, state, out)
        return 1

    print_result(result, state, out)
    return 0 if result.status in ("done", "done-with-denials") else 1


if __name__ == "__main__":
    config.load_env()                          # only the command line reads .env
    sys.exit(main())
