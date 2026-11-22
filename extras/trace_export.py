"""Trace export: show every step of a run in a trace viewer such as Langfuse, without editing harness.py.

The harness already records a trace (TraceEntry lines), and that is enough on a laptop. At work you also want
time per step, a tree of what happened, and a place the whole team can open. Trace viewers draw exactly that.
The harness code stays the same; only WHERE the lines go changes. This file does that by wrapping the three
things the harness already accepts as arguments: the model, the tools and the approve callback. So you never
edit harness.py just to add logging, and never risk the safety behaviour the tests protect.

What to look at
    Tracer          wrap_model(), wrap_tools() and wrap_approve() report each call as a span. If the exporter
                    fails, it warns once and the run carries on unchanged, so a broken viewer never breaks the agent.
    MAX_TEXT, _cut  every input and output is cut to MAX_TEXT characters before it leaves the program, so a run
                    cannot leak too much. The one exception is what the model was shown (MAX_PROMPT_TEXT): a generation
                    is only useful in the viewer if you can read the exact prompt that led to each decision.
    _prompt()       records each model call's input as chat messages (system, then user), the way the model saw it.
    choose_sink()   picks Langfuse when its keys and package are present, and falls back to printing the tree here.
    run_traced()    runs an engine exactly as before under one "run" span, and returns the same RunResult.
    state_session_id()
                    groups every run on the same saved task list into one Langfuse session, so you can replay how a
                    list changed across runs (memory, Phase 4). The id is a hash: your folder names never leave.
    Tracer(log=...) the live log (on by default; --quiet turns it off): the same wrappers also write one line per model reply, tool call
                    and approval while the run is going, through the callable you pass (print, or agent.py's out).

PRIVACY: spans carry prompts and task text. Send only toy data to a hosted viewer, never real tasks.

Standard library only. The Langfuse package is optional and imported lazily, so the rest of the repo needs no install.
"""
import dataclasses
import hashlib
import json
import os
import sys
import time
from contextlib import contextmanager
from typing import Any, Callable

from harness import RunResult, Tool, run
from tools import State

MAX_TEXT = 500          # every input and output is cut to this many characters before it is exported...
MAX_PROMPT_TEXT = 8000  # ...except each message the model was shown: a cut prompt hides why the model decided what it did
DEFAULT_ENVIRONMENT = "development"   # Langfuse environment for these runs; set LANGFUSE_TRACING_ENVIRONMENT to change it


def _cut(value: Any, limit: int = MAX_TEXT) -> Any:
    """Return a short, safe version of a value. Strings and JSON-like values are cut to `limit` characters."""
    text = value if isinstance(value, str) else json.dumps(value, default=str, sort_keys=True)
    return text if len(text) <= limit else text[:limit] + " [cut]"


def _prompt(history: list[str], system_prompt: str | None) -> list[dict[str, str]]:
    """A model call's input as role-labelled chat messages, exactly as llm.py sends them: system, then the history as one user message."""
    messages = [{"role": "system", "content": _cut(system_prompt, MAX_PROMPT_TEXT)}] if system_prompt else []
    return messages + [{"role": "user", "content": _cut("\n".join(history), MAX_PROMPT_TEXT)}]


def state_session_id(state_path: str) -> str:
    """A stable, private session id for one saved task list: runs on the same file share it, and the path itself never leaves."""
    digest = hashlib.sha256(os.path.abspath(state_path).encode("utf-8")).hexdigest()[:12]
    return f"todo-list-{digest}"


def _line(text: str) -> str:
    """Keep a live line on one line: newlines inside a reply or a result become " | ", as in the end-of-run trace."""
    return str(text).replace("\r", "").replace("\n", " | ")


def _count(n: int, word: str) -> str:
    """'1 line', '5 lines': plain words for a number of things."""
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _retry_words(facts: dict[str, Any]) -> str:
    """How hard llm.py had to try for one call, from its facts: ' after 3 attempts, waited 4.1s (HTTP 429)'."""
    text = f" after {facts.get('attempts', 1)} attempts"
    if facts.get("waited"):
        text += f", waited {facts['waited']}s"
    if facts.get("last_error"):
        text += f" ({facts['last_error']})"
    return text


def _last_call() -> dict[str, Any]:
    """A copy of llm.last_call (the facts about the latest model call), or {} when llm.py is not importable."""
    try:
        import llm
        return dict(getattr(llm, "last_call", {}))
    except Exception:
        return {}


# ---- Sinks: where spans go. A sink has span(kind, name, input) -> context manager yielding an object with .update(output=...) ----

class _Span:
    """One recorded step, kept in memory. Used by ListSink (tests) and ConsoleSink (the terminal)."""

    def __init__(self, kind: str, name: str, input: Any, depth: int, session_id: str | None = None):
        self.kind, self.name, self.input, self.depth, self.session_id = kind, name, input, depth, session_id
        self.output: Any = None
        self.seconds: float | None = None

    def update(self, output: Any = None, **extra: Any) -> None:
        """Attach the result of the step. Extra fields (level, metadata, model, usage_details) are kept for the tests."""
        if output is not None:
            self.output = output
        self.extra = {**getattr(self, "extra", {}), **extra}


class ListSink:
    """Keeps every span in a list. The offline tests read this list; nothing leaves the program."""

    def __init__(self) -> None:
        self.spans: list[_Span] = []
        self._depth = 0

    @contextmanager
    def span(self, kind: str, name: str, input: Any, session_id: str | None = None):
        record = _Span(kind, name, input, self._depth, session_id)
        self.spans.append(record)
        self._depth += 1
        start = time.perf_counter()
        try:
            yield record
        finally:
            record.seconds = round(time.perf_counter() - start, 4)
            self._depth -= 1

    def score(self, name: str, value: float, comment: str = "") -> None:
        """Record a score on the run (an 'eval' number). Tests read this list."""
        self.scores = [*getattr(self, "scores", []), (name, value)]

    def flush(self) -> None:
        """Nothing to send."""


class ConsoleSink(ListSink):
    """Same as ListSink, then prints the finished tree as indented lines (the no-key fallback for the live demo)."""

    def flush(self) -> None:
        for s in self.spans:
            print(f'{"  " * s.depth}{s.kind:<6} {s.name}  {s.seconds}s  out={_cut(s.output)}')
        self.spans.clear()


def langfuse_sink() -> Any | None:
    """Return a sink that sends spans to Langfuse, or None when the keys or the package are missing.

    Needs LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY (and LANGFUSE_BASE_URL for the region) in the
    environment (config.load_env() fills them from your .env), plus `pip install langfuse`. Written against
    langfuse 5.2.0 (`start_as_current_observation`); older v3 used different method names, so check the docs if a
    version change breaks it. The exporter never breaks the run: a failure only prints one warning.
    """
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        return None
    try:
        from langfuse import get_client, propagate_attributes
    except ImportError:
        return None
    # Demo runs are labelled "development" so they never mix with real traffic in shared dashboards; your shell or
    # .env wins if it sets LANGFUSE_TRACING_ENVIRONMENT (lowercase letters, digits, - and _ only).
    os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", DEFAULT_ENVIRONMENT)
    os.environ.setdefault("OTEL_SERVICE_NAME", "todo-agent")   # otherwise every span says "unknown_service:python"
    try:
        client = get_client()
    except Exception:                      # a bad key or setup must not stop the run: fall back to the console
        return None
    # Langfuse draws each observation type differently: the run is an agent, a model call is a generation (time and
    # tokens), a tool is a tool, and the approval gate is a guardrail.
    as_type = {"run": "agent", "model": "generation", "tool": "tool", "gate": "guardrail"}

    class LangfuseSink:
        @contextmanager
        def span(self, kind: str, name: str, input: Any, session_id: str | None = None):
            with client.start_as_current_observation(name=name, as_type=as_type.get(kind, "span"), input=input) as observation:
                if kind == "run":                  # a readable trace name, a tag to filter on, and the session it belongs to
                    with propagate_attributes(trace_name="todo-agent run", tags=["todo-agent"], session_id=session_id):
                        yield observation
                else:
                    yield observation

        def score(self, name: str, value: float, comment: str = "") -> None:
            """Attach a score to the trace that is currently open (call it inside the run span)."""
            client.score_current_trace(name=name, value=value, data_type="NUMERIC", comment=comment or None)

        def flush(self) -> None:
            client.flush()

    return LangfuseSink()


def choose_sink(name: str) -> Any:
    """Pick a sink by name: "langfuse" (falls back to the console when keys or package are missing) or "console"."""
    if name == "langfuse":
        sink = langfuse_sink()
        if sink is not None:
            print("export: sending spans to Langfuse. Prompts and task text leave this machine: toy data only.", file=sys.stderr)
            return sink
        if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
            print("export: no Langfuse keys in .env (see .env.example); printing the trace tree here instead.", file=sys.stderr)
        else:
            print("export: Langfuse keys found but the package is missing (pip install langfuse); printing the tree here instead.", file=sys.stderr)
    return ConsoleSink()


# ---- The tracer: wraps the model, the tools and the approver. harness.py is never edited. ----

class Tracer:
    """Wraps what the harness already takes as arguments, and reports each call to a sink."""

    def __init__(self, sink: Any, log: Callable[[str], Any] | None = None, *,
                 system_prompt: str | None = None, session_id: str | None = None):
        self.sink = sink
        self.log = log              # where live lines go (the live log); None keeps the run silent until it ends
        self.system_prompt = system_prompt   # the instructions the model reads (llm.build_system_prompt); None for fake models
        self.session_id = session_id         # groups runs on the same task list (state_session_id); None = no session
        self._warned = False
        self.retries = 0            # model-call retries seen so far (hidden inside llm.py otherwise)
        self.calls = 0              # model calls so far in this run: the step number on every live line

    def _warn(self, err: Exception) -> None:
        """A broken exporter must never break the agent: say so once, then carry on."""
        if not self._warned:
            self._warned = True
            print(f"trace export problem (the run is unaffected): {err}", file=sys.stderr)

    def _say(self, level: str, kind: str, text: str, step: int | None = None) -> None:
        """Write one live line (level, step, kind, text) when a log is set; a failing log never stops the run."""
        if self.log is None:
            return
        try:
            where = f"step {step:<2}" if step is not None else " " * 7
            self.log(f"{level:<5} {where} {kind:<6} {_line(text)}")
        except Exception as err:
            self._warn(err)

    def _say_reask(self, facts: dict[str, Any]) -> None:
        """If llm.py asked again after an invalid reply, say so and show that first reply."""
        if facts.get("reasked"):
            self._say("WARN", "model", f"first reply was not valid JSON, asked again. First reply: {_cut(facts.get('first_raw', ''))}", self.calls)

    def _say_model(self, decision: Any, facts: dict[str, Any], seconds: float) -> None:
        """The live lines for one model call that returned: a re-ask (if any), then the time, tokens and raw reply."""
        self._say_reask(facts)
        attempts = facts.get("attempts", 1)
        text = f"reply in {seconds:.1f}s"
        usage = facts.get("usage") or {}
        if "prompt_tokens" in usage and "completion_tokens" in usage:
            text += f" ({usage['prompt_tokens']} in / {usage['completion_tokens']} out tokens)"
        if attempts > 1:
            text += _retry_words(facts)
        raw = _cut(facts.get("raw") or json.dumps(decision, default=str))   # a fake model has no raw text: show its decision
        if not isinstance(decision, dict) or not ({"tool", "final"} & set(decision)):
            self._say("ERROR", "model", f'{text}, with neither "tool" nor "final" (the run stops): {raw}', self.calls)
        else:
            self._say("WARN" if attempts > 1 else "INFO", "model", f"{text}: {raw}", self.calls)

    def _set(self, span: Any, output: Any, **extra: Any) -> None:
        """Attach a result (and optional level, metadata, model, usage) to a span, ignoring any exporter failure."""
        try:
            if span is not None:
                span.update(output=_cut(output), **extra)
        except Exception as err:
            self._warn(err)

    def _call_facts_error(self) -> dict[str, Any]:
        """Like _call_facts, but for a call that failed: keep the retry facts, drop the usage and the WARNING level."""
        extra = self._call_facts()
        extra.pop("level", None)
        extra.pop("usage_details", None)
        return extra

    def _call_facts(self) -> dict[str, Any]:
        """What llm.py knows about the latest model call: the model name, token usage, and retries the harness never sees. Never raises."""
        try:
            return self._read_call_facts()
        except Exception as err:
            self._warn(err)
            return {}

    def _read_call_facts(self) -> dict[str, Any]:
        """The unguarded part of _call_facts."""
        try:
            import llm
        except ImportError:
            return {}
        facts = dict(getattr(llm, "last_call", {}))
        extra: dict[str, Any] = {}
        if facts.get("model"):
            extra["model"] = facts["model"]
        if facts.get("usage"):
            extra["usage_details"] = {"input": facts["usage"].get("prompt_tokens", 0), "output": facts["usage"].get("completion_tokens", 0)}
        meta = {k: facts[k] for k in ("attempts", "waited", "last_error", "reasked") if k in facts}
        if "first_raw" in facts:                       # the invalid first reply, kept when llm.py had to ask again
            meta["first_raw"] = _cut(facts["first_raw"])
        if meta:
            extra["metadata"] = meta
        if facts.get("attempts", 1) > 1:
            self.retries += facts["attempts"] - 1
            extra["level"] = "WARNING"                 # it worked, but only after a retry: worth a glance
        return extra

    @contextmanager
    def span(self, kind: str, name: str, input: Any):
        """Open a span; if the sink fails to open or close it, the wrapped call still runs normally."""
        manager = span = None
        try:
            # A model call's input is already a list of cut messages (_prompt); everything else is cut here.
            safe = input if kind == "model" else _cut(input)
            session = {"session_id": self.session_id} if kind == "run" and self.session_id else {}
            manager = self.sink.span(kind, name, safe, **session)
            span = manager.__enter__()
        except Exception as err:
            self._warn(err)
            manager = None
        try:
            yield span
        finally:
            if manager is not None:
                try:
                    manager.__exit__(None, None, None)
                except Exception as err:
                    self._warn(err)

    def wrap_model(self, model: Callable[[list[str]], dict]) -> Callable[[list[str]], dict]:
        """One "model" span (a Langfuse generation) per model call: the exact prompt it was shown, and what it decided."""
        def traced(history: list[str]) -> dict:
            """Call the model inside a span, and write its live lines (before the call and after it)."""
            self.calls += 1                        # raised before the call, so a failed call still has its own step
            # The name stays the same on every call (Langfuse groups, filters and evaluates by name); the step number
            # and the history size go in metadata instead.
            where = {"step": self.calls, "lines_shown": len(history)}
            with self.span("model", "decide-next-step", _prompt(history, self.system_prompt)) as sp:
                try:
                    import llm
                    llm.last_call.clear()          # so a fake model never inherits facts from an earlier call
                except ImportError:
                    pass
                # Said before the call, so a slow model (up to 70 s of retries inside llm.py) is not a silent wait.
                self._say("INFO", "model", f"asking the model (shown {_count(len(history), 'line')})", self.calls)
                start = time.perf_counter()
                try:
                    decision = model(history)
                except Exception as err:
                    # Read the facts once: _call_facts adds to self.retries, so a second read would double the score.
                    facts, extra = _last_call(), self._call_facts_error()
                    extra["metadata"] = {**where, **extra.get("metadata", {})}
                    self._set(sp, f"error: {err}", level="ERROR", status_message=str(err)[:200], **extra)
                    self._say_reask(facts)
                    retry = _retry_words(facts) if facts.get("attempts", 1) > 1 else ""
                    self._say("ERROR", "model", f"call failed{retry}: {_cut(str(err))}", self.calls)
                    raise  # the harness handles it exactly as before (status model-error)
                seconds = time.perf_counter() - start
                facts, extra = _last_call(), self._call_facts()
                bad = not isinstance(decision, dict) or not ({"tool", "final"} & set(decision))
                # Langfuse shows what the model actually said (the raw text); the parsed decision rides along as metadata.
                meta = {**where, **extra.get("metadata", {}), "decision": _cut(decision)}
                self._set(sp, facts.get("raw") or decision, **{**extra, "metadata": meta},
                          **({"level": "ERROR", "status_message": "malformed model output"} if bad else {}))
                self._say_model(decision, facts, seconds)
                return decision
        return traced

    def wrap_tools(self, tools: list[Tool]) -> list[Tool]:
        """One "tool" span per tool that really runs. A denied call never reaches run(), so it has no tool span."""
        def wrap(tool: Tool) -> Tool:
            """Wrap one tool's run and input check."""
            def traced_run(args: dict[str, Any], state: State) -> str:
                """Run the tool inside a span, and write its live line."""
                call = f"{tool.name} {json.dumps(args, sort_keys=True, default=str)}"
                with self.span("tool", tool.name, args) as sp:
                    try:
                        out = tool.run(args, state)
                    except Exception as err:
                        self._set(sp, f"error: {err}", level="WARNING", status_message=str(err)[:200])   # the model gets this as data
                        self._say("WARN", "tool", f"{call}  ->  error: {_cut(str(err))}", self.calls)
                        raise
                    self._set(sp, out)
                    self._say("INFO", "tool", f"{call}  ->  {_cut(out)}", self.calls)
                    return out

            def traced_validate(args: dict[str, Any]) -> str | None:
                """Check the input as before; a rejection gets a live line, and a crash is passed on to the harness."""
                call = f"{tool.name} {json.dumps(args, sort_keys=True, default=str)}"
                try:
                    problem = tool.validate(args)
                except Exception as err:
                    self._say("WARN", "tool", f"{call}  input check crashed: {_cut(str(err))}", self.calls)
                    raise                  # the harness turns it into "input check crashed", exactly as before
                if problem is not None:
                    self._say("WARN", "tool", f"{call}  input rejected: {problem}", self.calls)
                return problem
            return dataclasses.replace(tool, run=traced_run, validate=traced_validate)
        return [wrap(t) for t in tools]

    def wrap_approve(self, approve: Callable[[str, dict[str, Any], str], bool] | None):
        """One "gate" span per approval question, with the human's answer. No approver stays no approver."""
        if approve is None:
            return None

        def traced(name: str, args: dict[str, Any], words: str) -> bool:
            """Ask the approver inside a span, and write the question and the answer as a live line."""
            with self.span("gate", f"approve {name}", words) as sp:
                try:
                    answer = bool(approve(name, args, words))
                except Exception as err:
                    self._set(sp, "denied (the approver crashed)", level="WARNING", status_message=str(err)[:200])
                    self._say("WARN", "gate", f"{words}  ->  denied (the approver crashed: {_cut(str(err))})", self.calls)
                    raise                  # the harness treats a crash as a refusal, exactly as before
                self._set(sp, "approved" if answer else "denied", **({} if answer else {"level": "WARNING", "status_message": "the human said no"}))
                self._say("INFO" if answer else "WARN", "gate", f"{words}  ->  {'approved' if answer else 'denied'}", self.calls)
                return answer
        return traced

    def score_run(self, result: RunResult, finished: bool) -> None:
        """Attach four scores to the run: outcome (1 = finished normally), denials, steps (model calls), model_retries (HTTP retries and JSON re-asks inside llm.py)."""
        try:
            for name, value in (("outcome", 1.0 if finished else 0.0), ("denials", float(result.denials)),
                                ("steps", float(steps_used(result))), ("model_retries", float(self.retries))):
                self.sink.score(name, value, result.status)
        except Exception as err:
            self._warn(err)

    def flush(self) -> None:
        """Send anything still waiting (Langfuse batches in the background)."""
        try:
            self.sink.flush()
        except Exception as err:
            self._warn(err)


def steps_used(result: RunResult) -> int:
    """How many steps the run took: the highest step number in its trace."""
    return max((e.step for e in result.trace), default=0)


def run_traced(goal: str, tracer: Tracer, *, model: Callable, tools: list[Tool], state: State,
               approve: Callable | None = None, engine: Callable = run, **limits: Any) -> RunResult:
    """Run an engine (harness.run by default, or graph.run_graph) exactly as before, under one "run" span. Same RunResult."""
    tracer.calls = 0
    with tracer.span("run", "todo-agent run", goal) as sp:
        result = engine(goal, model=tracer.wrap_model(model), tools=tracer.wrap_tools(tools), state=state,
                     approve=tracer.wrap_approve(approve), **limits)
        finished = result.status in ("done", "done-with-denials")
        # The run's output becomes the trace output in Langfuse's table: what a reviewer needs at a glance is the answer
        # (or, for a run that stopped early, the hand-over note). The counts ride along as metadata.
        tracer._set(sp, result.answer or result.handover or f"(no answer, status {result.status})",
                    **({} if finished else {"level": "ERROR", "status_message": result.handover[:300]}),
                    metadata={"engine": engine.__name__, "status": result.status, "steps": steps_used(result),
                              "denials": result.denials, "model_retries": tracer.retries})
        tracer.score_run(result, finished)             # small numbers you can filter and chart in the viewer
    # Stops such as the repeat check or the step cap are decided inside the engine, so the trace's last line says why.
    last = result.trace[-1].detail if result.trace else ""     # a custom engine may return an empty trace
    tracer._say("INFO" if finished else "WARN", "run", f"ended: status={result.status} after {_count(tracer.calls, 'model call')}. Last: {_cut(last)}")
    tracer.flush()
    return result
