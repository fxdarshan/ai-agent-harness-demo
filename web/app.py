"""A small web app that lets anyone try the Desk Assistant in a browser, with a real Approve / Deny button at the gate.

agent.py needs a terminal and your own Gemini key, so only you can try it. This file puts the very same harness
behind a web page: a visitor types a goal, watches each step arrive, and answers the approval question with a
button. Nothing about the safety changes: every run goes through harness.run with the same tools, gate and limits.

Putting an agent on the open web adds risks a terminal never had, and this file handles each one:
    your key        it stays on the server (an environment variable); no page or response ever contains it.
    your quota      caps on runs per visitor, runs per day and goal length, so strangers cannot drain it.
    other people    every visitor gets their own in-memory task list; nobody can touch anyone else's list.
    no answer       an approval question left unanswered for APPROVAL_TIMEOUT seconds counts as no (default deny).

What to look at
    Session          one visitor: their task list, the events of the current run, and the open approval question.
    Session.ask()    the approve callback the harness calls. It blocks the run's thread until the visitor clicks,
                     or the timeout says no. The harness never knows a web page is involved.
    watch_model(), watch_tools()
                     record each model reply and tool result as an event the page polls for. The model call is
                     made under MODEL_LOCK, because llm.py keeps the latest call's facts in one shared place.
    Handler          the JSON endpoints (/api/session, /api/run, /api/poll, /api/answer, /api/reset) and the page.
                     /api/poll?wait=1 is a long poll: it answers as soon as there is news, so the page needs no timer.
    page_bytes()     the page is sent gzipped with an ETag, so a repeat visit downloads nothing.
    DEMO_MODEL       "gemini" (needs GEMINI_API_KEY) or "stand-in" (the scripted fooled model, no key, no quota).

Run locally:  python web/app.py            then open http://localhost:8000
Environment:  PORT, DEMO_MODEL, DAILY_RUN_CAP, RUNS_PER_VISITOR, APPROVAL_TIMEOUT (see the README section).
"""
import dataclasses
import gzip
import hashlib
import json
import os
import secrets
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tests")]

import config  # noqa: E402

config.load_env(os.path.join(ROOT, ".env"))       # a local .env if present; on a host the variables are set directly

from harness import run                            # noqa: E402
from store import load_state                       # noqa: E402
from tools import ALL_TOOLS                        # noqa: E402

DEMO_MODEL = os.environ.get("DEMO_MODEL", "gemini" if os.environ.get("GEMINI_API_KEY") or os.environ.get("LLM_API_KEY") else "stand-in")
DAILY_RUN_CAP = int(os.environ.get("DAILY_RUN_CAP", "150"))       # all visitors together, per UTC day
RUNS_PER_VISITOR = int(os.environ.get("RUNS_PER_VISITOR", "12"))   # per session
APPROVAL_TIMEOUT = int(os.environ.get("APPROVAL_TIMEOUT", "120"))  # seconds; no answer means no
MAX_GOAL_CHARS = 200
MAX_STEPS = 8                                      # a little under the CLI's 10: a public demo spends less per run
SESSION_IDLE_SECONDS = 30 * 60
MAX_SESSIONS = 500
SEEDS = {
    "week_start": ("A normal week", os.path.join(ROOT, "data", "seeds", "week_start.json")),
    "trap": ("A list with a planted note", os.path.join(ROOT, "data", "seeds", "trap.json")),
}
MODEL_LOCK = threading.Lock()                      # llm.py keeps the latest call's facts globally: one call at a time
SESSIONS: dict[str, "Session"] = {}
SESSIONS_LOCK = threading.Lock()
DAILY = {"day": "", "runs": 0}


LONG_POLL_SECONDS = 20                             # a waiting poll returns after this long even if nothing happened
_PAGE: dict = {}


def page_bytes() -> tuple[bytes, bytes, str]:
    """The page, its gzip copy and an ETag, rebuilt only when index.html changes on disk."""
    path = os.path.join(os.path.dirname(__file__), "index.html")
    stamp = os.path.getmtime(path)
    if _PAGE.get("stamp") != stamp:
        raw = open(path, "rb").read()
        # A weak ETag (W/"...") because proxies such as Cloudflare in front of Render drop strong ETags from
        # compressed replies; Last-Modified is a second validator for the same reason.
        _PAGE.update(stamp=stamp, raw=raw, gz=gzip.compress(raw, 9), etag='W/"%s"' % hashlib.sha256(raw).hexdigest()[:16],
                     modified=time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime(stamp)))
    return _PAGE["raw"], _PAGE["gz"], _PAGE["etag"]


def tasks_of(state) -> list[list]:
    """The list as [id, text] pairs, the shape the page draws."""
    return [[t.id, t.text] for t in state.tasks]


class Session:
    """One visitor: their own task list, the events of their runs, and an open approval question, if any."""

    def __init__(self, seed: str = "week_start"):
        self.id = secrets.token_urlsafe(16)
        self.lock = threading.Lock()
        self.changed = threading.Condition(self.lock)   # wakes a waiting poll the moment something new happens
        self.answered = threading.Event()
        self.reset(seed)
        self.runs = 0

    def reset(self, seed: str) -> None:
        """Start again from a fresh copy of a seed list."""
        self.seed = seed if seed in SEEDS else "week_start"
        self.state = load_state(SEEDS[self.seed][1])
        self.events: list[dict] = []
        self.running = False
        self.question: dict | None = None
        self.answer = False
        self.touched = time.time()

    def push(self, kind: str, **data) -> None:
        """Add one event for the page to pick up on its next poll."""
        with self.lock:
            self.events.append({"n": len(self.events), "kind": kind, "t": round(time.time(), 2), **data})
            self.changed.notify_all()

    def wait_for_news(self, since: int, seconds: float) -> None:
        """Long polling: hold a poll until there is a new event or the run ends, so the page gets news at once."""
        with self.changed:
            self.changed.wait_for(lambda: len(self.events) > since or not self.running, timeout=seconds)

    def ask(self, name: str, args: dict, words: str) -> bool:
        """The harness's approve callback: wait for the visitor's click. No answer in time means no."""
        self.answered.clear()
        with self.lock:
            self.question = {"words": words, "tool": name, "deadline": time.time() + APPROVAL_TIMEOUT}
        self.push("ask", words=words, timeout=APPROVAL_TIMEOUT)
        clicked = self.answered.wait(APPROVAL_TIMEOUT)
        with self.lock:
            allowed = bool(clicked and self.answer)
            self.question = None
        self.push("gate", words=words, allowed=allowed, reason="clicked" if clicked else "timed out")
        return allowed


def make_model():
    """The model for this server: real Gemini through llm.py, or the scripted stand-in that obeys planted notes."""
    if DEMO_MODEL == "stand-in":
        from fakes import gullible
        return gullible, None
    import llm
    return llm.make_model(ALL_TOOLS), llm


def watch_model(model, llm_module, session: Session):
    """Wrap the model so each reply becomes an event with its raw text, time and tokens."""
    def watched(history: list[str]) -> dict:
        """Call the model under the lock and record what it said."""
        session.push("thinking", lines=len(history))
        start = time.time()
        with MODEL_LOCK:
            try:
                decision = model(history)
                facts = dict(getattr(llm_module, "last_call", {})) if llm_module else {}
            except Exception as err:
                session.push("model_error", text=str(err)[:300])
                raise
        usage = facts.get("usage") or {}
        session.push("model", raw=(facts.get("raw") or json.dumps(decision))[:1200], seconds=round(time.time() - start, 1),
                     tokens_in=usage.get("prompt_tokens"), tokens_out=usage.get("completion_tokens"),
                     attempts=facts.get("attempts", 1), waited=facts.get("waited", 0), last_error=facts.get("last_error"))
        return decision
    return watched


def watch_tools(session: Session):
    """Wrap every tool so each result (or error) becomes an event. The tools themselves are unchanged."""
    def wrap(tool):
        """Record one tool's input and output."""
        def traced(args, state):
            """Run the tool and report the outcome."""
            call = f"{tool.name} {json.dumps(args, sort_keys=True)}"
            try:
                out = tool.run(args, state)
            except Exception as err:
                session.push("tool", call=call, result=f"error: {err}", ok=False, tasks=tasks_of(state))
                raise
            session.push("tool", call=call, result=out[:1200], ok=True, tasks=tasks_of(state))
            return out
        return dataclasses.replace(tool, run=traced)
    return [wrap(t) for t in ALL_TOOLS]


def run_goal(session: Session, goal: str) -> None:
    """The run's thread: the same harness.run as agent.py, with the visitor's list and the web approver."""
    try:
        model, llm_module = make_model()
        result = run(goal, model=watch_model(model, llm_module, session), tools=watch_tools(session),
                     state=session.state, approve=session.ask, max_steps=MAX_STEPS)
        session.push("end", status=result.status, answer=result.answer, handover=result.handover,
                     denials=result.denials, trace=[[e.step, e.kind, e.detail] for e in result.trace],
                     tasks=tasks_of(session.state))
    except Exception as err:                      # a setup problem (no key, say): tell the page, never crash the server
        session.push("end", status="setup-error", answer=f"The demo could not start: {err}", handover="", denials=0,
                     trace=[], tasks=tasks_of(session.state))
    finally:
        with session.lock:
            session.running = False
            session.changed.notify_all()


def daily_allows() -> bool:
    """Count one run against the shared daily cap; False once the cap is reached."""
    day = time.strftime("%Y-%m-%d", time.gmtime())
    with SESSIONS_LOCK:
        if DAILY["day"] != day:
            DAILY.update(day=day, runs=0)
        if DAILY["runs"] >= DAILY_RUN_CAP:
            return False
        DAILY["runs"] += 1
        return True


def forget_idle_sessions() -> None:
    """Drop visitors who left long ago, so memory stays bounded."""
    cutoff = time.time() - SESSION_IDLE_SECONDS
    with SESSIONS_LOCK:
        for sid in [sid for sid, s in SESSIONS.items() if s.touched < cutoff and not s.running]:
            del SESSIONS[sid]


class Handler(BaseHTTPRequestHandler):
    """The page and its five JSON endpoints."""
    server_version = "DeskAssistantDemo/1"

    def log_message(self, fmt, *args):
        """Keep the server log to one short line per request, without query strings (they carry session ids)."""
        sys.stderr.write(f"{self.command} {urlparse(self.path).path} {args[1] if len(args) > 1 else ''}\n")

    def send_json(self, code: int, body: dict) -> None:
        """Reply with JSON and no caching."""
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def read_json(self) -> dict:
        """The request body as a dict; anything unreadable is an empty dict."""
        try:
            length = min(int(self.headers.get("Content-Length", 0)), 4096)
            return json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, OSError):
            return {}

    def session(self, sid) -> "Session | None":
        """Look up a visitor by id and mark them active."""
        with SESSIONS_LOCK:
            s = SESSIONS.get(sid or "")
        if s:
            s.touched = time.time()
        return s

    def do_HEAD(self):
        """Answer uptime monitors and link previews that only ask whether the page is there."""
        path = urlparse(self.path).path
        ok = path in ("/", "/index.html", "/healthz")
        self.send_response(200 if ok else 404)
        self.send_header("Content-Type", "text/html; charset=utf-8" if ok else "application/json")
        if path in ("/", "/index.html"):                  # the same validators a GET sends, for caches and monitors
            _, _, etag = page_bytes()
            self.send_header("ETag", etag)
            self.send_header("Last-Modified", _PAGE["modified"])
        self.end_headers()

    def do_GET(self):
        """The page, a health check, or a poll for new events."""
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            page, packed, etag = page_bytes()
            modified = _PAGE["modified"]
            sent = self.headers.get("If-None-Match", "")
            if (sent and sent.replace("W/", "") == etag.replace("W/", "")) or (not sent and self.headers.get("If-Modified-Since") == modified):
                self.send_response(304)                    # the browser's copy is current: send nothing
                self.send_header("ETag", etag)
                self.send_header("Last-Modified", modified)
                self.end_headers()
                return
            body = packed if "gzip" in self.headers.get("Accept-Encoding", "") else page
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")   # always check, but reuse the copy when unchanged
            self.send_header("ETag", etag)
            self.send_header("Last-Modified", modified)
            if body is packed:
                self.send_header("Content-Encoding", "gzip")
            self.send_header("Vary", "Accept-Encoding")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif url.path == "/healthz":
            self.send_json(200, {"ok": True})
        elif url.path == "/api/poll":
            q = parse_qs(url.query)
            s = self.session(q.get("sid", [""])[0])
            if not s:
                return self.send_json(404, {"error": "Your session expired. Reload the page to start again."})
            since = int(q.get("since", ["0"])[0] or 0)
            if q.get("wait", ["0"])[0] == "1":
                s.wait_for_news(since, LONG_POLL_SECONDS)
            with s.lock:
                q_open = dict(s.question) if s.question else None
                body = {"events": s.events[since:], "running": s.running, "tasks": tasks_of(s.state),
                        "question": q_open and {"words": q_open["words"], "seconds_left": max(0, int(q_open["deadline"] - time.time()))}}
            self.send_json(200, body)
        else:
            self.send_json(404, {"error": "not found"})

    def do_POST(self):
        """Start a session, run a goal, answer the gate, or reset the list."""
        path, body = urlparse(self.path).path, self.read_json()
        if path == "/api/session":
            forget_idle_sessions()
            with SESSIONS_LOCK:
                if len(SESSIONS) >= MAX_SESSIONS:
                    return self.send_json(503, {"error": "The demo is full right now. Please try again in a few minutes."})
                s = Session(body.get("seed", "week_start"))
                SESSIONS[s.id] = s
            return self.send_json(200, {"sid": s.id, "model": DEMO_MODEL, "tasks": tasks_of(s.state), "seed": s.seed,
                                        "seeds": {k: v[0] for k, v in SEEDS.items()}, "runs_left": RUNS_PER_VISITOR,
                                        "max_goal": MAX_GOAL_CHARS, "approval_timeout": APPROVAL_TIMEOUT})
        s = self.session(body.get("sid"))
        if not s:
            return self.send_json(404, {"error": "Your session expired. Reload the page to start again."})
        if path == "/api/run":
            goal = str(body.get("goal", "")).strip()
            if not goal or len(goal) > MAX_GOAL_CHARS:
                return self.send_json(400, {"error": f"Type a goal of 1 to {MAX_GOAL_CHARS} characters."})
            with s.lock:
                if s.running:
                    return self.send_json(409, {"error": "A goal is already running. Wait for it to finish."})
                if s.runs >= RUNS_PER_VISITOR:
                    return self.send_json(429, {"error": f"You have used all {RUNS_PER_VISITOR} runs for this visit."})
            if not daily_allows():
                return self.send_json(429, {"error": "The demo has reached today's limit. Please come back tomorrow."})
            with s.lock:
                s.running, s.runs = True, s.runs + 1
            s.push("start", goal=goal, model=DEMO_MODEL)
            threading.Thread(target=run_goal, args=(s, goal), daemon=True).start()
            return self.send_json(200, {"ok": True, "runs_left": RUNS_PER_VISITOR - s.runs})
        if path == "/api/answer":
            with s.lock:
                if not s.question:
                    return self.send_json(409, {"error": "There is no question waiting."})
                s.answer = bool(body.get("allow"))
            s.answered.set()
            return self.send_json(200, {"ok": True})
        if path == "/api/reset":
            with s.lock:
                if s.running:
                    return self.send_json(409, {"error": "Wait for the running goal to finish first."})
            s.reset(body.get("seed", s.seed))
            return self.send_json(200, {"ok": True, "tasks": tasks_of(s.state), "seed": s.seed})
        self.send_json(404, {"error": "not found"})


def main() -> None:
    """Serve the demo until stopped."""
    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST", "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1")
    print(f"Desk Assistant demo on http://{'localhost' if host == '127.0.0.1' else host}:{port}  "
          f"(model: {DEMO_MODEL}, {DAILY_RUN_CAP} runs/day, {RUNS_PER_VISITOR} per visitor)", flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
