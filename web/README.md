# Desk Assistant Live: the harness as a web page

A small web app that lets anyone try the to-do agent in a browser. A visitor types a goal, watches each step
arrive, and answers the approval question with a real **Approve / Deny** button. Every goal runs through the same
`harness.run` as `agent.py`: the same tools, the same gate, the step cap, the repeat check and the denial cap.

It uses only the Python standard library (no Flask, no Gradio), so it runs anywhere Python 3.11+ runs.

This folder is an add-on on the `web-demo` branch. It is not part of the course phases, and it changes no course file.

## Run it on your computer

```
python web/app.py
```
Then open http://localhost:8000. With `GEMINI_API_KEY` in your `.env` it uses Gemini; without a key it uses a
scripted stand-in model (no AI, no quota) so you can still click through the gate.

## Put it on the web (Render, free tier)

1. Push the `web-demo` branch to your GitHub repo.
2. On https://render.com, sign in with GitHub and create a **Web Service** from the repo.
3. Settings:
   - Branch: `web-demo`
   - Runtime: Python 3
   - Build command: leave empty, or `pip install -r requirements.txt` if you also want the LangGraph engine
   - Start command: `python web/app.py`
4. Under **Environment**, add `GEMINI_API_KEY` as a **secret** (paste it there, never in the code or a chat). Add
   `GEMINI_MODEL=gemini-flash-lite-latest` if you want to pin the model.
5. Deploy. Render sets `PORT` itself; the app listens on it.

The free tier sleeps when idle, so the first visit after a pause takes about 30 seconds to wake up.

## Settings (environment variables)

| Variable | Default | What it does |
|---|---|---|
| `DEMO_MODEL` | `gemini` if a key is set, else `stand-in` | `stand-in` uses the scripted fooled model: no key, no quota |
| `DAILY_RUN_CAP` | `150` | runs for all visitors together per UTC day; protects your free quota |
| `RUNS_PER_VISITOR` | `12` | runs per browser visit |
| `APPROVAL_TIMEOUT` | `120` | seconds before an unanswered question counts as **no** |
| `PORT` | `8000` | set by most hosts automatically |

## What keeps a public demo safe

- **Your key** stays on the server, in an environment variable. No page or API response contains it.
- **Your quota**: the daily cap, the per-visitor cap, a 200-character goal limit and an 8-step cap per run.
- **Other visitors**: each visit gets its own in-memory copy of a starting list. Nothing is written to disk, and
  nobody can see or change anyone else's list. A restart resets everything.
- **No answer means no**: an approval question that times out is a refusal, exactly like `--approve` with no human.
- **Privacy**: goals go to the model provider, so the page asks visitors to type toy text only. Langfuse export is
  off here.

## Files

- `app.py`: the server, the per-visitor sessions, the web approver and the JSON endpoints.
- `index.html`: the page.
