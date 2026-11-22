"""The saved file: how the to-do list survives from one run of the agent to the next.

The model remembers nothing between calls, and the harness keeps the list only in memory, so without this file
every new run would start from an empty list and "what is on my list?" would have no answer. Saving is also
where work gets lost quietly: a crash halfway through writing leaves a broken file, and a loader that shrugs
at a broken file and starts empty throws your whole list away without telling you.

The harness never imports this file. A caller (try_it.py now, agent.py in Phase 6) passes the harness a checkpoint
callback that calls save_state() after every change, and calls load_state() once before a run starts.

What to look at
    save_state()   writes to "<path>.tmp" first and then swaps it in with os.replace, so the real file is always
                   either the old list or the new one, never half of each.
    load_state()   a missing file is a fresh start; anything damaged or inconsistent is a ValueError with the
                   reason, so a person decides what to do instead of the program silently resetting the list.
    _whole_number(), _text()
                   strict type checks: a text id, a decimal id or a boolean means the file was edited or broken,
                   and guessing what it meant could delete or duplicate the wrong task.
"""
import json
import os
from typing import Any

from tools import State, Task, new_state


def save_state(state: State, path: str) -> None:
    """Write the list to `path` atomically, so a crash mid-save never leaves half a file."""
    data = {"next_id": state.next_id, "tasks": [{"id": task.id, "text": task.text} for task in state.tasks]}
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    os.replace(temporary, path)        # one step on disk: the old file or the new one, nothing in between


def _whole_number(value: Any, what: str) -> int:
    """Return `value` only if it is a real integer; True, 4.0 or "4" mean a damaged file, so never guess."""
    if type(value) is not int:
        raise ValueError(f"{what} must be a whole number, got {value!r}")
    return value


def _text(value: Any, what: str) -> str:
    """Return `value` only if it is a string."""
    if not isinstance(value, str):
        raise ValueError(f"{what} must be text, got {value!r}")
    return value


def load_state(path: str) -> State:
    """Read the list from `path`: a missing file is a fresh start, a damaged one is an error, never a quiet reset."""
    if not os.path.exists(path):
        return new_state()
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict) or not isinstance(data.get("tasks"), list) or "next_id" not in data:
            raise ValueError('expected {"next_id": <number>, "tasks": [...]}')
        tasks = []
        for item in data["tasks"]:
            if not isinstance(item, dict) or "id" not in item or "text" not in item:
                raise ValueError(f'every task must look like {{"id": 1, "text": "..."}}, got {item!r}')
            tasks.append(Task(id=_whole_number(item["id"], "a task id"), text=_text(item["text"], "a task text")))
        next_id = _whole_number(data["next_id"], "next_id")
    except (OSError, ValueError) as err:          # json.JSONDecodeError is a ValueError too
        raise ValueError(f"cannot read the saved tasks in {path}: {err}") from err
    ids = [task.id for task in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError(f"the saved tasks in {path} repeat an id, so a delete could hit the wrong task")
    if ids and next_id <= max(ids):
        raise ValueError(f"the saved tasks in {path} have next_id {next_id}, not above every id, so an id would be reused")
    return State(tasks=tasks, next_id=next_id)
