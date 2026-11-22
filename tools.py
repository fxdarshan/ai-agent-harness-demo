"""The to-do app the agent works on: tasks, the list that holds them, and the tools the model may ask for.

The harness only knows that a tool has a name, a risk, an input check and a run function. What a task is, and
what counts as a valid or duplicate one, belongs here, next to the data. Without these rules a model could add
empty tasks, a 5,000-character note, or the same task five times, and say "done" each time. Putting the rules in
the tool means every caller gets them, whatever the model or the prompt says.

tools.py imports Tool from harness.py; harness.py never imports this file, so the loop stays the same for any app.
The model reads each tool's name, params and description from the menu llm.py builds; it never sees the risk.

What to look at
    Task, State       one to-do item, and the whole list plus the next free id (ids are never reused).
    _same()           the yardstick for the duplicate rule: case, outer spaces and repeated inner spaces are ignored.
    add_task          refuses blank or over-long text, and raises "duplicate of task <id>" instead of adding twice,
                      so the model hears the truth and cannot pretend it added something.
    list_tasks        "id: text" per line, each text cut to PREVIEW_CHARS so one huge note cannot hide the other ids.
    delete_task       the one irreversible tool, so the harness asks a human first. _describe_delete() turns the id
                      into the real task text for that question, because nobody can judge "delete task 4" blind.
    ALL_TOOLS         the menu the agent gets.
    make_oldest_verifier()
                      the "delete my N oldest" rule as a check on the real list. It photographs the ids before the
                      run, because after a wrong delete you can no longer tell which tasks were the oldest.
"""
from dataclasses import dataclass, field
from typing import Any

from harness import Tool

MAX_TEXT_CHARS = 200    # the product promise: a task is 1 to 200 characters
PREVIEW_CHARS = 100     # list_tasks shows this much of each task, so every id stays visible


@dataclass
class Task:
    """One item on the to-do list."""
    id: int
    text: str


@dataclass
class State:
    """The list the agent works on. It lives in the harness's hands, not in the model's memory."""
    tasks: list[Task] = field(default_factory=list)
    next_id: int = 1


def new_state() -> State:
    """An empty to-do list."""
    return State(tasks=[], next_id=1)


def _same(text: str) -> str:
    """The duplicate rule's yardstick: lower case, outer spaces trimmed, inner runs of spaces collapsed to one."""
    return " ".join(text.lower().split())


def _check_text(args: dict[str, Any]) -> "str | None":
    """add_task's input check: the text must be a real, non-blank string within the length limit."""
    text = args.get("text")
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT_CHARS:
        return f"text must be a non-empty string of at most {MAX_TEXT_CHARS} characters"
    return None


def _run_add(args: dict[str, Any], state: State) -> str:
    """Add a task unless an equal one exists; the check comes first, so a refusal changes nothing."""
    wanted = _same(args["text"])
    for task in state.tasks:
        if _same(task.text) == wanted:
            raise ValueError(f"duplicate of task {task.id}")
    task = Task(id=state.next_id, text=args["text"].strip())
    state.tasks.append(task)
    state.next_id += 1
    return f"added task {task.id}"


def _run_list(_args: dict[str, Any], state: State) -> str:
    """Every task as 'id: text', so the model can refer to a task by its number later."""
    if not state.tasks:
        return "no tasks"
    return "\n".join(f"{task.id}: {task.text[:PREVIEW_CHARS]}" for task in state.tasks)


def _check_id(args: dict[str, Any]) -> "str | None":
    """delete_task's input check: the id must be a real integer (Python counts True as 1, so booleans are refused)."""
    value = args.get("id")
    if not isinstance(value, int) or isinstance(value, bool):
        return "id must be an integer"
    return None


def _describe_delete(args: dict[str, Any], state: State) -> str:
    """The approval question in plain words, showing the task's real text so the human sees what would be lost."""
    task = next((t for t in state.tasks if t.id == args.get("id")), None)
    if task is None:
        return f"delete task {args.get('id')} (no such task)"
    return f'delete task {task.id}: "{task.text[:60]}"'


def _run_delete(args: dict[str, Any], state: State) -> str:
    """Remove one task by id; a missing id is an error, never a pretend success, and nothing changes before it fails."""
    for index, task in enumerate(state.tasks):
        if task.id == args["id"]:
            del state.tasks[index]
            return f"deleted task {task.id}"
    raise ValueError(f"no task with id {args['id']}")


add_task = Tool(
    name="add_task",
    description="Add a new task to the to-do list.",
    params={"text": f"string, 1 to {MAX_TEXT_CHARS} characters"},
    risk="write",
    validate=_check_text,
    run=_run_add,
)

list_tasks = Tool(
    name="list_tasks",
    description="List all tasks with their numeric ids.",
    params={},
    risk="read",
    validate=lambda _args: None,          # takes no input, so there is nothing to refuse
    run=_run_list,
)

delete_task = Tool(
    name="delete_task",
    description="Permanently delete a task by its numeric id.",
    params={"id": "integer"},
    risk="irreversible",                  # the one tool the gate stops: once gone, a task cannot come back
    validate=_check_id,
    run=_run_delete,
    describe=_describe_delete,
)

ALL_TOOLS: list[Tool] = [add_task, list_tasks, delete_task]


def make_oldest_verifier(state: State, n: int):
    """Snapshot the ids now and return a check that passes only if exactly the n oldest (lowest-id) tasks are gone."""
    starting_ids = sorted(task.id for task in state.tasks)
    oldest = set(starting_ids[:n])
    others = set(starting_ids[n:])

    def verify(current: State) -> "str | None":
        """Compare the real list with the snapshot: a problem sentence, or None when the goal truly holds."""
        now = {task.id for task in current.tasks}
        left_behind = sorted(oldest & now)
        if left_behind:
            return f"task {left_behind[0]} is one of the {n} oldest but still exists"
        wrongly_gone = sorted(others - now)
        if wrongly_gone:
            return f"task {wrongly_gone[0]} was deleted but is not among the {n} oldest"
        return None      # tasks added during the run are fine: the goal said nothing about them

    return verify
