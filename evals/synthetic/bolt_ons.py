"""Synthetic cases of the Session 3 bolt-on evals: skills and mcp_select.

Labeled by construction. The skill prompts are new requests, not the `examples` of the manifests (those are
embedded for tool selection, so testing on them would test on training data). The MCP catalog is made up: 40 tools
in the style of common MCP servers (files, git, calendar, mail, chat, search...), with names `<server>__<tool>` as
our client registers them. No server process runs for this eval: it measures how the model chooses among many
tools, which is the same whether the tools came over MCP or not.
"""

from pathlib import Path

from evals.synthetic.text import write_jsonl

# (prompt, the skill that must fire, or None when no skill should)
SKILL_PROMPTS = [
    ("How many kilometers is a 10 mile run?", "unit_convert"),
    ("Convert 180 pounds to kilograms.", "unit_convert"),
    ("It is 30 degrees Celsius outside, what is that in Fahrenheit?", "unit_convert"),
    ("How many liters are in 3 gallons?", "unit_convert"),
    ("What is 65 miles per hour in km/h?", "unit_convert"),
    ("Give me my update for the daily stand-up.", "standup"),
    ("What did I get done since yesterday? I need it for standup.", "standup"),
    ("Draft what I should say at this morning's stand-up meeting.", "standup"),
    ("Summarize my last day of work as a stand-up update.", "standup"),
    ("What is 15 percent of 80?", None),
    ("Convert this document to PDF.", None),
    ("When is the stand-up meeting tomorrow?", None),
    ("Remind me to join the standup at 10.", None),
    ("Tell me a joke about kilometers.", None),
    ("Translate good morning into Spanish.", None),
    ("What time is it?", None),
]


def _p(**props: str) -> dict:
    """Build a JSON schema with string properties, all required."""
    return {"type": "object", "properties": {k: {"type": "string", "description": v} for k, v in props.items()},
            "required": list(props)}  # fmt: skip


# 40 tools: (name, description, input schema)
CATALOG = [
    ("files__read_file", "Read the contents of a file.", _p(path="file path")),
    ("files__write_file", "Write text to a file, replacing it.", _p(path="file path", content="text")),
    ("files__list_directory", "List the files in a folder.", _p(path="folder path")),
    ("files__search_files", "Find files whose name matches a pattern.", _p(pattern="glob pattern")),
    ("files__move_file", "Move or rename a file.", _p(source="from", destination="to")),
    ("git__status", "Show the working tree status of a git repository.", _p(repo="repository path")),
    ("git__diff", "Show the unstaged changes in a git repository.", _p(repo="repository path")),
    ("git__commit", "Record staged changes with a commit message.", _p(repo="repository path", message="message")),
    ("git__log", "Show the recent commits of a repository.", _p(repo="repository path")),
    ("github__create_issue", "Open a new issue in a GitHub repository.", _p(repo="owner/name", title="issue title")),
    ("github__list_pull_requests", "List the open pull requests of a GitHub repository.", _p(repo="owner/name")),
    ("github__search_code", "Search code across GitHub.", _p(query="search text")),
    ("calendar__list_events", "List the calendar events of a day.", _p(date="YYYY-MM-DD")),
    ("calendar__create_event", "Add an event to the calendar.", _p(title="event title", start="start time")),
    ("calendar__find_free_time", "Find a free slot in the calendar.", _p(duration="minutes")),
    ("mail__search", "Search the email inbox.", _p(query="search text")),
    ("mail__send", "Send an email.", _p(to="address", subject="subject", body="text")),
    ("mail__draft", "Save an email draft without sending it.", _p(to="address", body="text")),
    ("chat__send_message", "Post a message to a team chat channel.", _p(channel="channel name", text="message")),
    ("chat__read_channel", "Read the latest messages of a chat channel.", _p(channel="channel name")),
    ("search__web", "Search the web.", _p(query="search text")),
    ("search__news", "Search recent news articles.", _p(query="search text")),
    ("weather__forecast", "Get the weather forecast for a city.", _p(city="city name")),
    ("weather__current", "Get the current weather in a city.", _p(city="city name")),
    ("maps__directions", "Get directions between two places.", _p(origin="from", destination="to")),
    ("maps__find_place", "Find a place, such as a restaurant, near a location.", _p(query="what", near="where")),
    ("notes__create", "Create a note in the notes app.", _p(title="title", text="note text")),
    ("notes__search", "Search the notes app.", _p(query="search text")),
    ("todo__add_task", "Add a task to the to-do list.", _p(task="task text")),
    ("todo__list_tasks", "List the open tasks of the to-do list.", _p(project="project name")),
    ("todo__complete_task", "Mark a to-do task as done.", _p(task="task text")),
    ("music__play", "Play a song or playlist.", _p(query="song or playlist")),
    ("music__pause", "Pause the music.", {"type": "object", "properties": {}}),
    ("db__query", "Run a read-only SQL query on the project database.", _p(sql="SQL")),
    ("db__list_tables", "List the tables of the project database.", {"type": "object", "properties": {}}),
    ("browser__open_url", "Open a web page in the browser.", _p(url="URL")),
    ("browser__screenshot", "Take a screenshot of a web page.", _p(url="URL")),
    ("translate__text", "Translate text into another language.", _p(text="text", language="target language")),
    ("finance__stock_price", "Get the latest price of a stock.", _p(symbol="ticker")),
    ("reminders__add_reminder", "Set a reminder at a time.", _p(text="what", at="HH:MM")),
]

# (prompt, the tool it needs)
MCP_PROMPTS = [
    ("What is in the file notes/todo.md?", "files__read_file"),
    ("Show me what changed in my repo since the last commit.", "git__diff"),
    ("Open an issue in acme/web titled 'Login button is broken'.", "github__create_issue"),
    ("What meetings do I have on 2026-11-28?", "calendar__list_events"),
    ("Find a free half hour in my calendar.", "calendar__find_free_time"),
    ("Look for the email from Emily about the deposit.", "mail__search"),
    ("Tell the #launch channel that the build is green.", "chat__send_message"),
    ("Will it rain in Chicago tomorrow?", "weather__forecast"),
    ("How do I get from the office to the airport?", "maps__directions"),
    ("Add 'renew passport' to my to-do list.", "todo__add_task"),
    ("Play some lo-fi music.", "music__play"),
    ("How many tables does the project database have?", "db__list_tables"),
    ("Translate 'thank you very much' into Telugu.", "translate__text"),
    ("What is Apple's stock trading at?", "finance__stock_price"),
    ("Write a note titled 'Ideas' with the text 'voice demo first'.", "notes__create"),
    ("Search the news for the latest llama.cpp release.", "search__news"),
    ("Which pull requests are open in acme/web?", "github__list_pull_requests"),
    ("Rename report.txt to report-final.txt.", "files__move_file"),
    ("Remind me at 16:30 to send the slides.", "reminders__add_reminder"),
    ("Draft an email to sam@example.com saying I will be late.", "mail__draft"),
]


def build_skills(out: Path) -> None:
    """Write the skill prompts.

    Args:
        out (Path): The case folder of the `skills` eval.
    """
    rows = [{"id": f"k{i:02d}", "prompt": p, "skill": s} for i, (p, s) in enumerate(SKILL_PROMPTS, 1)]
    write_jsonl(out / "prompts.jsonl", rows)


def build_mcp_select(out: Path) -> None:
    """Write the 40-tool catalog and the prompts.

    Args:
        out (Path): The case folder of the `mcp_select` eval.
    """
    write_jsonl(out / "catalog.jsonl", [{"name": n, "description": d, "input_schema": s} for n, d, s in CATALOG])
    write_jsonl(
        out / "prompts.jsonl", [{"id": f"m{i:02d}", "prompt": p, "tool": t} for i, (p, t) in enumerate(MCP_PROMPTS, 1)]
    )
