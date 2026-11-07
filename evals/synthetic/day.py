"""Synthetic Session 2 cases: a spoken morning, raw day logs, brief days with checkable preferences, and media queries.

Every table is written by hand below, so you can read what is tested. The answers are known by
construction: the conversation says "lakeside lodge", so a correct answer mentions it. The simulated
user of Task 4 has preferences that code can check; its feedback is labeled synthetic, never human.
"""

import json
import tempfile
from pathlib import Path

import numpy as np

from agent import platforms
from agent.senses import audio as audio_sense
from evals.synthetic import audio, screens
from evals.synthetic.text import write_jsonl

# --- Task 3: a morning conversation (voice index, sentence) -------------------------------
CONVERSATION = [
    (0, "Morning. Let's sort out the plan for the team offsite before standup."),
    (1, "Sure. I looked at two venues, the lakeside lodge and the city hotel."),
    (0, "The lakeside lodge is cheaper and has the bigger meeting room, so let's book the lakeside lodge."),
    (1, "Agreed. I'll ask Priya to send the deposit by Wednesday."),
    (0, "For the date, the fourteenth works better than the twenty first."),
    (1, "Fine, the offsite is on the fourteenth then."),
    (2, "Sorry to jump in. The budget review with Anita moved from Tuesday to Thursday at two."),
    (0, "Thanks. Then I'll finish the slides on Wednesday evening."),
    (1, "One more thing. We decided to drop the printed handouts and share a PDF instead."),
    (2, "And the new intern, Marco, starts on Monday. He needs a laptop."),
    (0, "I'll order the laptop today."),
]
# (question, must_mention groups, unanswerable). Asked at 14:00 the same day.
QUESTIONS = [
    ("What did we decide about the offsite venue this morning?", [["lakeside"]], False),
    ("When is the offsite?", [["fourteenth", "14"]], False),
    ("Who is sending the deposit, and by when?", [["priya"], ["wednesday"]], False),
    ("When is the budget review with Anita now?", [["thursday"], ["two", "2"]], False),
    ("What did we decide about the handouts this morning?", [["pdf"]], False),
    ("Who starts on Monday?", [["marco"]], False),
    ("What did I say I would order today?", [["laptop"]], False),
    ("What did we decide about the catering this morning?", [], True),
]

# --- Consolidation: raw logs of one day (HH:MM, kind, text) and queries (text, row numbers) --
LOGS = [
    {
        "id": "workday",
        "rows": [
            ("09:02", "transcript", "Let's book the lakeside lodge for the offsite, it has the bigger room."),
            ("09:03", "transcript", "Priya will send the deposit for the lodge by Wednesday."),
            ("09:05", "transcript", "The offsite is on the fourteenth, not the twenty first."),
            ("09:07", "action_item", "Finish the budget slides (due Wednesday evening)"),
            ("09:10", "transcript", "Standup moves to nine thirty every weekday from next week."),
            ("09:12", "screen_help", "Terminal shows ModuleNotFoundError for requests. Fix: pip install requests."),
            ("09:14", "transcript", "Marco the new intern starts on Monday and needs a laptop."),
            ("10:40", "transcript", "Anita said the budget review is now on Thursday at two."),
            ("10:42", "transcript", "We will share the handouts as a PDF instead of printing them."),
            ("10:45", "voice_note", "Remember to call the caterer about vegetarian options for the offsite."),
            ("10:47", "transcript", "Correction, standup stays at nine, the nine thirty change is cancelled."),
        ],
        "queries": [
            ("which venue did we pick for the offsite?", [0]),
            ("who pays the deposit?", [1]),
            ("what date is the offsite?", [2]),
            ("when is the budget review?", [7]),
            ("what time is standup now?", [4, 10]),
            ("what is happening with the handouts?", [8]),
            ("what does the intern need?", [6]),
            ("what error did I hit in the terminal?", [5]),
        ],
    },
    {
        "id": "homeday",
        "rows": [
            ("08:15", "voice_note", "The plumber comes on Monday morning to fix the kitchen sink."),
            ("08:20", "transcript", "Mum's birthday dinner is on Saturday at the Italian place."),
            ("08:22", "transcript", "I booked a table for six at seven in the evening."),
            ("08:25", "action_item", "Buy a present for mum (due Friday)"),
            ("08:30", "transcript", "The car service moved to next Tuesday at the Bosch garage."),
            ("12:05", "transcript", "Electricity bill is due at the end of the month, about ninety euros."),
            ("12:07", "transcript", "Passport expires in March, so I should renew it in January."),
            ("12:10", "voice_note", "Book club picked the new Murakami novel for this month."),
        ],
        "queries": [
            ("when is the plumber coming?", [0]),
            ("where is mum's birthday dinner?", [1]),
            ("how many people is the table for?", [2]),
            ("when is the car service?", [4]),
            ("how much is the electricity bill?", [5]),
            ("when should I renew my passport?", [6]),
            ("what is the book club reading?", [7]),
        ],
    },
]

# --- Task 4: days for the brief (split dev or test) ----------------------------------------
DAYS = [
    (
        "d1",
        "dev",
        [
            "(09:00, summary) Picked the lakeside lodge for the offsite on the 14th; Priya sends the deposit.",
            "(09:07, action_item) Finish the budget slides (due Wednesday evening)",
            "(10:40, fact) The budget review with Anita is on Thursday at 2pm.",
            "(11:30, screen_help) Fixed a ModuleNotFoundError for requests with pip install.",
            "(15:00, fact) Dentist appointment with Dr. Lee tomorrow at 3pm.",
            "(16:20, action_item) Order a laptop for Marco (due Monday)",
        ],
    ),
    (
        "d2",
        "dev",
        [
            "(08:30, summary) The plumber comes Monday morning for the kitchen sink.",
            "(09:15, action_item) Pay the electricity bill (due Friday)",
            "(12:00, fact) Mum's birthday dinner is Saturday at 7pm, table for six.",
            "(14:10, screen_help) A merge conflict in app.py was resolved.",
            "(17:45, action_item) Buy a present for mum (due tomorrow)",
        ],
    ),
    (
        "d3",
        "dev",
        [
            "(09:05, summary) Standup stays at 9am; the 9:30 change was cancelled.",
            "(10:00, action_item) Send Ravi the demo notes (due tonight)",
            "(11:20, screen_help) Port 8000 was already in use; the server moved to port 8001.",
            "(13:40, fact) The car service is next Tuesday at the Bosch garage.",
            "(16:00, summary) The launch plan covers marketing, hiring and next year's budget.",
            "(17:30, action_item) Review the hiring plan (due tomorrow at 10am)",
        ],
    ),
    (
        "d4",
        "dev",
        [
            "(08:50, fact) Passport expires in March; renew it in January.",
            "(10:30, summary) The book club picked the new Murakami novel.",
            "(12:15, action_item) Book a room for the Thursday meeting (due Wednesday)",
            "(15:30, screen_help) One test failed: test_add expected 5 but got 4.",
            "(18:00, action_item) Call the caterer about vegetarian options (due tomorrow)",
        ],
    ),
    (
        "d5",
        "test",
        [
            "(09:10, summary) The offsite agenda has three workshops and a team dinner.",
            "(10:00, action_item) Send the agenda to the team (due tomorrow at 9am)",
            "(11:45, screen_help) The backup failed with a disk full error; old logs were deleted.",
            "(14:00, fact) Anita approved the budget at the Thursday review.",
            "(16:30, action_item) Update the budget sheet (due Friday)",
        ],
    ),
    (
        "d6",
        "test",
        [
            "(08:40, fact) School pickup moves to 3:30pm on Wednesdays.",
            "(09:30, action_item) Renew the car insurance (due tomorrow)",
            "(12:20, summary) Lunch with Sam was moved to next week.",
            "(15:10, screen_help) A SyntaxError in main.py was fixed by adding a colon.",
            "(17:00, action_item) Pick up the dry cleaning (due Saturday)",
            "(19:00, fact) The gym opens at 6am on weekdays from Monday.",
        ],
    ),
    (
        "d7",
        "test",
        [
            "(09:00, summary) Marco started today and got his laptop.",
            "(10:15, action_item) Write Marco's onboarding plan (due tomorrow)",
            "(13:30, fact) The team photo is on Friday at 11am.",
            "(15:45, screen_help) config.yaml was missing; it was created from the example file.",
            "(17:20, action_item) Reply to the venue about the dinner menu (due Thursday)",
        ],
    ),
    (
        "d8",
        "test",
        [
            "(08:30, summary) The kitchen sink is fixed; the plumber's invoice arrived.",
            "(09:45, action_item) Pay the plumber's invoice (due tomorrow)",
            "(11:00, fact) The book club meets next Tuesday at 7pm.",
            "(14:30, screen_help) All 13 tests passed after the refactor.",
            "(16:40, action_item) Send the refactor notes to Ravi (due Friday)",
        ],
    ),
]
# The simulated user's preferences. `check` names a function in evals/task4/eval.py. `complaint` is its feedback.
PREFERENCES = [
    ("no_greeting", "Do not start the headline with a greeting.", "Skip the greeting, just give me the headline."),
    ("max_bullets", "Use at most 6 bullets in total.", "Too long. Six bullets at most, please."),
    (
        "deadlines_first",
        "Put the items that have a deadline in the first section, called 'Deadlines'.",
        "I want the deadlines first, in a section called Deadlines.",
    ),
    (
        "times_24h",
        "Write times in 24-hour format such as 14:30, never am or pm.",
        "Please write times like 14:30, not 2pm.",
    ),
    (
        "end_tomorrow",
        "End with a section called 'Tomorrow' with what is due tomorrow.",
        "End with a Tomorrow section so I know what is next.",
    ),
    ("short_bullets", "Keep every bullet under 15 words.", "The bullets are too wordy. Under 15 words each."),
]
# A deliberate overfitting example: feedback that is true for one dev day only.
OVERFIT = {
    "day": "d1",
    "feedback": [
        "Start the headline with the word Dentist.",
        "Always mention Dr. Lee in the first bullet.",
    ],
    "leak_words": ["dentist", "dr. lee"],  # not in any test day: a test brief that names them invented it
    "headline_starts": "dentist",  # the one-day rule, checked on its own day
    "first_bullet_has": "dr. lee",
}

# --- Multimodal memory: which picture or clip each query is about ----------------------------
IMAGE_QUERIES = [
    ("the screenshot where the requests package was missing", "module-missing"),
    ("when port 8000 was already taken", "port-in-use"),
    ("the python file with a missing colon", "syntax-error"),
    ("the yaml config file that could not be found", "file-missing"),
    ("the git merge that failed", "merge-conflict"),
    ("the failing unit test for addition", "test-failure"),
    ("a clean git status with nothing to commit", "all-clean"),
    ("the run where every test passed", "tests-pass"),
]
AUDIO_QUERIES = [
    ("who needs the quarterly report?", 1),
    ("when does Ravi's demo begin?", 2),
    ("what groceries do I need?", 3),
    ("when is the sink being repaired?", 4),
    ("which document do I have to renew?", 5),
    ("where is the Thursday meeting?", 6),
    ("which restaurant did she book?", 7),
    ("did the backup work?", 8),
    ("what does the launch plan include?", 9),
    ("how do I get to the library?", 10),
]


def speak_as(text: str, voice: int) -> np.ndarray:
    """Synthesize one sentence with the n-th voice of the OS plug-in, without the silence at both ends.

    Args:
        text (str): What to say.
        voice (int): Voice index (0 = default).

    Returns:
        np.ndarray: 16 kHz mono float32 audio.
    """
    with tempfile.TemporaryDirectory() as d:
        path = platforms.current().synthesize_speech(text, Path(d) / "speech", voice=voice)
        return audio.trim(audio_sense.load_wav(path))


def build_task3(out: Path) -> None:
    """Write the morning conversation and the questions about it.

    Args:
        out (Path): The case folder of the `task3` eval.
    """
    rng = np.random.default_rng(4)
    parts: list[tuple[np.ndarray, bool]] = [(audio.room_tone(1.0, rng), False)]
    for voice, text in CONVERSATION:
        parts += [(speak_as(text, voice), True), (audio.room_tone(1.4, rng), False)]
    audio.write_wav(out / "conversation.wav", audio.compose(parts)[0])
    questions = [{"question": q, "must_mention": m, "unanswerable": u} for q, m, u in QUESTIONS]
    case = {"start": "09:00", "ask_at": "14:00", "questions": questions, "script": [t for _, t in CONVERSATION]}
    (out / "conversation.json").write_text(json.dumps(case, indent=1))


def build_consolidation(out: Path) -> None:
    """Write the raw day logs and the queries about them.

    Args:
        out (Path): The case folder of the `consolidation` eval.
    """
    for log in LOGS:
        rows = [{"at": at, "kind": k, "text": t} for at, k, t in log["rows"]]
        queries = [{"query": q, "relevant": r} for q, r in log["queries"]]
        (out / f"{log['id']}.json").write_text(json.dumps({"rows": rows, "queries": queries}, indent=1))


def build_task4(out: Path) -> None:
    """Write the brief days, the simulated user's preferences and the overfitting example.

    Args:
        out (Path): The case folder of the `task4` eval.
    """
    write_jsonl(out / "days.jsonl", [{"id": i, "split": s, "memories": m} for i, s, m in DAYS])
    write_jsonl(out / "preferences.jsonl", [{"id": i, "text": t, "complaint": c} for i, t, c in PREFERENCES])
    (out / "overfit.json").write_text(json.dumps(OVERFIT, indent=1))


def build_multimodal(out: Path) -> None:
    """Write the screenshots and speech clips to remember, and the queries that must find each one.

    Args:
        out (Path): The case folder of the `multimodal` eval.
    """
    out.mkdir(parents=True, exist_ok=True)
    for i, (name, (lines, _problem, _mention)) in enumerate(screens.SHOTS.items()):
        screens.render(lines, light=i % 3 == 2).save(out / f"{name}.png")
    for i, text in enumerate(audio.SENTENCES, 1):
        audio.write_wav(out / f"clip{i:02d}.wav", audio.speak(text))
    rows = [{"query": q, "item": f"{n}.png", "modality": "image"} for q, n in IMAGE_QUERIES]
    rows += [{"query": q, "item": f"clip{n:02d}.wav", "modality": "audio"} for q, n in AUDIO_QUERIES]
    write_jsonl(out / "queries.jsonl", rows)
