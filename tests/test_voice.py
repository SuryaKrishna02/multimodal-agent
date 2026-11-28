"""Tests for the Session 3 voice parts that need no model and no speakers.

How a reply is cut into spoken pieces, the IPA override for words that espeak says wrong, the route (dashboard tab
or this machine), stopping, and the voice endpoints of the dashboard. Kokoro and the OS playback are mocked. The
module is skipped when the third-party packages of the agent are not installed.
"""

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

try:
    import numpy as np
    from fastapi import HTTPException

    from agent import config, events, serving
    from agent.platforms.base import Unsupported
    from agent.voice import speaker, tts

    with (
        mock.patch.object(serving.accel, "llama_server", return_value=Path("llama-server")),
        mock.patch.object(serving.accel, "current", return_value=mock.Mock(server_args=list)),
    ):
        from dashboard import voice
except ImportError as e:  # bare interpreter without requirements/base.txt
    raise unittest.SkipTest(f"agent dependencies are not installed: {e}") from e


class Sentences(unittest.TestCase):
    """A reply is spoken in pieces, and the first piece is short, so the first sound comes early."""

    def test_splits_at_sentence_ends(self):
        """Each sentence is one piece, and extra spaces are dropped."""
        self.assertEqual(tts.sentences("Hello there.  How are you?\nFine!"), ["Hello there.", "How are you?", "Fine!"])

    def test_first_piece_is_cut_at_a_comma(self):
        """A long first sentence is cut at a comma; later sentences may be longer."""
        text = "The test parse dates is failing, with a key error on line forty two. " + "Then " * 20 + "done."
        pieces = tts.sentences(text, max_chars=160, first_chars=40)
        self.assertEqual(pieces[:2], ["The test parse dates is failing,", "with a key error on line forty two."])
        self.assertTrue(pieces[2].startswith("Then"))

    def test_clause_longer_than_the_limit_stays_whole(self):
        """Without a comma there is nowhere to cut, so the piece stays whole."""
        long = "word " * 30
        self.assertEqual(tts.sentences(long, max_chars=20, first_chars=20), [long.strip()])

    def test_empty_text_has_no_pieces(self):
        """Blank text gives nothing to say."""
        self.assertEqual(tts.sentences("   "), [])


class Speakable(unittest.TestCase):
    """Markdown, symbols and emoji in a reply are not read out (espeak says "asterisk", "hash"...)."""

    def test_markdown_becomes_plain_sentences(self):
        """Emphasis, headings, bullets, numbered items, links and emoji go; each list item becomes a sentence."""
        reply = (
            "Try **North Park** \U0001f60a\n\n## Top picks\n- **Museums:** Balboa Park\n"
            "2. Gaslamp -> nightlife & bars\n- See [the zoo](https://zoo.org)"
        )
        self.assertEqual(
            tts.speakable(reply),
            "Try North Park. Top picks. Museums: Balboa Park. 2: Gaslamp to nightlife and bars. See the zoo.",
        )

    def test_tables_code_and_symbols(self):
        """A table is read cell by cell, a code block is skipped, and a bare link or symbol is said in words."""
        reply = "| Day | Plan |\n|---|---|\n| 1 | Museums |\n```py\nprint(1)\n```\nIt is 70\u00b0F, see https://x.com."
        self.assertEqual(tts.speakable(reply), "Day, Plan. 1, Museums. It is 70 degrees F, see the link.")

    def test_plain_text_is_unchanged(self):
        """A plain reply passes through as it is, and snake_case is said as two words."""
        self.assertEqual(tts.speakable("It is 6:00 PM, Wednesday."), "It is 6:00 PM, Wednesday.")
        self.assertEqual(tts.speakable("Open file_name.py #3"), "Open file name.py number 3.")


class Pronunciation(unittest.TestCase):
    """A word in `config.PRONUNCIATIONS` goes to Kokoro as IPA; the rest goes through espeak."""

    def test_name_is_replaced_by_its_ipa(self):
        """The name never reaches espeak, in any letter case."""
        tokenizer = mock.Mock(phonemize=lambda text, lang: f"<{text.strip()}>")
        with (
            mock.patch.object(tts, "engine", return_value=mock.Mock(tokenizer=tokenizer)),
            mock.patch.dict(config.PRONUNCIATIONS, {"jarvis": "IPA"}, clear=True),
        ):
            self.assertEqual(tts.phonemes("Hey JARVIS, remind me."), "<Hey> IPA <, remind me.>")


class FakePlayback:
    """Stands in for the OS player: it plays for `seconds`, or until stopped."""

    def __init__(self, seconds: float):
        """Start playing."""
        self.until, self.stopped = time.monotonic() + seconds, False

    def stop(self) -> None:
        """Cut the sound off."""
        self.stopped = True

    def done(self) -> bool:
        """Tell whether the sound has ended."""
        return self.stopped or time.monotonic() >= self.until


class VoiceRuntime(unittest.TestCase):
    """Keep the event log, the chunks and the tab heartbeat in a temporary folder, with Kokoro mocked."""

    def setUp(self) -> None:
        """Patch the paths and replace speech synthesis with 0.1 s of silence."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        for patcher in (
            mock.patch.object(config, "RUNTIME", root),
            mock.patch.object(config, "VOICE_DIR", root / "voice"),
            mock.patch.object(config, "VOICE_TAB_FILE", root / "voice_tab.json"),
            mock.patch.object(config, "VOICE_POLL_S", 0.005),
            mock.patch.object(tts, "synthesize", return_value=np.zeros(2400, dtype=np.float32)),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def kinds(self) -> list[str]:
        """Return the kinds of the voice events so far."""
        return [e["kind"] for e in events.since(0, "voice.")]


class Route(VoiceRuntime):
    """Speech goes to the dashboard tab while one listens, else to this machine."""

    def test_tab_heartbeat(self):
        """A fresh heartbeat means a tab listens; an old one does not."""
        self.assertFalse(speaker.tab_listening())
        speaker.mark_tab()
        self.assertTrue(speaker.tab_listening())
        config.VOICE_TAB_FILE.write_text(json.dumps({"ts": time.time() - config.VOICE_TAB_ALIVE_S - 1}))
        self.assertFalse(speaker.tab_listening())

    def test_two_tabs_write_the_heartbeat_at_once(self):
        """Two event streams that write the heartbeat at the same time never fail, and leave no temporary file."""
        errors: list[Exception] = []

        def beat() -> None:
            for _ in range(200):
                try:
                    speaker.mark_tab()
                except OSError as e:
                    errors.append(e)

        writers = [threading.Thread(target=beat) for _ in range(4)]
        for w in writers:
            w.start()
        for w in writers:
            w.join()
        self.assertEqual(errors, [])
        self.assertTrue(speaker.tab_listening())
        self.assertEqual(list(config.VOICE_TAB_FILE.parent.glob("*.part")), [])

    def test_tab_route_writes_chunks_and_plays_nothing_here(self):
        """With a tab listening, each piece becomes a WAV chunk and the OS player is not used."""
        speaker.mark_tab()
        plat = mock.Mock()
        with mock.patch.object(speaker.platforms, "current", return_value=plat):
            out = speaker.speak("One. Two.")
        self.assertEqual((out["route"], out["chunks"], out["stopped"]), ("tab", 2, False))
        self.assertTrue((config.VOICE_DIR / out["utt"] / "1.wav").is_file())
        self.assertEqual(self.kinds(), ["voice.say", "voice.chunk", "voice.chunk"])
        plat.play_audio.assert_not_called()

    def test_local_route_plays_in_order(self):
        """Without a tab, the OS player plays each chunk, and the first sound is recorded."""
        plat = mock.Mock(play_audio=mock.Mock(side_effect=lambda path: FakePlayback(0.01)))
        with mock.patch.object(speaker.platforms, "current", return_value=plat):
            out = speaker.speak("One. Two.")
        self.assertEqual(out["route"], "local")
        self.assertEqual([c.args[0].name for c in plat.play_audio.call_args_list], ["0.wav", "1.wav"])
        self.assertEqual(self.kinds().count("voice.played"), 1)

    def test_os_without_playback_reports_it(self):
        """An OS plug-in without play_audio gives a voice.error event that says to open the dashboard, no crash."""
        plat = mock.Mock(play_audio=mock.Mock(side_effect=Unsupported("no play_audio here")))
        with mock.patch.object(speaker.platforms, "current", return_value=plat):
            speaker.speak("One.")
        error = events.last("voice.error")
        self.assertIsNotNone(error)
        self.assertIn("No dashboard tab listens", error["error"] if error else "")


class Stop(VoiceRuntime):
    """A stop request ends speech that plays here, and stops the writer before the next piece."""

    def test_stop_cuts_local_playback(self):
        """The player is stopped, the rest is not written, and the stop latency is recorded."""
        players = []

        def play(path):
            players.append(FakePlayback(10.0))
            speaker.stop("test")  # someone presses Stop while the first chunk plays
            return players[-1]

        with mock.patch.object(speaker.platforms, "current", return_value=mock.Mock(play_audio=play)):
            out = speaker.speak("One. Two. Three.")
        self.assertTrue(out["stopped"])
        self.assertTrue(players[0].stopped)
        self.assertEqual(len(players), 1)
        stopped = events.last("voice.stopped")
        self.assertIsNotNone(stopped)
        self.assertLess(stopped["after_stop_s"] if stopped else 99, 1.0)

    def test_cancelled_turn_writes_no_chunk(self):
        """Speech of a turn that a newer turn cancelled stops before its first chunk."""
        cancel = threading.Event()
        cancel.set()
        self.assertEqual(speaker.speak("One. Two.", route="tab", cancel=cancel)["chunks"], 0)

    def test_old_stop_does_not_silence_new_speech(self):
        """Only a stop request after the start of an utterance stops it."""
        speaker.stop("earlier")
        speaker.mark_tab()
        self.assertFalse(speaker.speak("One. Two.")["stopped"])


class Endpoints(VoiceRuntime):
    """The dashboard serves only chunks that speak() made, and records the tab's timings."""

    def test_audio_rejects_other_paths(self):
        """An id in any other format, or a missing chunk, is a 404, so no other file can be read."""
        for utt, n in (("../../etc", 0), ("20261124-204023-9e367f", 0), ("20261124-204023-9e367f", -1)):
            with self.subTest(utt=utt, n=n), self.assertRaises(HTTPException):
                voice.audio(utt, n)

    def test_audio_serves_a_chunk(self):
        """A chunk that speak() wrote is served."""
        speaker.mark_tab()
        utt = speaker.speak("One.")["utt"]
        self.assertTrue(str(voice.audio(utt, 0).path).endswith("0.wav"))

    def test_played_and_stopped_record_latency(self):
        """The tab's reports become events with the time since the say or the stop event."""
        played = voice.played({"utt": "u", "say_ts": time.time() - 0.5})
        stopped = voice.stopped({"utt": "u", "stop_ts": time.time() - 0.05})
        self.assertAlmostEqual(played["since_say_s"], 0.5, delta=0.2)
        self.assertAlmostEqual(stopped["after_stop_s"], 0.05, delta=0.2)

    def test_say_needs_text(self):
        """An empty text is refused."""
        with self.assertRaises(HTTPException):
            voice.say({"text": "  "})


if __name__ == "__main__":
    unittest.main()


class Turn(VoiceRuntime):
    """A spoken request becomes a reply that is spoken, with the time of each stage recorded."""

    def test_answer_speaks_the_reply(self):
        """The transcript goes to `ask`, the reply is spoken, and voice.turn holds both."""
        utterance = {"samples": np.zeros(16000, np.float32), "start_ts": 1.0, "end_ts": 2.0}
        with (
            mock.patch("agent.voice.turn.stt.transcribe", return_value="remind me to call Mike"),
            mock.patch.object(speaker, "speak") as speak,
        ):
            from agent.voice import turn

            event = turn.answer(utterance, lambda text, details: f"OK: {text}", "talk")
        self.assertEqual((event["text"], event["reply"]), ("remind me to call Mike", "OK: remind me to call Mike"))
        speak.assert_called_once()
        self.assertEqual(speak.call_args.args[0], "OK: remind me to call Mike")

    def test_no_words_says_so(self):
        """An empty transcript is not sent to the model."""
        ask = mock.Mock()
        with (
            mock.patch("agent.voice.turn.stt.transcribe", return_value="  "),
            mock.patch.object(speaker, "speak") as speak,
        ):
            from agent.voice import turn

            turn.answer({"samples": np.zeros(8000, np.float32), "start_ts": 0, "end_ts": 0.5}, ask, "wake word")
        ask.assert_not_called()
        self.assertIn("did not catch", speak.call_args.args[0])

    def test_barge_in_drops_the_reply_being_written(self):
        """A second turn while Gemma still writes the first reply: only the second reply is spoken."""
        from agent.voice import turn

        release = threading.Event()

        def ask(text: str, details: dict) -> str:
            if text == "first":
                release.wait(5)  # Gemma still writes the first reply
            return f"reply to {text}"

        def utterance(seconds: float) -> dict:
            return {"samples": np.zeros(int(16000 * seconds), np.float32), "start_ts": 0.0, "end_ts": seconds}

        loop = turn.WakeLoop(ask, use_wake=False)
        with (
            mock.patch.object(turn, "request", side_effect=[utterance(1), utterance(2)]),
            mock.patch.object(turn.stt, "transcribe", side_effect=lambda s: "first" if len(s) == 16000 else "second"),
            mock.patch.object(speaker, "speak") as speak,
        ):
            loop._turn("talk")  # returns once the request is recorded: the detector listens again
            first = loop.worker
            assert first is not None
            self.assertTrue(first.is_alive())  # the first reply is still being written
            loop._turn("wake word")  # the user barges in
            assert loop.worker is not None
            loop.worker.join(5)
            release.set()
            first.join(5)
        self.assertEqual([c.args[0] for c in speak.call_args_list], ["reply to second"])
        self.assertEqual(sorted(e["interrupted"] for e in events.since(0, "voice.turn")), [False, True])
        self.assertIn("voice.stop", self.kinds())

    def test_request_gives_up_on_silence(self):
        """Nobody speaks after the wake: no request, after the wait."""
        from agent.voice import turn

        with mock.patch.object(config, "VOICE_REQUEST_WAIT_S", 0.05), mock.patch("agent.voice.turn.audio.VAD") as vad:
            vad.return_value.segments.return_value = []
            self.assertIsNone(turn.request(lambda: np.zeros(1280, np.float32), time.time()))


class WakeGate(unittest.TestCase):
    """The two stages of the wake word: a sure score wakes, a "maybe" is checked once (no model needed)."""

    def setUp(self) -> None:
        """Import the wake module (its models load only in Detector)."""
        from agent.voice import wake

        self.wake = wake
        self.audio = lambda: np.zeros(16000, np.float32)

    def run_scores(self, gate, scores: list[float]) -> list[int]:
        """Feed one score per frame and return the frames that woke."""
        return [n for n, s in enumerate(scores, start=1) if gate.judge(n, s, self.audio)]

    def test_sure_score_wakes_once(self):
        """A score over the threshold wakes; the frames just after it do not wake again."""
        gate = self.wake.Gate(0.5, 0.1, confirm=mock.Mock(return_value=False))
        self.assertEqual(self.run_scores(gate, [0.0, 0.9, 0.95, 0.9, 0.0]), [2])

    def test_maybe_is_checked_once_and_can_wake(self):
        """A score under the threshold asks the second stage once, after the delay, and a yes is a wake."""
        confirm = mock.Mock(return_value=True)
        gate = self.wake.Gate(0.5, 0.1, confirm)
        delay = round(config.WAKE_CHECK_DELAY_S * config.SAMPLE_RATE / self.wake.FRAME)
        woke = self.run_scores(gate, [0.0, 0.3, 0.4, 0.2] + [0.0] * 10)
        self.assertEqual(woke, [2 + delay])
        confirm.assert_called_once()

    def test_maybe_that_fails_the_check_is_not_checked_again(self):
        """A "no" from the second stage: the same phrase does not ask Whisper over and over."""
        confirm = mock.Mock(return_value=False)
        gate = self.wake.Gate(0.5, 0.1, confirm)
        self.assertEqual(self.run_scores(gate, [0.3] * 15), [])
        confirm.assert_called_once()

    def test_without_second_stage_a_maybe_is_ignored(self):
        """check_threshold None: only the model decides, as before."""
        gate = self.wake.Gate(0.5, None, confirm=mock.Mock(return_value=True))
        self.assertEqual(self.run_scores(gate, [0.4] * 10), [])

    def test_name_check_accepts_near_spellings_only(self):
        """Near spellings of the name count; "Travis" (a near miss in the wake eval) and ordinary words do not."""
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(config, "RUNTIME", Path(tmp)):
            for text, expected in (
                ("Hey Javis.", True),
                ("Hi Jarvis", True),
                ("Hey Travis, the report", False),
                ("I need the service.", False),
            ):
                with mock.patch.object(self.wake.stt, "transcribe", return_value=text):
                    self.assertIs(self.wake.said_name(self.audio()), expected, text)


class Conversation(VoiceRuntime):
    """After a reply has been heard, the next request needs no wake word."""

    def test_tab_reports_the_end(self):
        """wait_heard returns once the tab posts /ended, and False after a stop."""
        speaker.mark_tab()
        out = speaker.speak("One.")
        voice.ended({"utt": out["utt"]})
        self.assertTrue(speaker.wait_heard(out))
        again = speaker.speak("Two.")
        speaker.stop("button")
        self.assertFalse(speaker.wait_heard(again))

    def test_local_speech_reports_the_end(self):
        """On this machine, speak() returns after the last chunk and records voice.ended."""
        plat = mock.Mock(play_audio=mock.Mock(side_effect=lambda path: FakePlayback(0.01)))
        with mock.patch.object(speaker.platforms, "current", return_value=plat):
            out = speaker.speak("One. Two.")
        self.assertIn("voice.ended", self.kinds())
        self.assertTrue(speaker.wait_heard(out))

    def test_follow_up_opens_only_in_a_conversation(self):
        """With the switch on, a heard reply sets follow_up, and the loop's next trigger is "follow-up"."""
        from agent.voice import turn

        spoken = {"utt": "x", "say_id": 0, "route": "tab", "chunks": 1, "audio_s": 0.1, "stopped": False}
        with (
            mock.patch.object(config, "VOICE_ECHO_TAIL_S", 0),
            mock.patch.object(speaker, "wait_heard", return_value=True),
        ):
            off = turn.WakeLoop(mock.Mock(), use_wake=False, conversation=False)
            off._open_follow_up(spoken, threading.Event())
            on = turn.WakeLoop(mock.Mock(), use_wake=False, conversation=True)
            on._open_follow_up(spoken, threading.Event())
        self.assertFalse(off.follow_up.is_set())
        self.assertEqual(on._woke(None), "follow-up")

    def test_silent_follow_up_ends_quietly(self):
        """No words after the reply: the agent does not say "Sorry", and opens no new follow-up."""
        from agent.voice import turn

        with mock.patch.object(turn.stt, "transcribe", return_value=""), mock.patch.object(speaker, "speak") as speak:
            out = turn.answer(
                {"samples": np.zeros(8000, np.float32), "start_ts": 0, "end_ts": 0.5}, mock.Mock(), "follow-up"
            )
        speak.assert_not_called()
        self.assertIsNone(out["spoken"])
