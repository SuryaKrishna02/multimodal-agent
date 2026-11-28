"""A spoken turn: wake word (or the Talk button), the request, speech to text, the reply, speech.

    "Hey Jarvis" (wake.py) or Talk          any speech that plays stops: the user barges in
      -> record until a pause                 the Segmenter of the ambient listener (Silero VAD)
      -> speech to text                        agent/senses/stt.py (Whisper)
      -> reply                                 `ask(text)`: the dashboard chat (memories, tools, approvals)
      -> speak                                 agent/voice/speaker.py (the dashboard tab, or this machine)
      -> follow-up                             conversation on: once the reply has played, the next request needs
                                               no wake word for `config.VOICE_FOLLOW_UP_S`

`WakeLoop` owns one microphone stream in 80 ms blocks. It closes the stream while the pause switch
(`audio.paused()`) is on. Each turn emits `voice.turn` with the time of each stage, and the Voice tab compares them
with `config.LATENCY_BUDGET`. The dashboard runs the loop, because the dashboard holds the tool approvals and the
tab that Google Meet shares. macOS lets the daemon's ambient listener read the same microphone at the same time.

Barge-in: the loop records the request itself, then answers it in a worker thread and goes back to the wake detector
at once. So "Hey Jarvis" (or Talk) works while the agent still thinks or speaks, on the tab and on this machine's
speakers. A new turn stops the speech (`voice.stop`) and cancels the turn before it: a reply that is still being
written is dropped, not spoken over the new request. Only the wake word interrupts, not plain speech: without echo
cancellation the microphone also hears the agent (docs/ROADMAP.md).

Conversation: the worker waits until the reply has been heard (`speaker.wait_heard`: the tab reports `voice.ended`),
then `config.VOICE_ECHO_TAIL_S` more, so the microphone does not take the agent's own voice for the user's. Then
it sets `follow_up`, and the loop records the next request at once, with trigger "follow-up". Silence for
`config.VOICE_FOLLOW_UP_S` ends the conversation: the loop waits for the wake word again.
"""

import queue
import threading
import time
from collections.abc import Callable

import numpy as np

from agent import config, events
from agent.senses import ambient, audio, stt
from agent.voice import speaker, wake

BLOCK = wake.FRAME  # 80 ms: the wake detector's frame


def request(blocks: Callable[[], np.ndarray | None], start_ts: float, wait_s: float | None = None) -> dict | None:
    """Record one spoken request: from now until the speaker pauses.

    Args:
        blocks (Callable[[], np.ndarray | None]): Returns the next block of 16 kHz audio, or None if none came.
        start_ts (float): Wall-clock time of the first block.
        wait_s (float | None, optional): Give up if no speech starts within this time. Defaults to
            `config.VOICE_REQUEST_WAIT_S`.

    Returns:
        dict | None: The utterance (`samples`, `start_ts`, `end_ts`), or None when nobody spoke in time.
    """
    wait_s = config.VOICE_REQUEST_WAIT_S if wait_s is None else wait_s
    seg = ambient.Segmenter(audio.VAD(), start_ts)
    heard, began = 0.0, time.monotonic()
    while time.monotonic() - began < config.VOICE_REQUEST_MAX_S:
        block = blocks()
        if block is None:
            continue
        done = seg.feed(block)
        if done:
            return done[0]
        heard = heard or (time.monotonic() if seg.vad.segments(seg.buf) else 0.0)
        if not heard and time.monotonic() - began > wait_s:
            return None
    done = seg.feed(np.zeros(0, np.float32), final=True)
    return done[0] if done else None


def answer(
    utterance: dict, ask: Callable[[str, dict], str], trigger: str, cancel: threading.Event | None = None
) -> dict:
    """Turn a recorded request into a spoken reply, and time each stage.

    Args:
        utterance (dict): From `request()`.
        ask (Callable[[str, dict], str]): Takes the text and the transcript details, returns the reply text.
        trigger (str): "wake word", "talk" or "follow-up".
        cancel (threading.Event | None, optional): Set when a newer turn barges in: the reply is then not spoken.
            Defaults to None.

    Returns:
        dict: The voice.turn event: `text`, `reply`, the seconds of each stage and `interrupted`, plus `spoken`
            (the result of `speaker.speak`, or None when nothing was said).
    """
    cancel = cancel or threading.Event()
    t0 = time.time()
    text = stt.transcribe(utterance["samples"]).strip()
    stt_s = time.time() - t0
    if not text:
        turn = events.emit("voice.turn", trigger=trigger, text="", stt_s=round(stt_s, 2))
        if cancel.is_set() or trigger == "follow-up":  # a noise after the reply ends the conversation quietly
            return {**turn, "spoken": None}
        return {**turn, "spoken": speaker.speak("Sorry, I did not catch that.", reason="no words", cancel=cancel)}
    details = {"engine": config.STT_ENGINE, "took_s": round(stt_s, 2), "audio_s": len(utterance["samples"]) / 16000}
    t1 = time.time()
    reply = ask(text, details).strip() or "Done."
    reply_s = time.time() - t1
    turn = events.emit(
        "voice.turn",
        trigger=trigger,
        text=text,
        reply=reply,
        end_of_speech_ts=utterance["end_ts"],
        stt_s=round(stt_s, 2),
        reply_s=round(reply_s, 2),
        interrupted=cancel.is_set(),
    )
    spoken = None if cancel.is_set() else speaker.speak(reply, reason=f"answer to: {text}", cancel=cancel)
    return {**turn, "spoken": spoken}


class WakeLoop:
    """Listens for "Hey Jarvis" (when a wake model exists) and runs one turn per wake, Talk press or follow-up."""

    def __init__(self, ask: Callable[[str, dict], str], use_wake: bool = True, conversation: bool = False):
        """Prepare the loop without opening the microphone.

        Args:
            ask (Callable[[str, dict], str]): Writes the reply to a request (see `answer`).
            use_wake (bool, optional): Listen for the wake word. False = only the Talk button. Defaults to True.
            conversation (bool, optional): After each reply, listen for a follow-up without the wake word. It can
                change while the loop runs. Defaults to False.

        Raises:
            FileNotFoundError: `use_wake` is True and a wake-word model file is missing.
        """
        self.ask = ask
        self.detector = wake.Detector() if use_wake else None
        self.blocks: queue.Queue = queue.Queue()
        self.talk_now = threading.Event()
        self.follow_up = threading.Event()  # set when a reply has been heard: the next request needs no wake word
        self.conversation = conversation
        self.stop_flag = threading.Event()
        self.thread: threading.Thread | None = None
        self.cancel = threading.Event()  # the cancel flag of the turn that is answered now
        self.worker: threading.Thread | None = None  # the thread that answers it

    def start(self) -> None:
        """Start listening in a background thread."""
        self.thread = threading.Thread(target=self._run, name="wake", daemon=True)
        self.thread.start()
        events.emit("voice.wake", state="listening" if self.detector else "talk only")

    def stop(self) -> None:
        """Stop listening, close the microphone, and drop the reply of a turn that is still answered."""
        self.stop_flag.set()
        self.cancel.set()
        events.emit("voice.wake", state="off")

    def talk(self) -> None:
        """Start a turn now, as if the wake word was heard (push to talk)."""
        self.talk_now.set()

    def _next(self) -> np.ndarray | None:
        try:
            return self.blocks.get(timeout=0.5)
        except queue.Empty:
            return None

    def _turn(self, trigger: str) -> None:
        """Interrupt the turn before, record the request, and answer it in a worker thread."""
        self.follow_up.clear()
        if trigger != "follow-up":  # a follow-up starts after the reply ended: nothing to interrupt
            # Barge-in. The cancel flag is set before the stop event, so the old turn cannot start speaking after it.
            self.cancel.set()
            speaker.stop(trigger)
        events.emit("voice.listening", trigger=trigger)
        wait = config.VOICE_FOLLOW_UP_S if trigger == "follow-up" else None
        utterance = request(self._next, time.time(), wait)
        if utterance is None:
            note = "conversation ended" if trigger == "follow-up" else "no speech after the wake"
            events.emit("voice.turn", trigger=trigger, text="", note=note)
            return
        # The request is recorded: the dashboard shows "thinking" until the reply is spoken.
        events.emit("voice.recorded", trigger=trigger, audio_s=round(len(utterance["samples"]) / 16000, 2))
        self.cancel = threading.Event()
        self.worker = threading.Thread(
            target=self._answer, args=(utterance, trigger, self.cancel), name="voice-turn", daemon=True
        )
        self.worker.start()

    def _answer(self, utterance: dict, trigger: str, cancel: threading.Event) -> None:
        """Answer one request (the worker thread)."""
        try:
            turn = answer(utterance, self.ask, trigger, cancel)
        except Exception as e:  # noqa: BLE001 - one failed turn must not stop the loop
            events.emit("voice.error", error=f"voice turn failed: {type(e).__name__}: {e}")
            return
        self._open_follow_up(turn["spoken"], cancel)

    def _open_follow_up(self, spoken: dict | None, cancel: threading.Event) -> None:
        """In a conversation, let the next request skip the wake word once this reply has been heard."""
        if not (self.conversation and config.VOICE_FOLLOW_UP_S > 0 and spoken):
            return
        if not speaker.wait_heard(spoken, cancel):
            return
        time.sleep(config.VOICE_ECHO_TAIL_S)  # the speaker's delay and the room's echo end too
        if self.conversation and not cancel.is_set() and not self.stop_flag.is_set():
            self.follow_up.set()

    def _woke(self, block: np.ndarray | None) -> str | None:
        """Return why a turn starts now ("talk", "follow-up" or "wake word"), or None."""
        if self.talk_now.is_set():
            self.talk_now.clear()
            return "talk"
        if self.follow_up.is_set():
            return "follow-up"
        return "wake word" if block is not None and self.detector and self.detector.heard(block) else None

    def _run(self) -> None:
        """Open the microphone while the pause switch is off, and start a turn on each wake."""
        import sounddevice as sd

        while not self.stop_flag.is_set():
            if audio.paused():
                time.sleep(0.5)
                continue
            push = lambda data, *_: self.blocks.put(data[:, 0].copy())  # noqa: E731
            with sd.InputStream(samplerate=16000, channels=1, dtype="float32", blocksize=BLOCK, callback=push):
                self._listen()

    def _listen(self) -> None:
        """Feed blocks to the detector until the pause switch or stop() closes the stream.

        A turn returns as soon as its request is recorded, so the detector also hears the user while the agent
        answers.
        """
        while not self.stop_flag.is_set() and not audio.paused():
            trigger = self._woke(self._next())
            if trigger:
                self._turn(trigger)
                if self.detector:
                    self.detector.reset()  # the wake phrase and the request are not heard again
