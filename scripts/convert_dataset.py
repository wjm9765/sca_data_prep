#!/usr/bin/env -S uv run

from pathlib import Path
from typing import Tuple

from datasets import Dataset, Features, Value, Audio, DatasetDict

from sca_data.models.events import AudienceEvent, ComedianEvent, ComedySession, BaseEvent, EnvironmentEvent

DATASET_DIR = Path("./dataset").absolute()
SAVE_PATH = DATASET_DIR / "sca_comedy_dataset"
AUDIO_DIR = (DATASET_DIR / "audio_outputs").absolute()
INFERENCE_OUTPUTS = DATASET_DIR / "inference_outputs.jsonl"
assert DATASET_DIR.exists(), f"Dataset directory {DATASET_DIR} does not exist."
assert AUDIO_DIR.exists(), f"Audio directory {AUDIO_DIR} does not exist."
assert INFERENCE_OUTPUTS.exists(), f"Inference outputs file {INFERENCE_OUTPUTS} does not exist."

with open(INFERENCE_OUTPUTS, "r") as f:
    inference_outputs = [ComedySession.model_validate_json(line) for line in f.readlines()]

audio_exts = set(file.suffix for file in AUDIO_DIR.glob("*") if file.is_file())
assert len(audio_exts) == 1, f"Expected all audio files to have the same extension, but found: {audio_exts}"
audio_ext = audio_exts.pop()
del audio_exts


def remove_extras(session: ComedySession, remove_events: Tuple[BaseEvent] = (AudienceEvent, EnvironmentEvent)) -> ComedySession:
    filtered = []
    for event in session.timeline:
        if not isinstance(event, remove_events):
            filtered.append(event)

    return ComedySession(timeline=filtered, video_id=session.video_id)


def assert_overlap(session: ComedySession) -> None:
    overlap_threshold = 0.00001
    for i, event in enumerate(session.timeline):
        if len(session.timeline) - 1 == i:
            break
        next_event = session.timeline[i + 1]

        assert next_event.start + overlap_threshold >= event.end, f"Events overlap: {event} and {next_event}"


inference_outputs = [remove_extras(session) for session in inference_outputs]
for session in inference_outputs:
    assert_overlap(session)


# event data
event_rows = []
unique_sessions = {}
for session in inference_outputs:
    audio_path = AUDIO_DIR / f"{session.video_id}{audio_ext}"

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found for session {session.video_id} at {audio_path}")

    if session.video_id not in unique_sessions:
        unique_sessions[session.video_id] = str(audio_path)

    for i, event in enumerate(session.timeline):
        if isinstance(event, ComedianEvent) and event.event_type == 'speech':
            event_rows.append({
                "session_id": session.video_id,
                "start_sec": 0.0,
                "end_sec": event.start,
                "target_text": event.content,
                "event_index": i
            })
        else:
            raise ValueError(f"Unexpected event type in session {session.video_id}: {event}")

# audio data
audio_rows = []
for sess_id, path in unique_sessions.items():
    with open(path, "rb") as f:
        audio_rows.append({
            "session_id": sess_id,
            "audio": {"path": path, "bytes": f.read()}
        })

audio_features = Features({
    "session_id": Value("string"),
    "audio": Audio(decode=False)
})

ds_audio = Dataset.from_list(audio_rows, features=audio_features)

instruct_features = Features({
    "session_id": Value("string"),
    "start_sec": Value("float"),
    "end_sec": Value("float"),
    "target_text": Value("string"),
    "event_index": Value("int32"),
})

ds_instruct = Dataset.from_list(event_rows, features=instruct_features)

# save in one dataset with datasetdict
dataset_dict = DatasetDict({
    "storage": ds_audio,
    "train": ds_instruct,
})

dataset_dict.save_to_disk(SAVE_PATH)
