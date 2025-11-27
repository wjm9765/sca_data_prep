import base64
import csv
import io
import re
from pathlib import Path
from typing import Optional, List

import numpy as np
import soundfile as sf
import tensorflow as tf

from .models.audio import AudioSlice
from .models.events import ComedianEvent, ComedySession, AudienceEvent


def get_video_id(url: str) -> Optional[str]:
    matched = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})(?:&|\/|$)", url)
    if matched:
        return matched.group(1)
    return None

def get_sliced_audio_base64(base_path: Path, slice: AudioSlice) -> str:
    audio_path = base_path / slice.file
    info = sf.info(audio_path)
    sr = info.samplerate

    start_frame = int(slice.start_time * sr)
    stop_frame = int(slice.end_time * sr)

    data, _ = sf.read(audio_path, start=start_frame, stop=stop_frame, dtype='int16')

    buffered = io.BytesIO()
    sf.write(buffered, data, sr, format='WAV')

    return base64.b64encode(buffered.getvalue()).decode()


def cut_audio_base64(base_path: Path, slice: AudioSlice, sample_interval: int = 4) -> List[str]:
    """
    Cut audio into segments optimized for Qwen3-Omni AuT encoder.

    Args:
        sample_interval: Duration of each segment in seconds (default: 6s)
                        Recommended range: 4-8 seconds for AuT's attention window
    """
    audio_path = base_path / slice.file
    info = sf.info(audio_path)
    sr = info.samplerate

    start_frame = int(slice.start_time * sr)
    stop_frame = int(slice.end_time * sr)

    total_frames = stop_frame - start_frame
    segment_frames = int(sample_interval * sr)

    segments = []
    for i in range(0, total_frames, segment_frames):
        segment_start = start_frame + i
        segment_stop = min(segment_start + segment_frames, stop_frame)

        data, _ = sf.read(audio_path, start=segment_start, stop=segment_stop, dtype='int16')

        buffered = io.BytesIO()
        sf.write(buffered, data, sr, format='WAV')
        segments.append(base64.b64encode(buffered.getvalue()).decode())

    return segments


def class_names_from_csv(class_map_csv_text):
    class_names = []
    with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row['display_name'])
    return class_names


def to_comedian_event(whisper_segment: dict) -> ComedianEvent:
    return ComedianEvent(
        event_type="speech",
        start=whisper_segment["start"],
        end=whisper_segment["end"],
        delivery_tag=None,
        content=whisper_segment['text'],
    )


def build_comedy_session(video_id: str, comedian_events: List[ComedianEvent], audience_events: List[AudienceEvent]) -> ComedySession:
    timeline = comedian_events + audience_events
    timeline.sort(key=lambda event: event.start)

    return ComedySession(
        video_id=video_id,
        timeline=timeline
    )


class RelationalAudioLoader:
    def __init__(self, audio_dataset):
        self.audio_lookup = {
            row['session_id']: row['audio']['bytes']
            for row in audio_dataset
        }

    def __call__(self, batch):
        audio_arrays = []
        sampling_rates = []

        for session_id, start, end in zip(batch['session_id'], batch['start_sec'], batch['end_sec']):
            try:
                raw_bytes = self.audio_lookup.get(session_id)

                if raw_bytes is None:
                    raise ValueError(f"Session {session_id} not found in storage")

                with io.BytesIO(raw_bytes) as file_obj:
                    with sf.SoundFile(file_obj) as f:
                        sr = f.samplerate
                        start_frame = int(start * sr)
                        frames_to_read = int((end - start) * sr)

                        if frames_to_read <= 0:
                            audio_arrays.append(np.array([0.0], dtype=np.float32))
                            sampling_rates.append(sr)
                            continue

                        f.seek(start_frame)
                        y = f.read(frames=frames_to_read, dtype='float32')
                        if y.ndim > 1: y = y.mean(axis=1)

                        audio_arrays.append(y)
                        sampling_rates.append(sr)

            except Exception as e:
                print(f"Error: {e}")
                audio_arrays.append(np.array([0.0], dtype=np.float32))
                sampling_rates.append(16000)

        batch["audio"] = [{"array": arr, "sampling_rate": sr} for arr, sr in zip(audio_arrays, sampling_rates)]
        return batch
