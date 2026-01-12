#! /usr/bin/env -S uv run

from sca_data.dataset_utils import easy_load

dataset = easy_load(
    format="talker_chat"
)

item = dataset[20]
print(item)
user = item['messages'][1]
assistant = item['messages'][2]

user_audio = user['content'][0]['audio_waveform']
assistant_audio = assistant['content'][1]['audio_waveform']

user_sample_rate = user['content'][0]['sampling_rate']
assistant_sample_rate = assistant['content'][1]['sampling_rate']

print(f"User role: {user['role']}, Assistant role: {assistant['role']}")
print(f"User audio shape: {user_audio.shape}, Sample rate: {user_sample_rate}")
print(f"Assistant audio shape: {assistant_audio.shape}, Sample rate: {assistant_sample_rate}")

# save to file
import soundfile as sf
sf.write("user_audio.wav", user_audio.squeeze(), user_sample_rate)
sf.write("assistant_audio.wav", assistant_audio.squeeze(), assistant_sample_rate)

from tqdm import tqdm
assistant_sample_rate = 24000
user_sample_rate = 16000
min_assistant_audio_len_sec = 4.9
min_user_audio_len_sec = 10
assistant_min_samples = assistant_sample_rate * min_assistant_audio_len_sec
user_min_samples = user_sample_rate * min_user_audio_len_sec
for item in tqdm(dataset):
    system_segment = [val for val in item['messages'] if val['role'] == 'system']
    assert len(system_segment) == 1, "Each item must have exactly one system segment."
    assert isinstance(system_segment[0]['content'], str), "System content must have exactly one entry."

    user_segment = [val for val in item['messages'] if val['role']   == 'user']
    assert len(user_segment) == 1, "Each item must have exactly one user segment."
    user_content = user_segment[0]['content']
    assert len(user_content) == 2, "User content must have exactly two entries."
    user_audio_entries = [content for content in user_content if content['type'] == 'audio']
    assert len(user_audio_entries) == 1, "User content must have exactly one audio entry."
    user_audio = user_audio_entries[0]['audio_waveform']
    assert len(user_audio.shape) == 1, f"User audio must be mono, but got shape {user_audio.shape}."
    assert user_audio.shape[0] >= user_min_samples, f"User audio must be at least {min_user_audio_len_sec} seconds long, but got {user_audio.shape[0] / user_sample_rate} seconds."
    assert user_audio_entries[0]['sampling_rate'] == user_sample_rate, f"User audio sampling rate must be {user_sample_rate} Hz."

    assistant_segment = [val for val in item['messages'] if val['role'] == 'assistant']
    assert len(assistant_segment) == 1, "Each item must have exactly one assistant segment."
    assistant_content = assistant_segment[0]['content']
    assert len(assistant_content) == 2, "Assistant content must have exactly two entries."
    assistant_audio_entries = [content for content in assistant_content if content['type'] == 'audio']
    assert len(assistant_audio_entries) == 1, "Assistant content must have exactly one audio entry."
    assistant_audio = assistant_audio_entries[0]['audio_waveform']
    assert len(assistant_audio.shape) == 1, f"Assistant audio must be mono, but got shape {assistant_audio.shape}."
    assert assistant_audio.shape[0] >= assistant_min_samples, f"Assistant audio must be at least {min_assistant_audio_len_sec} seconds long, but got {assistant_audio.shape[0] / assistant_sample_rate} seconds."
    assert assistant_audio_entries[0]['sampling_rate'] == assistant_sample_rate, f"Assistant audio sampling rate must be {assistant_sample_rate} Hz."
