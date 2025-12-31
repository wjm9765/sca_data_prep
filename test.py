import os
import soundfile as sf
import numpy as np
from src.sca_data.dataset_utils import easy_load

def main():
    # 저장할 디렉토리 생성
    output_dir = "./testaudio"
    os.makedirs(output_dir, exist_ok=True)

    print("="*50)
    print("1. 데이터셋 로드")
    print("="*50)

    # --- [Talker Dataset] 로드 ---
    print("[Loading] Talker Dataset (format='talker_chat')...")
    ds_talker = easy_load(format="talker_chat")
    
    # 데이터셋이 비어있는지 확인
    if len(ds_talker) == 0:
        print("❌ 데이터셋이 비어있습니다.")
        return

    print("\n" + "="*50)
    print("2. Talker 데이터셋 정답 오디오(Target) 5개 저장")
    print("="*50)

    # 최대 5개, 혹은 데이터셋 전체 길이만큼 반복
    num_samples = 5
    loop_count = min(num_samples, len(ds_talker))

    for i in range(loop_count):
        # i번째 샘플 가져오기
        sample_messages = ds_talker[i]['messages']
        
        # Talker 포맷의 Assistant 메시지 가져오기 (보통 Index 2)
        # 구조: User(Audio+Text) -> Assistant(Text+Audio)
        assistant_msg = sample_messages[2] 
        
        # 역할이 assistant가 맞는지 확인 (안전장치)
        if assistant_msg['role'] != 'assistant':
            print(f"[{i}] Warning: Index 2 is not assistant message. Skipping.")
            continue

        content_list = assistant_msg['content']
        
        # 오디오 찾기
        audio_data = None
        sampling_rate = 16000

        for item in content_list:
            # content 리스트 안에서 type이 audio인 것을 찾음
            if isinstance(item, dict) and item.get('type') == 'audio':
                audio_data = item['audio_waveform']
                sampling_rate = item.get('sampling_rate', 16000)
                break
        
        # 저장 로직
        if audio_data is not None:
            # 파일명에 인덱스 추가 (target_output_0.wav, target_output_1.wav ...)
            save_filename = f"target_output_{i}.wav"
            save_path = os.path.join(output_dir, save_filename)
            
            # soundfile로 저장
            sf.write(save_path, audio_data, sampling_rate)
            
            print(f"[{i}] ✅ 저장 완료: {save_path}")
            print(f"      - Duration: {len(audio_data)/sampling_rate:.2f} sec")
        else:
            print(f"[{i}] ❌ 오디오 데이터를 찾을 수 없습니다.")

    print("\n[Done] 모든 작업이 완료되었습니다.")

if __name__ == "__main__":
    main()