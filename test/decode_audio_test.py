#!/usr/bin/env -S uv run python

import sys
import os
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from transformers import Qwen3OmniMoeProcessor

# -------------------------------------------------------------------------
# [설정] 저장 경로 및 샘플 개수
# -------------------------------------------------------------------------
OUTPUT_DIR = Path("./test_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
NUM_SAMPLES_TO_SAVE = 5  # 몇 개의 샘플을 저장할지 설정

# -------------------------------------------------------------------------
# [Import] sca_data 패키지 로드
# -------------------------------------------------------------------------
try:
    # 현재 위치가 패키지 루트라면 바로 import
    from src.sca_data.dataset_utils import easy_load
except ImportError:
    # 아니라면 경로 추가 후 import
    current_dir = os.getcwd()
    src_path = os.path.join(current_dir, "src")
    if src_path not in sys.path:
        sys.path.append(src_path)
    from sca_data.dataset_utils import easy_load

def main():
    print(">>> [1/3] 데이터셋 로드 중 (Format: Duplex)...")
    try:
        # 데이터셋 로드 (캐시된 데이터가 있으면 자동으로 가져옴)
        dataset = easy_load(format="duplex")
        print(f"✅ 데이터셋 로드 완료. 총 샘플 수: {len(dataset)}")
    except Exception as e:
        print(f"❌ 데이터셋 로드 실패: {e}")
        return

    print(">>> [2/3] 토크나이저 로드 중 (텍스트 디코딩용)...")
    try:
        processor = Qwen3OmniMoeProcessor.from_pretrained(
            "Qwen/Qwen3-Omni-30B-A3B-Instruct", 
            trust_remote_code=True
        )
        tokenizer = processor.tokenizer
    except Exception as e:
        print(f"⚠️ 토크나이저 로드 실패: {e}")
        print("   -> 텍스트 디코딩 기능이 제한됩니다.")
        tokenizer = None

    print(f">>> [3/3] 상위 {NUM_SAMPLES_TO_SAVE}개 샘플 디코딩 및 저장 시작...")

    # 지정한 개수만큼 반복
    for i in range(min(NUM_SAMPLES_TO_SAVE, len(dataset))):
        
        # 1. 데이터 Row 가져오기
        row = dataset[i]["dataset_row_obj"]
        session_id = dataset[i]["session_id"]
        
        print(f"   Processing Sample {i} (Session: {session_id})...")

        # ---------------------------------------------------------
        # (A) Target Audio 복원 (이어 붙이기)
        # ---------------------------------------------------------
        target_segments = []
        for seg in row.target_audios:
            # seg.audio.waveform은 numpy array
            target_segments.append(seg.audio.waveform)
        
        if target_segments:
            # 끊겨있는 세그먼트들을 하나로 이어 붙여서 듣기 편하게 만듦
            # (실제 학습에선 끊겨 있지만, 사람이 듣기 위해 concat)
            full_target_wav = np.concatenate(target_segments)
            
            wav_filename = OUTPUT_DIR / f"sample_{i}_target.wav"
            sf.write(wav_filename, full_target_wav, 24000) # Target은 24kHz
        else:
            print(f"      [Warning] Sample {i} has no target audio segments.")

        # ---------------------------------------------------------
        # (B) Text Transcript 복원 (전체 시퀀스 디코딩)
        # ---------------------------------------------------------
        if tokenizer:
            full_text = tokenizer.decode(row.input_sequence)
            
            txt_filename = OUTPUT_DIR / f"sample_{i}_transcript.txt"
            with open(txt_filename, "w", encoding="utf-8") as f:
                f.write(f"Session ID: {session_id}\n")
                f.write(f"Total Sequence Length: {len(row.input_sequence)}\n")
                f.write("=" * 80 + "\n\n")
                f.write(full_text)
                f.write("\n\n" + "=" * 80 + "\n")
                f.write("[Note] <|audio_bos|>...<|audio_eos|> 태그나 특수 토큰이 보일 수 있습니다.\n")

    print(f"\n🎉 모든 작업 완료! 결과물은 '{OUTPUT_DIR}' 폴더를 확인하세요.")

if __name__ == "__main__":
    main()