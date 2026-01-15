#!/usr/bin/env -S uv run python

import numpy as np
import soundfile as sf
from pathlib import Path
from transformers import Qwen3OmniMoeProcessor
from sca_data.dataset_utils import easy_load

# -------------------------------------------------------------------------
# [설정] 저장 경로 및 샘플 개수
# -------------------------------------------------------------------------
OUTPUT_DIR = Path("./test_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
NUM_SAMPLES_TO_SAVE = 5  # 몇 개의 샘플을 저장할지 설정

# -------------------------------------------------------------------------
# [Import] sca_data 패키지 로드
# -------------------------------------------------------------------------


def main():
    print(">>> [1/3] 데이터셋 로드 중 (Format: Duplex)...")
    try:
        # 데이터셋 로드
        dataset = easy_load(format="duplex")
        print(f"✅ 데이터셋 로드 완료. 총 샘플 수: {len(dataset)}")
    except Exception as e:
        print(f"❌ 데이터셋 로드 실패: {e}")
        return

    print(">>> [2/3] 토크나이저 로드 중 (텍스트 디코딩용)...")
    try:
        processor = Qwen3OmniMoeProcessor.from_pretrained(
            "Qwen/Qwen3-Omni-30B-A3B-Instruct", trust_remote_code=True
        )
        tokenizer = processor.tokenizer  # type:ignore
    except Exception as e:
        print(f"⚠️ 토크나이저 로드 실패: {e}")
        print("   -> 텍스트 디코딩 기능이 제한됩니다.")
        tokenizer = None

    print(f">>> [3/3] 상위 {NUM_SAMPLES_TO_SAVE}개 샘플 디코딩 및 저장 시작...")

    # 지정한 개수만큼 반복
    for i in range(min(NUM_SAMPLES_TO_SAVE, len(dataset))):
        # 1. 데이터 Row 가져오기
        row = dataset[i]["dataset_row_obj"]

        print(f"   Processing Sample {i}...")

        # ---------------------------------------------------------
        # (A) Target Audio 복원 (이어 붙이기)
        # ---------------------------------------------------------
        target_segments = []
        for seg in row.target_audios:
            # seg.audio.waveform은 numpy array
            target_segments.append(seg.audio.waveform)

        if target_segments:
            # 끊겨있는 세그먼트들을 하나로 이어 붙여서 듣기 편하게 만듦
            full_target_wav = np.concatenate(target_segments)

            wav_filename = OUTPUT_DIR / f"sample_{i}_target.wav"
            sf.write(wav_filename, full_target_wav, 24000)  # Target은 24kHz
        else:
            print(f"      [Warning] Sample {i} has no target audio segments.")

        # ---------------------------------------------------------
        # (B) Text Transcript 복원 (전체 시퀀스 디코딩)
        # ---------------------------------------------------------
        if tokenizer:
            # [수정됨] -100 (Audio Placeholder) 토큰 제거 후 디코딩
            # 이유: 토크나이저는 음수(-100)를 처리하지 못해 OverflowError 발생
            valid_ids = [tid for tid in row.input_sequence if tid != -100]

            try:
                full_text = tokenizer.decode(valid_ids)
            except Exception as e:
                full_text = f"[Decoding Error]: {e}"

            txt_filename = OUTPUT_DIR / f"sample_{i}_transcript.txt"
            with open(txt_filename, "w", encoding="utf-8") as f:
                f.write(f"Sample Index: {i}\n")
                f.write(f"Total Sequence Length: {len(row.input_sequence)}\n")
                f.write(f"Valid Text Tokens: {len(valid_ids)}\n")
                f.write("=" * 80 + "\n\n")
                f.write(full_text)
                f.write("\n\n" + "=" * 80 + "\n")
                f.write("[Note] 오디오(-100) 구간은 텍스트에서 생략되었습니다.\n")

    print(f"\n🎉 모든 작업 완료! 결과물은 '{OUTPUT_DIR}' 폴더를 확인하세요.")


if __name__ == "__main__":
    main()
