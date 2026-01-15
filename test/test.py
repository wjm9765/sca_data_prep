#!/usr/bin/env -S uv run python

import numpy as np
from pathlib import Path
from tqdm import tqdm
import textwrap
from transformers import Qwen3OmniMoeProcessor

# =============================================================================
# [설정] 사용자 환경에 맞게 수정
# =============================================================================
DEFAULT_INPUT_DIR = Path("./Multi-stream Spontaneous Conversation Training Dataset")
NUM_SAMPLES_TO_CHECK = None  # None = 전체 데이터 검증 (엄격 모드)

# [Import] 원본 함수 호출 방식 유지
try:
    from src.sca_data.dataset_utils import easy_load
except ImportError:
    from sca_data.dataset_utils import easy_load


def print_separator(title):
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")


def verify_dataset():
    print_separator("데이터셋 로드 및 정밀 검증 시작")

    # -------------------------------------------------------------------------
    # 0. 토크나이저 로드 (시스템 프롬프트 디코딩 및 검증용)
    # -------------------------------------------------------------------------
    print(">>> 토크나이저 로드 중 (Qwen/Qwen3-Omni-30B-A3B-Instruct)...")
    try:
        processor = Qwen3OmniMoeProcessor.from_pretrained(
            "Qwen/Qwen3-Omni-30B-A3B-Instruct", trust_remote_code=True
        )
        tokenizer = processor.tokenizer
    except Exception as e:
        print(f"⚠️ 토크나이저 로드 실패: {e}")
        return

    # -------------------------------------------------------------------------
    # 1. 데이터셋 로드 (dataset_utils.easy_load 사용)
    # -------------------------------------------------------------------------
    try:
        ds = easy_load(format="duplex")

        total_len = len(ds)
        if NUM_SAMPLES_TO_CHECK is not None and NUM_SAMPLES_TO_CHECK < total_len:
            print(f"✂️  설정에 따라 앞부분 {NUM_SAMPLES_TO_CHECK}개만 검증합니다.")
            ds = ds.select(range(NUM_SAMPLES_TO_CHECK))
        else:
            print(f"🔍 전체 데이터셋 {total_len}개를 검증합니다.")

        print(f"✅ 데이터셋 로드 성공! 총 샘플 수: {len(ds)}")
    except Exception as e:
        print(f"❌ 데이터셋 로드 실패: {e}")
        return

    # -------------------------------------------------------------------------
    # 검증 변수 및 상수
    # -------------------------------------------------------------------------
    stats = {
        "max_seq_len": 0,
        "min_seq_len": 999999,
        "total_tokens": 0,
        # 에러 카운터
        "over_40k_count": 0,  # 4만 토큰 초과
        "short_target_audio": 0,  # 1초 미만 Target Audio (통계용)
        "sr_mismatch": 0,  # SR 불일치 (16k/24k)
        "zero_embedding": 0,  # 임베딩 0
        "sys_prompt_error": 0,  # 시스템 프롬프트 누락/불일치
        "audio_count_mismatch": 0,  # Audio Object * 4 != Token Count
        "structure_pattern_error": 0,  # 4:2 / 4:1 패턴 깨짐
    }

    # Config 상수 (dataset_utils 설정과 일치해야 함)
    AUDIO_TOKEN = -100
    SILENCE_TOKEN = 151646
    AUDIO_RATIO = 4
    TEXT_SLICE = 2

    # 시스템 프롬프트 기준값 (첫 번째 샘플에서 추출)
    ref_sys_prompt_ids = None

    # -------------------------------------------------------------------------
    # 2. 전체 데이터 순회 검증
    # -------------------------------------------------------------------------
    for i, sample in enumerate(tqdm(ds, desc="[Strict Check]")):
        row = sample["dataset_row_obj"]
        input_seq = row.input_sequence
        seq_len = len(input_seq)

        # 통계 집계
        stats["max_seq_len"] = max(stats["max_seq_len"], seq_len)
        stats["min_seq_len"] = min(stats["min_seq_len"], seq_len)
        stats["total_tokens"] += seq_len

        # =====================================================================
        # [Check 1] 시퀀스 길이 (40,000 토큰 이내)
        # =====================================================================
        if seq_len > 40000:
            stats["over_40k_count"] += 1
            if stats["over_40k_count"] == 1:
                print(f"\n❌ [Sample {i}] 길이 초과: {seq_len} tokens")

        # =====================================================================
        # [Check 2 & 7] Target Audio 검증 (저장 확인 & 1초 미만 체크)
        # =====================================================================
        if row.target_audios:
            for seg in row.target_audios:
                # 데이터가 실제 오디오(array)를 가지고 있는지 확인
                waveform = seg.audio.waveform
                sr = seg.audio.sampling_rate

                # SR 체크 (Assistant = 24k)
                if sr != 24000:
                    stats["sr_mismatch"] += 1

                # 길이 체크 (1초 미만은 카운트만, 에러 아님)
                duration = len(waveform) / sr
                if duration < 1.0:
                    stats["short_target_audio"] += 1

        # User Audio SR 체크 (User = 16k)
        if row.input_audios and row.input_audios[0].sampling_rate != 16000:
            stats["sr_mismatch"] += 1

        # =====================================================================
        # [Check 3] 오디오 리스트 개수 vs 토큰 개수 매칭
        # "list[audio]의 개수 4배가 -100개랑 같아야 함"
        # =====================================================================
        num_input_audios = len(row.input_audios)
        num_audio_tokens = input_seq.count(AUDIO_TOKEN)

        if num_input_audios * 4 != num_audio_tokens:
            stats["audio_count_mismatch"] += 1
            if stats["audio_count_mismatch"] == 1:
                print(
                    f"\n❌ [Sample {i}] 오디오 불일치: 객체 {num_input_audios}개 * 4 != 토큰 {num_audio_tokens}개"
                )

        # =====================================================================
        # [Check 5] Speaker Embedding (0으로 채워지지 않았는지)
        # =====================================================================
        emb = np.array(row.speaker_embedding)
        if np.all(emb == 0):
            stats["zero_embedding"] += 1
            if stats["zero_embedding"] == 1:
                print(f"\n❌ [Sample {i}] Speaker Embedding 실패 (All Zero)")

        # =====================================================================
        # [Check 6] 시스템 프롬프트 & 시퀀스 패턴 (4:1 or 4:2)
        # =====================================================================
        try:
            # 6-1. 시스템 프롬프트 확인
            try:
                first_audio_idx = input_seq.index(AUDIO_TOKEN)
            except ValueError:
                # 오디오가 아예 없는 경우 (빈 파일 등)
                continue

            current_sys_ids = input_seq[:first_audio_idx]

            if i == 0:
                # 첫 번째 샘플을 기준으로 설정 (Reference)
                ref_sys_prompt_ids = current_sys_ids
                decoded_sys = tokenizer.decode(ref_sys_prompt_ids)
                print(
                    f"\n🔹 [Sample 0] 감지된 시스템 프롬프트:\n{textwrap.fill(decoded_sys, width=80)}\n"
                )
            else:
                # 나머지 샘플은 Reference와 비교 (빠른 검증)
                if current_sys_ids != ref_sys_prompt_ids:
                    stats["sys_prompt_error"] += 1
                    if stats["sys_prompt_error"] == 1:
                        print(
                            f"\n❌ [Sample {i}] 시스템 프롬프트가 Sample 0과 다릅니다."
                        )

            # 6-2. 본문 패턴 확인 (4 Audio -> 1 Silence or 2 Text)
            body_seq = input_seq[first_audio_idx:]
            cursor = 0

            while cursor < len(body_seq):
                # (Step A) 오디오 4개 확인
                audio_part = body_seq[cursor : cursor + AUDIO_RATIO]

                # 마지막 자투리가 남을 수 있으므로 길이 체크
                if len(audio_part) < AUDIO_RATIO:
                    # 정확히 4개 단위로 안 끝나면 에러로 볼 것인지?
                    # 보통 마지막엔 잘릴 수 있으니 패스, 하지만 -100이 섞여있으면 안됨.
                    if any(t != AUDIO_TOKEN for t in audio_part):
                        stats["structure_pattern_error"] += 1
                    break

                if not all(t == AUDIO_TOKEN for t in audio_part):
                    stats["structure_pattern_error"] += 1
                    if stats["structure_pattern_error"] == 1:
                        print(
                            f"\n❌ [Sample {i}] 오디오 패턴 깨짐 (4연속 아님): {audio_part}"
                        )
                    break

                cursor += AUDIO_RATIO

                # (Step B) 텍스트/침묵 확인
                if cursor >= len(body_seq):
                    break
                first_token = body_seq[cursor]

                if first_token == SILENCE_TOKEN:
                    # 침묵은 1개
                    cursor += 1
                else:
                    # 텍스트는 2개 (오디오 토큰이 섞이면 안됨)
                    text_part = body_seq[cursor : cursor + TEXT_SLICE]

                    if len(text_part) < TEXT_SLICE:
                        break  # 끝부분 도달

                    if any(t == AUDIO_TOKEN for t in text_part):
                        stats["structure_pattern_error"] += 1
                        if stats["structure_pattern_error"] == 1:
                            print(
                                f"\n❌ [Sample {i}] 텍스트 위치에 오디오 토큰 발견: {text_part}"
                            )
                        break

                    cursor += TEXT_SLICE

        except Exception as e:
            print(f"⚠️ [Sample {i}] 패턴 검증 중 예외: {e}")
            stats["structure_pattern_error"] += 1

    # ---------------------------------------------------------------------
    # 최종 리포트
    # ---------------------------------------------------------------------
    avg_len = stats["total_tokens"] / len(ds) if len(ds) > 0 else 0

    print_separator("📊 토큰 길이 통계")
    print(f"▶ 최소 길이: {stats['min_seq_len']} tokens")
    print(f"▶ 최대 길이: {stats['max_seq_len']} tokens (Limit: 40000)")
    print(f"▶ 평균 길이: {avg_len:.2f} tokens")

    print_separator("🛠 최종 검증 결과")

    def status(count):
        return f"❌ {count} 건 발견" if count > 0 else "✅ 통과"

    print(f"1. [40k 초과]        : {status(stats['over_40k_count'])}")
    print(f"2. [SR 불일치]       : {status(stats['sr_mismatch'])}")
    print(
        f"3. [오디오 개수 매칭] : {status(stats['audio_count_mismatch'])} (List * 4 == Tokens)"
    )
    print(f"4. [패턴 구조 (4:2)] : {status(stats['structure_pattern_error'])}")
    print(f"5. [시스템 프롬프트]  : {status(stats['sys_prompt_error'])}")
    print(f"6. [임베딩 누락]      : {status(stats['zero_embedding'])}")
    print(
        f"7. [참고] 1초미만 타겟 : {stats['short_target_audio']} 건 (정상적인 짧은 대답)"
    )

    # 최종 판정 (1초 미만 오디오는 에러 아님)
    critical_errors = (
        stats["over_40k_count"]
        + stats["sr_mismatch"]
        + stats["audio_count_mismatch"]
        + stats["structure_pattern_error"]
        + stats["sys_prompt_error"]
        + stats["zero_embedding"]
    )

    if critical_errors == 0:
        print(
            "\n🎉🎉 [SUCCESS] 모든 엄격한 검증을 통과했습니다! 학습 가능한 데이터셋입니다. 🎉🎉"
        )
    else:
        print(
            f"\n🔥🔥 [FAILURE] 총 {critical_errors}개의 치명적인 문제가 발견되었습니다. 🔥🔥"
        )


if __name__ == "__main__":
    verify_dataset()
