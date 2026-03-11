# ---------------------------------------------------------------------------
# key_prame_check_multi_merge.py
# ---------------------------------------------------------------------------
# 키프레임 기반 삼분할 구간 탐색 → 구간 리스트(.txt) 저장/로드 → 컷팅·병합(MP4)
# 현재는 __main__ 블록 내 변수(target_video, MODE 등)로 설정하며,
# 추후 argparse 등으로 CLI 옵션화할 때 아래 항목들을 인자로 매핑하면 됩니다.
#
# ---------------------------------------------------------------------------
# [입력 가능 항목 정리] (CLI 옵션 후보)
# ---------------------------------------------------------------------------
#
#  • 영상 경로 (필수)
#      target_video  : 원본 영상 파일 경로 (.ts, .mp4 등)
#
#  • 실행 모드
#      MODE = 1  : 스캔 + 구간 리스트(_seg.txt)만 저장 (영상 자르기 없음)
#      MODE = 2  : 스캔 + 리스트 저장 + 컷팅·병합(_merged.mp4)까지 한 번에 실행
#      MODE = 3  : 스캔 생략, 기존 _seg.txt 로드 후 병합만 실행
#
#  • 출력/경로
#      (리스트) 저장 경로 : 원본과 동일 디렉터리, 파일명_seg.txt / _seg(1).txt ... (넘버링)
#      (병합본) 저장 경로 : 원본과 동일 디렉터리, 파일명_merged.mp4 / _merged(1).mp4 ... (넘버링)
#      모드 3 시 로드할 txt : 기본값 = 원본 basename 기준 "_seg.txt" (수정 시 경로 변경)
#
# ---------------------------------------------------------------------------
# [멀티 프로세스 / 워커 관련 옵션] (자세히)
# ---------------------------------------------------------------------------
#
#  (1) 스캔 단계 — 멀티프로세스 (ProcessPoolExecutor)
#      사용처: parallel_scan_with_keyframes(..., workers=, jump_step=)
#
#      • workers (기본 4)
#          - 키프레임 배열을 N등분해, 각 청크를 별도 프로세스에서 스캔합니다.
#          - ProcessPoolExecutor(max_workers=workers) 로 워커 수 = 동시 실행 프로세스 수.
#          - 키프레임 개수 / workers 로 대략적인 청크 크기가 정해지고,
#            jump_step 의 배수로 정렬(aligned_chunk_size)해 구간을 나눕니다.
#          - 권장: CPU 코어 수에 맞게 4~16. 너무 크면 오버헤드, 너무 작으면 활용도 감소.
#
#      • jump_step (기본 15)
#          - 러프 스캔 시 키프레임 인덱스를 몇 칸씩 건너뛸지 지정합니다.
#          - 예: 15 → 0, 15, 30, 45 ... 번째 키프레임만 먼저 검사 후, 구간 발견 시 이분 탐색.
#          - 작을수록: 더 촘촘히 검사 → 정밀하지만 스캔 시간 증가.
#          - 클수록: 더 성긴 검사 → 빠르지만 짧은 삼분할 구간을 놓칠 수 있음.
#          - 청크 경계 정렬에도 사용되므로, workers 와 함께 조정하는 것이 좋습니다.
#
#  (2) 병합 단계 — 멀티스레딩 (ThreadPoolExecutor)
#      사용처: cut_and_merge_video(..., temp_dir=, max_workers=)
#
#      • temp_dir (기본 r"C:\temp")
#          - 조각(segment)을 잘라낼 때 사용하는 임시 디렉터리.
#          - NVMe 등 고속 디스크를 지정하면 원본(HDD) 읽기 후 쓰기를 빠른 디스크로 모아
#            I/O 병목을 줄일 수 있습니다. 병합 후 concat 리스트·임시 조각 파일은 삭제됩니다.
#
#      • max_workers (기본 4)
#          - FFmpeg 로 각 구간을 잘라내는 작업을 동시에 돌리는 스레드 수.
#          - ThreadPoolExecutor(max_workers=max_workers) 로, 여러 개의 ffmpeg -ss -t ... 가
#            동시에 실행되어 temp_dir 에 조각 파일을 씁니다.
#          - 권장: 디스크 쓰기 성능과 CPU 부담을 고려해 4~8. 원본이 네트워크/HDD면 과도한 동시 접근 시 오히려 느려질 수 있음.
#
#  (3) 요약
#      - 스캔: 멀티프로세스(workers) + jump_step 으로 키프레임 구간을 병렬 탐색.
#      - 병합: 멀티스레드(max_workers)로 구간별 컷팅을 병렬 실행 후, concat 으로 한 파일로 병합.
#
# ---------------------------------------------------------------------------

import math
import os
import subprocess
import time
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


def _run_ffprobe(cmd, video_path):
    """공통: ffprobe 실행 (Windows 인코딩 대응)."""
    return subprocess.run(
        cmd + [video_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def get_keyframe_timestamps(video_path):
    """
    ffprobe로 영상의 모든 키프레임(I-Frame) 타임스탬프(초) 리스트를 반환합니다.
    방법 1 실패 시 패킷 플래그(K) 방식으로 재시도합니다 (.ts 등 호환).
    """
    print(f"[{video_path}] 키프레임 맵 추출 중...")

    # 방법 1: frame + skip_frame nokey (빠름, 일부 포맷에서 미출력)
    cmd1 = [
        "ffprobe", "-loglevel", "error",
        "-skip_frame", "nokey",
        "-select_streams", "v:0",
        "-show_entries", "frame=pkt_pts_time",
        "-of", "csv=print_section=0",
    ]
    result = _run_ffprobe(cmd1, video_path)
    keyframes = []
    for line in (result.stdout or "").splitlines():
        line = line.strip()
        if line:
            part = line.split(",")[-1].strip()
            try:
                keyframes.append(float(part))
            except ValueError:
                continue

    # 방법 2 폴백: packet의 pts_time + flags (K=키프레임). .ts 등에서 동작
    if not keyframes:
        print("  (.ts 등: packet 플래그 방식 사용)")
        cmd2 = [
            "ffprobe", "-loglevel", "error",
            "-select_streams", "v:0",
            "-show_entries", "packet=pts_time,flags",
            "-of", "csv=p=0",
        ]
        result2 = _run_ffprobe(cmd2, video_path)
        for line in (result2.stdout or "").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) >= 2 and "K" in (parts[1] or ""):
                try:
                    keyframes.append(float(parts[0]))
                except ValueError:
                    continue

    # 디버깅: 여전히 없으면 ffprobe 출력 확인
    if not keyframes:
        print("⚠️ 키프레임을 찾지 못했습니다.")
        print("  [returncode]", result.returncode)
        print("  [stdout 길이]", len(result.stdout or ""), "| 샘플:", repr((result.stdout or "")[:300]))
        print("  [stderr]", (result.stderr or "").strip() or "(비어있음)")
    else:
        print(f"✅ 총 {len(keyframes)}개 키프레임 (마지막: {keyframes[-1]:.2f}초)")

    return keyframes


def check_tripartite_orb(frame_bgr, orb, tol=3, min_matches=15):
    """삼분할 화면 여부를 ORB 특징점 매칭으로 판별 (orb 객체 재사용)."""
    h, w = frame_bgr.shape[:2]
    slice_w = w // 3
    left_img = frame_bgr[:, :slice_w]
    center_img = frame_bgr[:, slice_w : slice_w * 2]
    right_img = frame_bgr[:, slice_w * 2 :]

    kp_l, des_l = orb.detectAndCompute(left_img, None)
    kp_c, des_c = orb.detectAndCompute(center_img, None)
    kp_r, des_r = orb.detectAndCompute(right_img, None)
    if des_c is None:
        return False

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches_left = bf.match(des_c, des_l) if des_l is not None else []
    matches_right = bf.match(des_c, des_r) if des_r is not None else []
    left_valid = sum(
        1 for m in matches_left
        if abs(kp_c[m.queryIdx].pt[0] - kp_l[m.trainIdx].pt[0]) <= tol
        and abs(kp_c[m.queryIdx].pt[1] - kp_l[m.trainIdx].pt[1]) <= tol
    )
    right_valid = sum(
        1 for m in matches_right
        if abs(kp_c[m.queryIdx].pt[0] - kp_r[m.trainIdx].pt[0]) <= tol
        and abs(kp_c[m.queryIdx].pt[1] - kp_r[m.trainIdx].pt[1]) <= tol
    )
    return (left_valid >= min_matches) and (right_valid >= min_matches)


def check_frame_at_time(video_path, target_sec, orb):
    """지정된 초(sec)의 단일 프레임을 FFmpeg로 추출하여 ORB 판별을 수행합니다."""
    width, height = 480, 270
    cmd = [
        "ffmpeg",
        "-hwaccel", "auto",
        "-ss", str(target_sec),
        "-i", video_path,
        "-frames:v", "1",
        "-vf", f"scale={width}:{height}",
        "-f", "image2pipe",
        "-pix_fmt", "bgr24",
        "-vcodec", "rawvideo",
        "-loglevel", "quiet",
        "-",
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE)
    if not result.stdout:
        return False
    frame = np.frombuffer(result.stdout, dtype=np.uint8).reshape((height, width, 3))
    return check_tripartite_orb(frame, orb, tol=3, min_matches=15)


def find_exact_boundary(video_path, keyframes, left_idx, right_idx, orb, target_state):
    """
    이분 탐색으로 상태가 변하는 정확한 키프레임 인덱스를 찾습니다.
    target_state True: 삼분할 '시작' 지점, False: 삼분할 '종료' 지점.
    """
    result_idx = right_idx
    while left_idx <= right_idx:
        mid_idx = (left_idx + right_idx) // 2
        is_tri = check_frame_at_time(video_path, keyframes[mid_idx], orb)
        if is_tri == target_state:
            result_idx = mid_idx
            right_idx = mid_idx - 1
        else:
            left_idx = mid_idx + 1
    return result_idx


def worker_scan_keyframes(video_path, keyframes, start_idx, end_idx, jump_step=15):
    """
    개별 워커가 담당할 인덱스 구간(start_idx ~ end_idx)만 탐색합니다.
    (키프레임 배열 전체는 공유하지만, 반복문이 도는 구간만 나눕니다)
    """
    orb = cv2.ORB_create(nfeatures=500)
    segments = []
    is_tracking = False
    seg_start_time = 0.0

    for i in range(start_idx, end_idx, jump_step):
        current_time = keyframes[i]
        is_tri = check_frame_at_time(video_path, current_time, orb)

        if is_tri and not is_tracking:
            prev_idx = max(0, i - jump_step)
            exact_idx = find_exact_boundary(video_path, keyframes, prev_idx, i, orb, target_state=True)
            cut_idx = max(0, exact_idx - 1)
            seg_start_time = keyframes[cut_idx]
            is_tracking = True
        elif not is_tri and is_tracking:
            prev_idx = max(0, i - jump_step)
            exact_idx = find_exact_boundary(video_path, keyframes, prev_idx, i, orb, target_state=False)
            seg_end_time = keyframes[exact_idx]
            segments.append((seg_start_time, seg_end_time))
            is_tracking = False

    if is_tracking:
        segments.append((seg_start_time, keyframes[min(end_idx, len(keyframes) - 1)]))

    return segments


def parallel_scan_with_keyframes(video_path, keyframes, workers=4, jump_step=15):
    """키프레임 배열을 워커 수만큼 N등분하여 병렬 탐색합니다."""
    print(f"\n[🚀] 멀티프로세싱 탐색 시작 (워커: {workers}개)")
    start_time = time.time()

    total_frames = len(keyframes)
    chunk_size_raw = total_frames / workers
    aligned_chunk_size = math.ceil(chunk_size_raw / jump_step) * jump_step

    chunks = []
    for i in range(workers):
        start_idx = i * aligned_chunk_size
        if start_idx >= total_frames:
            break
        end_idx = min(start_idx + aligned_chunk_size, total_frames)
        chunks.append((start_idx, end_idx))

    print(f"인덱스 분할 현황: {chunks}")

    all_segments = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(worker_scan_keyframes, video_path, keyframes, c[0], c[1], jump_step)
            for c in chunks
        ]
        for future in futures:
            worker_segs = future.result()
            all_segments.extend(worker_segs)

    all_segments.sort(key=lambda x: x[0])
    merged_segments = []
    for seg in all_segments:
        if not merged_segments:
            merged_segments.append(seg)
        else:
            last_seg = merged_segments[-1]
            if seg[0] <= last_seg[1] + 5.0:
                merged_segments[-1] = (last_seg[0], max(last_seg[1], seg[1]))
            else:
                merged_segments.append(seg)

    elapsed = time.time() - start_time
    print("-" * 50)
    print(f"✅ 멀티프로세싱 탐색 완료! (소요 시간: {elapsed:.2f}초)")
    print(f"최종 컷 리스트: {merged_segments}")
    print("-" * 50)
    return merged_segments


def scan_video_with_keyframes(video_path, keyframes, jump_step=15):
    """키프레임 배열 기반 러프 스캔 → 이분 탐색으로 삼분할 구간을 찾습니다."""
    print(f"\n[🚀] 키프레임 기반 탐색 시작 (총 {len(keyframes)}개, {jump_step}칸 점프)")
    start_time = time.time()
    orb = cv2.ORB_create(nfeatures=500)
    segments = []
    is_tracking = False
    seg_start_time = 0.0

    for i in range(0, len(keyframes), jump_step):
        current_time = keyframes[i]
        is_tri = check_frame_at_time(video_path, current_time, orb)
        if is_tri and not is_tracking:
            prev_idx = max(0, i - jump_step)
            print(f"  [감지] {keyframes[prev_idx]:.2f}초 ~ {current_time:.2f}초 사이 시작점 이분탐색...")
            exact_idx = find_exact_boundary(video_path, keyframes, prev_idx, i, orb, target_state=True)
            cut_idx = max(0, exact_idx - 1)
            seg_start_time = keyframes[cut_idx]
            print(f"  [확정] 🎯 시작 컷: {seg_start_time:.2f}초 (인덱스 {cut_idx})")
            is_tracking = True
        elif not is_tri and is_tracking:
            prev_idx = max(0, i - jump_step)
            print(f"  [감지] {keyframes[prev_idx]:.2f}초 ~ {current_time:.2f}초 사이 종료점 이분탐색...")
            exact_idx = find_exact_boundary(video_path, keyframes, prev_idx, i, orb, target_state=False)
            seg_end_time = keyframes[exact_idx]
            print(f"  [확정] 🎯 종료 컷: {seg_end_time:.2f}초 (인덱스 {exact_idx})")
            segments.append((seg_start_time, seg_end_time))
            is_tracking = False

    if is_tracking:
        segments.append((seg_start_time, keyframes[-1]))

    elapsed = time.time() - start_time
    print("-" * 50)
    print(f"✅ 탐색 완료! (소요: {elapsed:.2f}초)")
    print(f"컷 리스트: {segments}")
    print("-" * 50)
    return segments


# ---------------------------------------------------------
# 파일명 중복 방지 (넘버링) 유틸리티
# ---------------------------------------------------------
def get_unique_filepath(original_path, suffix, ext):
    """
    원본 경로를 바탕으로 중복되지 않는 새 파일 경로를 생성합니다.
    예: 원본_seg.txt, 원본_seg(1).txt, 원본_seg(2).txt / 원본_merged.mp4, 원본_merged(1).mp4
    """
    base_dir = os.path.dirname(original_path)
    base_name = os.path.splitext(os.path.basename(original_path))[0]
    target_path = os.path.join(base_dir, f"{base_name}{suffix}{ext}")
    counter = 1
    while os.path.exists(target_path):
        target_path = os.path.join(base_dir, f"{base_name}{suffix}({counter}){ext}")
        counter += 1
    return target_path


def sec_to_hhmmss(sec):
    """초(float)를 HH:MM:SS.ms 문자열로 변환 (FFmpeg 스타일)."""
    h, rest = divmod(sec, 3600)
    m, rest = divmod(rest, 60)
    s_int = int(rest)
    frac = int(round((rest - s_int) * 1000))
    return f"{int(h):02d}:{int(m):02d}:{s_int:02d}.{frac:03d}"


def hhmmss_to_sec(s):
    """HH:MM:SS.ms 문자열을 초(float)로 변환."""
    parts = s.strip().split(":")
    if len(parts) != 3:
        return 0.0
    h, m, s_ms = int(parts[0]), int(parts[1]), parts[2]
    if "." in s_ms:
        s_part, ms_part = s_ms.split(".")
        sec = int(s_part) + int(ms_part.ljust(3, "0")[:3]) / 1000.0
    else:
        sec = int(s_ms)
    return 3600 * h + 60 * m + sec


# ---------------------------------------------------------
# 텍스트 파일 저장 및 불러오기 (넘버링 + HH:MM:SS 형식)
# ---------------------------------------------------------
def save_segments_to_txt(video_path, segments):
    """감지된 타임라인 리스트를 _seg.txt로 겹치지 않게 저장합니다. (넘버링 적용)"""
    txt_path = get_unique_filepath(video_path, "_seg", ".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for start, end in segments:
            duration_sec = int(round(end - start))
            f.write(f"{sec_to_hhmmss(start)} {sec_to_hhmmss(end)}  # {duration_sec}초\n")
    print(f"📝 [저장 완료] 분할 구간 텍스트: {txt_path}")
    return txt_path


def load_segments_from_txt(txt_path):
    """편집된 텍스트 파일을 읽어와서 (start, end) 초 단위 리스트로 반환합니다."""
    segments = []
    if not os.path.exists(txt_path):
        print(f"❌ [에러] 텍스트 파일을 찾을 수 없습니다: {txt_path}")
        return []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            # # 이후는 주석이므로 제거 후 파싱, 최소 2칸(시작 끝)만 있으면 사용
            line = line.split("#")[0].strip()
            parts = line.split()
            if len(parts) >= 2:
                segments.append((hhmmss_to_sec(parts[0]), hhmmss_to_sec(parts[1])))
    print(f"📂 [로드 완료] {len(segments)}개의 구간을 {os.path.basename(txt_path)}에서 불러왔습니다.")
    return segments


# ---------------------------------------------------------
# NVMe 활용 병렬 컷팅 워커 (멀티스레딩)
# ---------------------------------------------------------
def _worker_cut_segment(args):
    """단일 조각을 잘라내는 FFmpeg 서브프로세스 (ThreadPoolExecutor용)."""
    idx, start, end, video_path, temp_file = args
    duration = end - start
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", video_path,
        "-t", str(duration),
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        "-loglevel", "error",
        temp_file,
    ]
    subprocess.run(cmd)
    print(f"  - 조각 {idx + 1} 완료 ({duration:.2f}초 분량)")
    return temp_file


# ---------------------------------------------------------
# NVMe 최적화 멀티스레딩 병합 메인 함수
# ---------------------------------------------------------
def cut_and_merge_video(video_path, segments, temp_dir=r"C:\temp", max_workers=4):
    """
    목록을 바탕으로 빠른 NVMe(temp_dir)에 조각을 병렬로 잘라낸 뒤, 최종 병합합니다.
    출력 파일은 원본 폴더에 _merged.mp4 / _merged(1).mp4 ... 넘버링으로 저장됩니다.
    """
    if not segments:
        print("⚠️ 병합할 구간이 없습니다.")
        return
    output_mp4 = get_unique_filepath(video_path, "_merged", ".mp4")
    os.makedirs(temp_dir, exist_ok=True)
    base_name_only = os.path.splitext(os.path.basename(video_path))[0]

    print(f"\n✂️ 영상 병렬 자르기 시작 (총 {len(segments)}개 조각 -> {temp_dir} 활용)...")
    temp_files = []
    cut_tasks = []
    for i, (start, end) in enumerate(segments):
        temp_file = os.path.join(temp_dir, f"{base_name_only}_temp_part_{i}.mp4")
        temp_files.append(temp_file)
        cut_tasks.append((i, start, end, video_path, temp_file))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(_worker_cut_segment, cut_tasks))

    concat_txt = os.path.join(temp_dir, f"{base_name_only}_concat_list.txt")
    with open(concat_txt, "w", encoding="utf-8") as f:
        for tf in temp_files:
            safe_tf_path = tf.replace("\\", "/")
            f.write(f"file '{safe_tf_path}'\n")

    print(f"🔄 잘라낸 조각들을 하나로 합치는 중 (저장 위치: {os.path.dirname(output_mp4)})...")
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", concat_txt,
        "-c", "copy", "-loglevel", "error",
        output_mp4,
    ])

    try:
        os.remove(concat_txt)
    except OSError:
        pass
    for tf in temp_files:
        if os.path.exists(tf):
            try:
                os.remove(tf)
            except OSError:
                pass
    print(f"✅ [최종 완료] 하이라이트 영상 생성됨: {output_mp4}")


if __name__ == "__main__":
    import sys

    # 기본값 (CLI 미지정 시)
    target_video = r"H:\temp\recordWEB 1117a\chzzk\[2025-03-14] 고라니율 음메~ 1080p60.ts"
    MODE = 1

    # CLI 파싱: 영상경로(첫 번째 비옵션 인자), --mode 1|2|3
    argv = sys.argv[1:]
    positional = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--mode" and i + 1 < len(argv):
            try:
                MODE = int(argv[i + 1])
                if MODE not in (1, 2, 3):
                    MODE = 1
            except ValueError:
                pass
            i += 2
            continue
        if not a.startswith("-"):
            positional.append(a)
        i += 1

    if positional:
        target_video = positional[0]

    # [모드 1] "스캔 + 리스트(.txt)만 출력" (영상은 안 자름)
    if MODE == 1:
        timestamps = get_keyframe_timestamps(target_video)
        segments = parallel_scan_with_keyframes(target_video, timestamps, workers=4)
        save_segments_to_txt(target_video, segments)
        print("모드 1 종료: 텍스트 파일만 생성되었습니다.")

    # [모드 2] "스캔 + 리스트(.txt) 생성 + 컷팅 및 병합(MP4) 한 번에 실행"
    elif MODE == 2:
        timestamps = get_keyframe_timestamps(target_video)
        segments = parallel_scan_with_keyframes(target_video, timestamps, workers=4)
        save_segments_to_txt(target_video, segments)
        cut_and_merge_video(target_video, segments)

    # [모드 3] "스캔 생략하고 기존 리스트(.txt)를 읽어와서 병합만 실행"
    elif MODE == 3:
        # 원본과 같은 디렉터리의 _seg.txt 사용 (있으면 _seg.txt, 없으면 _seg(1).txt 등 탐색 가능)
        base = os.path.splitext(target_video)[0]
        txt_file = base + "_seg.txt"
        if not os.path.isfile(txt_file):
            # 넘버링된 파일 중 첫 번째 사용: _seg(1).txt, _seg(2).txt ...
            for n in range(1, 100):
                candidate = f"{base}_seg({n}).txt"
                if os.path.isfile(candidate):
                    txt_file = candidate
                    break
        edited_segments = load_segments_from_txt(txt_file)
        if edited_segments:
            cut_and_merge_video(target_video, edited_segments)
        else:
            print("⚠️ 로드된 구간이 없어 병합을 건너뜁니다.")