import math
import subprocess
import time
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor


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


if __name__ == "__main__":
    import sys
    args = [a for a in sys.argv[1:] if a not in ("--scan", "--parallel")]
    do_scan = "--scan" in sys.argv
    do_parallel = "--parallel" in sys.argv
    workers = 4
    if "--parallel" in sys.argv:
        try:
            i = sys.argv.index("--parallel")
            if i + 1 < len(sys.argv) and sys.argv[i + 1].isdigit():
                workers = int(sys.argv[i + 1])
        except (ValueError, IndexError):
            pass
    if args:
        video_path = args[0]
    else:
        video_path = "test_video.mp4"
        print("사용법: python key_prame_check_multi.py <영상경로> [--scan] [--parallel [N]]")
        print("  --scan     : 키프레임 추출 후 삼분할 구간 스캔")
        print("  --parallel : 멀티프로세싱 스캔 (기본 워커 4, --parallel 8 등으로 지정)")
        print(f"경로 미지정 시 기본값: {video_path}\n")

    timestamps = get_keyframe_timestamps(video_path)
    if timestamps:
        print("초반 10개 키프레임:", timestamps[:10])
    if do_scan and timestamps:
        if do_parallel:
            parallel_scan_with_keyframes(video_path, timestamps, workers=workers, jump_step=15)
        else:
            scan_video_with_keyframes(video_path, timestamps, jump_step=15)