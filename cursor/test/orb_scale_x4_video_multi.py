import cv2
import numpy as np
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor


def get_video_duration(video_path):
    """ffprobe를 사용하여 영상의 전체 길이(초)를 가져옵니다."""
    cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return float(result.stdout.strip())


def check_tripartite_orb(frame_bgr, orb, tol=3, min_matches=15):
    """1/4 해상도 전용 ORB 판별 로직 (orb 객체를 인자로 받아 재사용)"""
    h, w = frame_bgr.shape[:2]
    slice_w = w // 3

    left_img = frame_bgr[:, :slice_w]
    center_img = frame_bgr[:, slice_w:slice_w*2]
    right_img = frame_bgr[:, slice_w*2:]

    kp_l, des_l = orb.detectAndCompute(left_img, None)
    kp_c, des_c = orb.detectAndCompute(center_img, None)
    kp_r, des_r = orb.detectAndCompute(right_img, None)

    if des_c is None:
        return False

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches_left = bf.match(des_c, des_l) if des_l is not None else []
    matches_right = bf.match(des_c, des_r) if des_r is not None else []

    left_valid, right_valid = 0, 0
    for m in matches_left:
        if abs(kp_c[m.queryIdx].pt[0] - kp_l[m.trainIdx].pt[0]) <= tol and \
           abs(kp_c[m.queryIdx].pt[1] - kp_l[m.trainIdx].pt[1]) <= tol:
            left_valid += 1
    for m in matches_right:
        if abs(kp_c[m.queryIdx].pt[0] - kp_r[m.trainIdx].pt[0]) <= tol and \
           abs(kp_c[m.queryIdx].pt[1] - kp_r[m.trainIdx].pt[1]) <= tol:
            right_valid += 1

    return (left_valid >= min_matches) and (right_valid >= min_matches)


def scan_video_chunk(video_path, start_sec, duration_sec, skip_seconds=30):
    """개별 프로세스가 할당받은 특정 구간만 탐색하는 워커 함수"""
    width, height = 480, 270
    frame_bytes = width * height * 3

    command = [
        'ffmpeg',
        '-hwaccel', 'auto',
        '-ss', str(start_sec),
        '-i', video_path,
        '-t', str(duration_sec),
        '-vf', f'fps=1/{skip_seconds},scale={width}:{height}',
        '-f', 'image2pipe',
        '-pix_fmt', 'bgr24',
        '-vcodec', 'rawvideo',
        '-loglevel', 'quiet',
        '-'
    ]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=10**8)
    orb = cv2.ORB_create(nfeatures=500)

    current_internal_sec = 0.0
    segments = []
    is_tracking = False
    seg_start = 0

    while True:
        in_bytes = process.stdout.read(frame_bytes)
        if not in_bytes or len(in_bytes) != frame_bytes:
            break

        frame = np.frombuffer(in_bytes, np.uint8).reshape((height, width, 3))
        is_tri = check_tripartite_orb(frame, orb)
        global_sec = start_sec + current_internal_sec

        if is_tri and not is_tracking:
            seg_start = global_sec
            is_tracking = True
        elif not is_tri and is_tracking:
            segments.append((seg_start, global_sec))
            is_tracking = False

        current_internal_sec += skip_seconds

    if is_tracking:
        segments.append((seg_start, start_sec + current_internal_sec))

    process.terminate()
    return segments


def parallel_scan_video(video_path, skip_seconds=30, workers=3):
    """
    영상을 N등분하여 병렬로 탐색합니다.
    :param workers: 동시 실행할 프로세스 수 (기본값 3: 네트워크 HDD용 안전 설정)
    """
    print(f"[{video_path}] 병렬 탐색 준비 중... (워커: {workers}개)")
    start_time = time.time()

    total_duration = get_video_duration(video_path)
    chunk_duration = total_duration / workers

    chunks = []
    for i in range(workers):
        start_sec = i * chunk_duration
        dur = chunk_duration if i < workers - 1 else total_duration - start_sec
        chunks.append((start_sec, dur))

    print(f"총 길이: {total_duration:.2f}초 | 분할 구간: {len(chunks)}개")

    all_segments = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(scan_video_chunk, video_path, c[0], c[1], skip_seconds)
            for c in chunks
        ]
        for idx, future in enumerate(futures):
            worker_segments = future.result()
            print(f"워커 {idx+1} 탐색 완료! 발견 구간: {len(worker_segments)}개")
            all_segments.extend(worker_segments)

    all_segments.sort(key=lambda x: x[0])
    merged_segments = []
    for seg in all_segments:
        if not merged_segments:
            merged_segments.append(seg)
        else:
            last_seg = merged_segments[-1]
            if seg[0] <= last_seg[1] + skip_seconds:
                merged_segments[-1] = (last_seg[0], max(last_seg[1], seg[1]))
            else:
                merged_segments.append(seg)

    elapsed = time.time() - start_time
    print("-" * 50)
    print(f"✅ 병렬 탐색 완료 (소요: {elapsed:.2f}초)")
    print(f"삼분할 구간 {len(merged_segments)}개:")
    for i, (s, e) in enumerate(merged_segments):
        print(f"  - 구간 {i+1}: {s:07.2f}초 ~ {e:07.2f}초 (길이: {e-s:.2f}초)")
    print("-" * 50)
    return merged_segments


# --- 실행 부분 ---
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("사용법: python orb_scale_x4_video_multi.py <영상경로> [skip_seconds=30] [workers=3]")
        sys.exit(1)
    video_path = sys.argv[1]
    skip_seconds = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    parallel_scan_video(video_path, skip_seconds=skip_seconds, workers=workers)