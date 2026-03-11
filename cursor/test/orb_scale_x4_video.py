import cv2
import numpy as np
import subprocess
import time

def check_tripartite_orb(frame_bgr, tol=3, min_matches=15):
    """1/4 해상도 전용 ORB 판별 로직"""
    h, w = frame_bgr.shape[:2]
    slice_w = w // 3

    left_img = frame_bgr[:, :slice_w]
    center_img = frame_bgr[:, slice_w:slice_w*2]
    right_img = frame_bgr[:, slice_w*2:]

    orb = cv2.ORB_create(nfeatures=500) # 1/4 해상도라 500개면 충분합니다

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
        pt_c = kp_c[m.queryIdx].pt
        pt_l = kp_l[m.trainIdx].pt
        if abs(pt_c[0] - pt_l[0]) <= tol and abs(pt_c[1] - pt_l[1]) <= tol:
            left_valid += 1

    for m in matches_right:
        pt_c = kp_c[m.queryIdx].pt
        pt_r = kp_r[m.trainIdx].pt
        if abs(pt_c[0] - pt_r[0]) <= tol and abs(pt_c[1] - pt_r[1]) <= tol:
            right_valid += 1

    return (left_valid >= min_matches) and (right_valid >= min_matches)


def scan_video_fast_ffmpeg(video_path, skip_seconds=30):
    """FFmpeg 파이프링을 이용한 네트워크 최적화 초고속 탐색"""
    print(f"탐색 시작: {video_path}")
    start_time = time.time()

    # 1080p의 1/4 해상도 (480x270)
    width, height = 480, 270
    frame_bytes = width * height * 3  # BGR 픽셀당 3바이트

    # FFmpeg 명령어 (30초당 1프레임 추출 & 1/4 스케일링을 FFmpeg 단에서 처리)
    # CUDA가 사용 가능할 경우 하드웨어 디코딩(-hwaccel auto)이 개입하여 더 빨라집니다.
    command = [
        'ffmpeg',
        '-hwaccel', 'auto',
        '-i', video_path,
        '-vf', f'fps=1/{skip_seconds},scale={width}:{height}',
        '-f', 'image2pipe',
        '-pix_fmt', 'bgr24',
        '-vcodec', 'rawvideo',
        '-loglevel', 'quiet',
        '-'
    ]

    # 파이프 열기
    process = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=10**8)
    
    current_sec = 0
    tripartite_segments = []
    is_tracking = False
    start_sec = 0

    while True:
        # 정확히 1프레임 분량의 바이트만 읽어옴
        in_bytes = process.stdout.read(frame_bytes)
        if not in_bytes:
            break  # 영상 끝

        # 바이트를 Numpy BGR 이미지 배열로 변환
        frame = np.frombuffer(in_bytes, np.uint8).reshape((height, width, 3))
        
        # 삼분할 판별 알고리즘 실행
        is_tri = check_tripartite_orb(frame)
        
        if is_tri and not is_tracking:
            print(f"[🔍] {current_sec}초: 삼분할 구간 시작 감지!")
            start_sec = current_sec
            is_tracking = True
        elif not is_tri and is_tracking:
            print(f"[⏹️] {current_sec}초: 삼분할 구간 종료. (저장: {start_sec}s ~ {current_sec}s)")
            tripartite_segments.append((start_sec, current_sec))
            is_tracking = False

        current_sec += skip_seconds

    # 영상이 삼분할인 상태로 끝난 경우 마무리 처리
    if is_tracking:
        tripartite_segments.append((start_sec, current_sec))

    process.terminate()
    
    elapsed = time.time() - start_time
    print("-" * 40)
    print(f"✅ 탐색 완료 (소요 시간: {elapsed:.2f}초)")
    print(f"찾아낸 삼분할 구간 리스트: {tripartite_segments}")
    print("-" * 40)

# --- 실행 부분 ---
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("사용법: python orb_scale_x4_video.py <영상경로> [skip_seconds=30]")
        sys.exit(1)
    video_path = sys.argv[1]
    skip_seconds = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    scan_video_fast_ffmpeg(video_path, skip_seconds=skip_seconds)