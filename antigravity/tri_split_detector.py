import sys
import time
import os
import subprocess
import tempfile
import numpy as np
import cv2


def _get_video_info_ffprobe(video_path):
    """ffprobe로 영상 길이(초)와 fps를 반환. decord 미사용으로 초기 로딩 0초."""
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-show_entries", "stream=r_frame_rate",
            "-select_streams", "v:0",
            "-of", "default=noprint_wrappers=1",
            video_path,
        ],
        capture_output=True,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe 실패: {out.stderr or out.stdout}")
    info = {}
    for line in (out.stdout or "").strip().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            info[k.strip()] = v.strip()
    duration_sec = float(info["duration"])
    rate_str = info["r_frame_rate"]
    if "/" in rate_str:
        num, den = rate_str.split("/")
        fps = float(num) / float(den)
    else:
        fps = float(rate_str)
    return duration_sec, fps


def _extract_frame_ffmpeg(video_path, timestamp_sec, out_path):
    """FFmpeg로 지정 시각에 1프레임만 추출해 out_path에 저장. -ss를 입력 앞에 두어 빠른 seek."""
    cmd = [
        "ffmpeg", "-y", "-nostdin",
        "-ss", str(timestamp_sec),
        "-i", video_path,
        "-vframes", "1",
        "-q:v", "2",
        out_path,
    ]
    r = subprocess.run(
        cmd,
        capture_output=True,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    if r.returncode != 0:
        raise RuntimeError(f"FFmpeg 프레임 추출 실패 (t={timestamp_sec}s): {r.stderr.decode(errors='replace')}")


def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

# matchTemplate 연산 가속: 이 배율로 축소한 뒤 검사 (1=원본, 4=1/4 해상도)
DETECT_SCALE = 4


class SplitVideoDetector:
    def __init__(self, video_path, y_start=540, y_end=720):
        self.video_path = os.path.abspath(video_path)
        duration_sec, self.fps = _get_video_info_ffprobe(self.video_path)
        self.total_frames = int(duration_sec * self.fps)

        # Y축 안전 영역 (상/하단의 팝업, 자막 등을 피해 비교할 영역)
        self.y_start = y_start
        self.y_end = y_end

    def _get_frame_at_index(self, frame_idx):
        """해당 프레임 인덱스의 1장만 FFmpeg로 추출해 RGB numpy 배열로 반환 (decord 대체)."""
        timestamp_sec = frame_idx / self.fps
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = f.name
        try:
            _extract_frame_ffmpeg(self.video_path, timestamp_sec, tmp_path)
            bgr = cv2.imread(tmp_path)
            if bgr is None:
                raise RuntimeError(f"프레임 읽기 실패: {tmp_path}")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _is_tri_split_at_frame(self, frame):
        """
        한 프레임이 삼분할 화면인지 여부만 판별. (좌측 템플릿이 우측과 0.95 이상 일치하면 True)
        frame: RGB numpy array (H, W, 3)
        """
        safe_frame = frame[self.y_start : self.y_end]
        h, w = safe_frame.shape[:2]
        safe_small = cv2.resize(
            safe_frame, (w // DETECT_SCALE, h // DETECT_SCALE), interpolation=cv2.INTER_AREA
        )
        template = safe_small[:, 0 : 600 // DETECT_SCALE]
        search_area = safe_small[:, 1260 // DETECT_SCALE : 1900 // DETECT_SCALE]
        res = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        return max_val >= 0.95

    def scan_all_segments_coarse(self, jump_sec=30):
        """
        영상 전체를 점프 스캔하여 삼분할 구간을 대략적으로 찾습니다.
        이진 탐색 없이, 점프 시점 기준으로 (시작, 끝)을 수집합니다.

        Returns:
            list of (start_sec, end_sec): 대략적인 구간 리스트 (초 단위).
            구간이 없으면 [].
        """
        start_time = time.time()
        jump_frames = int(jump_sec * self.fps)
        print(f"[전체 스캔] 점프 간격 {jump_sec}초, 해상도 1/{DETECT_SCALE}로 삼분할 판별합니다.")

        segments = []  # (start_frame, end_frame) 또는 (start_sec, end_sec)
        in_tri_split = False
        segment_start_frame = 0
        frame_idx = 0

        while frame_idx < self.total_frames:
            frame = self._get_frame_at_index(frame_idx)
            is_tri = self._is_tri_split_at_frame(frame)

            if not in_tri_split and is_tri:
                # 일반 → 삼분할 전환: 구간 시작
                segment_start_frame = frame_idx
                in_tri_split = True
            elif in_tri_split and not is_tri:
                # 삼분할 → 일반 전환: 구간 끝
                segment_end_frame = frame_idx
                start_sec = segment_start_frame / self.fps
                end_sec = segment_end_frame / self.fps
                segments.append((start_sec, end_sec))
                print(f"  구간 {len(segments)}: {format_time(start_sec)} ~ {format_time(end_sec)}")
                in_tri_split = False

            frame_idx += jump_frames

        # 영상 끝까지 삼분할이면 마지막 구간 닫기
        if in_tri_split:
            segment_end_frame = min(frame_idx, self.total_frames - 1)
            start_sec = segment_start_frame / self.fps
            end_sec = segment_end_frame / self.fps
            segments.append((start_sec, end_sec))
            print(f"  구간 {len(segments)}: {format_time(start_sec)} ~ {format_time(end_sec)} (영상 끝)")

        elapsed = time.time() - start_time
        print(f"[전체 스캔] 완료: 구간 {len(segments)}개 (소요 {elapsed:.3f}초)")
        return segments

    def find_first_tri_split(self):
        """
        영상을 30초 단위로 듬성듬성 스캔하다가 첫 번째 삼분할 화면을 발견하면
        시간과 양쪽 분할선의 정확한 X 좌표를 출력합니다.
        """
        start_time = time.time()
        print(f"[탐색] matchTemplate/col_diff는 해상도 1/{DETECT_SCALE}로 축소하여 연산합니다.")

        # 30초 간격으로 스캔 (초당 프레임수 * 30)
        jump_frames = int(30 * self.fps)
        frame_idx = 0
        
        while frame_idx < self.total_frames:
            frame = self._get_frame_at_index(frame_idx)
            safe_frame = frame[self.y_start:self.y_end]
            # 연산 속도: 1/DETECT_SCALE 해상도로 줄인 뒤 템플릿/차이 검사
            h, w = safe_frame.shape[:2]
            safe_small = cv2.resize(
                safe_frame, (w // DETECT_SCALE, h // DETECT_SCALE), interpolation=cv2.INTER_AREA
            )
            ws = w // DETECT_SCALE  # 480 for 1920

            # 삼분할 판단: 왼쪽(0~600) 템플릿을 오른쪽(1260~1900)에서 매칭
            template = safe_small[:, 0 : 600 // DETECT_SCALE]
            search_area = safe_small[:, 1260 // DETECT_SCALE : 1900 // DETECT_SCALE]

            res = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            # 일치율 0.95 이상이면 삼분할로 간주
            if max_val >= 0.95:
                # 1. 우측 분할 시작 X (축소 좌표 → 원본)
                right_x_start = 1260 + max_loc[0] * DETECT_SCALE

                # 2. 좌측 분할선 끝 X (축소 영역에서 col_diff 후 원본 좌표로 복원)
                compare_width = 1920 - right_x_start
                cw_small = compare_width // DETECT_SCALE
                rs = right_x_start // DETECT_SCALE
                left_region = safe_small[:, 0:cw_small]
                right_region = safe_small[:, rs : rs + cw_small]

                diff = np.abs(left_region.astype(np.int16) - right_region.astype(np.int16))
                col_diff = np.mean(diff, axis=0)

                # 좌측 분할선 탐색 구간 640±20 → 축소 인덱스
                search_start = 620 // DETECT_SCALE
                search_end = min(660, compare_width) // DETECT_SCALE

                if search_end > search_start:
                    mismatch_indices = np.where(col_diff[search_start:search_end] > 15.0)[0]
                    if len(mismatch_indices) > 0:
                        left_x_end = (search_start + mismatch_indices[0]) * DETECT_SCALE
                    else:
                        left_x_end = search_end * DETECT_SCALE
                else:
                    left_x_end = compare_width
                
                # 3. 시간 포맷 변환 (HH:MM:SS.mmm)
                timestamp_sec = frame_idx / self.fps
                time_str = format_time(timestamp_sec)
                
                # ==========================================================
                # [디버깅 기능 추가] 발견 당시 프레임 시각화 및 오차 그래프 저장
                # ==========================================================
                debug_img = frame.copy()
                
                # 1) Y축 탐색영역 (초록색 상자)
                cv2.rectangle(debug_img, (0, self.y_start), (1920, self.y_end), (0, 255, 0), 2)
                
                # 2) 템플릿 매칭 찾은 영역 (파란색 상자, 원본 좌표)
                tw_orig = 600
                rx = 1260 + max_loc[0] * DETECT_SCALE
                cv2.rectangle(debug_img, (rx, self.y_start),
                              (rx + tw_orig, self.y_end), (0, 0, 255), 3)
                              
                # 3) 계산된 분할선 표시 (빨간색 두꺼운 선)
                cv2.line(debug_img, (left_x_end, 0), (left_x_end, 1080), (255, 0, 0), 4)
                cv2.line(debug_img, (right_x_start, 0), (right_x_start, 1080), (255, 0, 0), 4)
                
                # 4) 정보 텍스트
                cv2.putText(debug_img, f"Time: {time_str} | LeftX: {left_x_end} | RightX: {right_x_start}", 
                            (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
                
                time_safe = time_str.replace(":", "-").replace(".", "-")
                
                import os
                save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "picture")
                os.makedirs(save_dir, exist_ok=True)
                
                frame_path = os.path.join(save_dir, f"debug_frame_{time_safe}.jpg")
                graph_path = os.path.join(save_dir, f"debug_graph_{time_safe}.png")
                
                cv2.imwrite(frame_path, cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
                
                # 5) 오차(diff) 배열 값 그래프 렌더링 (축소 좌표이므로 x는 컬럼 인덱스)
                try:
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(10, 4))
                    plt.plot(col_diff, label="Column Diff")
                    plt.axhline(y=15.0, color='r', linestyle='--', label="Threshold (15.0)")
                    plt.axvline(x=left_x_end // DETECT_SCALE, color='g', linestyle='-', label=f"Detected Left: {left_x_end}")
                    plt.axvspan(620 // DETECT_SCALE, 660 // DETECT_SCALE, color='yellow', alpha=0.3, label='Search Zone (640+-20)')
                    plt.title(f"Column Difference Graph at {time_str}")
                    plt.xlabel("X Coordinate (relative to Left 0)")
                    plt.ylabel("Mean Pixel Abs Diff")
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(graph_path)
                    plt.close()
                except ImportError:
                    print("[디버그] matplotlib이 없어 오차 그래프 저장은 생략합니다. (필요시 pip install matplotlib)")

                # 요구하신 정확한 로그 출력
                print(f"{time_str} 에서 첫 삼분할 구간을 찾았습니다.")
                print(f"삼분할 영역 x 좌표는 {left_x_end}, {right_x_start} 입니다.")
                print(f"[디버그] 디버그 파일이 다음 경로에 저장되었습니다: {save_dir}")
                # ==========================================================
                
                # ----------------------------------------------------------
                # 정확한 시작 지점을 찾기 위한 이진 탐색 (Binary Search)
                # ----------------------------------------------------------
                search_left_idx = max(0, frame_idx - jump_frames)
                search_right_idx = frame_idx
                
                print(f"[탐색] {time_str} 부근에서 삼분할 감지. 대략적인 시작 지점 역추적 (1초 정밀도)...")
                
                exact_start_idx = frame_idx
                while search_right_idx - search_left_idx > self.fps:
                    mid_idx = (search_left_idx + search_right_idx) // 2
                    mid_frame = self._get_frame_at_index(mid_idx)
                    mid_safe = mid_frame[self.y_start:self.y_end]
                    hm, wm = mid_safe.shape[:2]
                    mid_small = cv2.resize(
                        mid_safe, (wm // DETECT_SCALE, hm // DETECT_SCALE), interpolation=cv2.INTER_AREA
                    )
                    mid_template = mid_small[:, 0 : 600 // DETECT_SCALE]
                    mid_search = mid_small[:, 1260 // DETECT_SCALE : 1900 // DETECT_SCALE]
                    res_mid = cv2.matchTemplate(mid_search, mid_template, cv2.TM_CCOEFF_NORMED)
                    _, mid_max_val, _, _ = cv2.minMaxLoc(res_mid)
                    
                    if mid_max_val >= 0.95:
                        exact_start_idx = mid_idx
                        search_right_idx = mid_idx
                    else:
                        search_left_idx = mid_idx
                        
                exact_time_sec = exact_start_idx / self.fps
                exact_time_str = format_time(exact_time_sec)
                
                elapsed_time = time.time() - start_time
                print(f"[완료] 대략적인 삼분할 시작 시간은 {exact_time_str} 근처입니다!")
                print(f"(총 탐색 소요 시간: {elapsed_time:.3f}초)")
                
                return exact_time_str, exact_start_idx, left_x_end, right_x_start
                
            frame_idx += jump_frames
            
        elapsed_time = time.time() - start_time
        print(f"영상 내에서 삼분할 구간을 찾지 못했습니다. (탐색 소요 시간: {elapsed_time:.3f}초)")
        return None


'''
# --- 이하 사용하지 않는 코드 임시 주석 처리 ---

    def calibrate(self, frame):
        pass

    def check_fast_match(self, frame):
        pass

    def scan_video(self):
        pass
'''

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python tri_split_detector.py <비디오파일경로.ts> [--all-segments]")
        print("  --all-segments : 전체 스캔(대략)으로 모든 삼분할 구간 목록 출력")
        sys.exit(1)

    video_file = sys.argv[1]
    do_all_segments = len(sys.argv) >= 3 and sys.argv[2] == "--all-segments"

    print("ffprobe로 영상 정보 확인 중... (전체 인덱싱 없음, 즉시 시작)")
    init_start = time.time()
    detector = SplitVideoDetector(video_file)
    init_end = time.time()
    print(f"영상 정보 준비 완료 (소요 시간: {init_end - init_start:.3f}초)\n")

    if do_all_segments:
        detector.scan_all_segments_coarse()
    else:
        detector.find_first_tri_split()
