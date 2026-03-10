import sys
import time
import numpy as np
import cv2
from decord import VideoReader, cpu

def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

class SplitVideoDetector:
    def __init__(self, video_path, y_start=540, y_end=720):
        self.video_path = video_path
        self.vr = VideoReader(video_path, ctx=cpu(0)) 
        self.fps = self.vr.get_avg_fps()
        self.total_frames = len(self.vr)
        
        # Y축 안전 영역 (상/하단의 팝업, 자막 등을 피해 비교할 영역)
        # 팝업 위치에 따라 객체 생성 시 인자로 넘기거나 여기서 직접 수정하세요.
        self.y_start = y_start
        self.y_end = y_end

    def find_first_tri_split(self):
        """
        영상을 10초 단위로 듬성듬성 스캔하다가 첫 번째 삼분할 화면을 발견하면
        시간과 양쪽 분할선의 정확한 X 좌표를 출력합니다.
        """
        start_time = time.time()
        
        jump_frames = int(10 * self.fps)
        frame_idx = 0
        
        while frame_idx < self.total_frames:
            frame = self.vr[frame_idx].asnumpy()
            safe_frame = frame[self.y_start:self.y_end]
            
            # 삼분할이 맞는지 판단하기 위해, 왼쪽(0~600)을 템플릿으로 사용하고
            # 오른쪽 프레임 기준선(1280 +- 20 범위인 1260~1300) 부근에서 위치를 매칭
            template = safe_frame[:, 0:600]
            # 탐색 영역 시작은 1260, 폭은 템플릿의 이동폭 40 + 템플릿 폭 600 = 640. 끝점 1900.
            search_area = safe_frame[:, 1260:1900]
            
            res = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            
            # 일치율이 0.95 이상이면 삼분할 공간으로 간주
            if max_val >= 0.95:
                # 1. 우측 분할 영역(Right) 시작 좌표 찾기
                # search_area가 1260부터 시작, 템플릿은 0부터 시작하므로 그대로 더함
                right_x_start = 1260 + max_loc[0]
                
                # 2. 좌측 분할 영역(Left) 끝 좌표 찾기
                compare_width = 1920 - right_x_start
                left_region = safe_frame[:, 0:compare_width]
                right_region = safe_frame[:, right_x_start:1920]
                
                diff = np.abs(left_region.astype(np.int16) - right_region.astype(np.int16))
                col_diff = np.mean(diff, axis=0)
                
                # 무조건 좌측 분할선은 640 +- 20 (620 ~ 660 범위) 안에 존재한다는 가정 추가
                search_start = 620
                search_end = min(660, compare_width)
                
                if search_end > search_start:
                    mismatch_indices = np.where(col_diff[search_start:search_end] > 15.0)[0]
                    if len(mismatch_indices) > 0:
                        left_x_end = search_start + mismatch_indices[0]
                    else:
                        left_x_end = search_end
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
                
                # 2) 템플릿 매칭 찾은 영역 (파란색 상자)
                tw = template.shape[1]
                cv2.rectangle(debug_img, (1260 + max_loc[0], self.y_start), 
                              (1260 + max_loc[0] + tw, self.y_end), (0, 0, 255), 3)
                              
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
                
                # 5) 오차(diff) 배열 값 그래프 렌더링 (matplotlib 활용, 없으면 패스)
                try:
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(10, 4))
                    plt.plot(col_diff, label="Column Diff")
                    plt.axhline(y=15.0, color='r', linestyle='--', label="Threshold (15.0)")
                    plt.axvline(x=left_x_end, color='g', linestyle='-', label=f"Detected Left: {left_x_end}")
                    plt.axvspan(620, 660, color='yellow', alpha=0.3, label='Search Zone (640+-20)')
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
                
                elapsed_time = time.time() - start_time
                print(f"(탐색 소요 시간: {elapsed_time:.3f}초)")
                
                return time_str, left_x_end, right_x_start
                
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
        print("사용법: python tri_split_detector.py <비디오파일경로.ts>")
        sys.exit(1)
        
    video_file = sys.argv[1]
    
    print("비디오 파일을 불러오고 인덱싱하는 중... (길이가 긴 ts 영상은 여기서 시간이 많이 걸립니다)")
    init_start = time.time()
    detector = SplitVideoDetector(video_file)
    init_end = time.time()
    print(f"비디오 로딩 및 준비 완료 (소요 시간: {init_end - init_start:.3f}초)\n")
    
    detector.find_first_tri_split()
