import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
import os
import concurrent.futures
import glob
import shutil
import time

class TriSplitDetector:
    def __init__(self, video_path, output_json=None, min_duration=30, ssim_threshold=0.80, debug=False):
        self.video_path = video_path
        self.output_json = output_json or f"{os.path.splitext(video_path)[0]}_tri_splits.json"
        self.min_duration = min_duration
        self.ssim_threshold = ssim_threshold
        self.frame_rate = 1/10  # 10초당 1프레임 (0.1 FPS)
        self.min_consecutive_frames = int(self.min_duration * self.frame_rate)
        self.debug = debug
        if debug:
            self.debug_dir = "debug_frames"
            os.makedirs(self.debug_dir, exist_ok=True)
        
    def get_video_info(self):
        # ffprobe를 사용하여 비디오 정보 가져오기
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,duration",
            "-of", "json",
            self.video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        info = json.loads(result.stdout)
        stream = info['streams'][0]
        
        # FPS 계산
        fps_parts = stream['r_frame_rate'].split('/')
        fps = float(fps_parts[0]) / float(fps_parts[1])
        
        return {
            'width': int(stream['width']),
            'height': int(stream['height']),
            'fps': fps,
            'duration': float(stream['duration'])
        }

    def calculate_ssim(self, img1, img2):
        # 이미지를 그레이스케일로 변환
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # 이미지 크기 조정 (처리 속도 향상)
        scale = 0.5
        gray1 = cv2.resize(gray1, None, fx=scale, fy=scale)
        gray2 = cv2.resize(gray2, None, fx=scale, fy=scale)
        
        return ssim(gray1, gray2)

    def save_debug_frame(self, frame, left_ssim, right_ssim, frame_idx):
        if not self.debug:
            return
            
        # 프레임을 3등분
        height, width = frame.shape[:2]
        center_y = height // 2
        band_start = center_y - 15
        band_end = center_y + 15
        
        # 디버그 이미지 생성
        debug_img = np.zeros((height, width * 2, 3), dtype=np.uint8)
        debug_img[:, :width] = frame
        
        # 비교 밴드 표시
        cv2.rectangle(debug_img, (0, band_start), (width, band_end), (0, 255, 0), 2)
        
        # SSIM 값 표시
        cv2.putText(debug_img, f"Left SSIM: {left_ssim:.3f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(debug_img, f"Right SSIM: {right_ssim:.3f}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 구분선 그리기
        cv2.line(debug_img, (640, 0), (640, height), (0, 0, 255), 2)
        cv2.line(debug_img, (1280, 0), (1280, height), (0, 0, 255), 2)
        
        # 저장
        cv2.imwrite(os.path.join(self.debug_dir, f"frame_{frame_idx:06d}.jpg"), debug_img)

    def process_frame(self, frame, frame_idx):
        try:
            # 프레임이 3840x1080인 경우 좌측 1920x1080만 사용
            height, width = frame.shape[:2]
            if width == 3840:
                frame = frame[:, :1920]
                width = 1920

            # 세로 중앙 ±15픽셀 범위의 수평 밴드 추출 (y=525~555)
            center_y = height // 2
            band_start = center_y - 15
            band_end = center_y + 15
            horizontal_band = frame[band_start:band_end, :]
            
            # 밴드를 3등분
            section_width = width // 3
            left = horizontal_band[:, :section_width]
            center = horizontal_band[:, section_width:section_width*2]
            right = horizontal_band[:, section_width*2:]
            
            # SSIM 계산
            left_ssim = self.calculate_ssim(center, left)
            right_ssim = self.calculate_ssim(center, right)
            
            if self.debug:
                print(f"\n프레임 {frame_idx} 처리:")
                print(f"이미지 크기: {width}x{height}")
                print(f"밴드 위치: y={band_start}~{band_end}")
                print(f"Left SSIM: {left_ssim:.3f}")
                print(f"Right SSIM: {right_ssim:.3f}")
                print(f"임계값: {self.ssim_threshold}")
                self.save_debug_frame(frame, left_ssim, right_ssim, frame_idx)
            
            return left_ssim >= self.ssim_threshold and right_ssim >= self.ssim_threshold
        except Exception as e:
            print(f"프레임 처리 중 오류 발생: {e}")
            return False

    def process_frame_parallel(self, frame_idx, video_info):
        try:
            # ffmpeg를 사용하여 특정 프레임 추출
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(frame_idx / video_info['fps']),
                "-i", self.video_path,
                "-vframes", "1",
                "-f", "image2pipe",
                "-vcodec", "rawvideo",
                "-pix_fmt", "bgr24",
                "-"
            ]
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            frame_data, _ = process.communicate()
            
            if frame_data:
                frame = np.frombuffer(frame_data, dtype=np.uint8)
                frame = frame.reshape((video_info['height'], video_info['width'], 3))
                is_tri_split = self.process_frame(frame, frame_idx)
                return frame_idx, is_tri_split
            else:
                if self.debug:
                    print(f"프레임 {frame_idx} 데이터 추출 실패")
        except Exception as e:
            print(f"프레임 {frame_idx} 처리 중 오류: {e}")
        return frame_idx, False

    def format_time(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

    def find_precise_time(self, start_time, end_time, video_info, is_start=True):
        """정밀한 시작/종료 시간을 찾습니다."""
        search_range = 10  # 10초 범위 내에서 탐색
        
        if is_start:
            search_start = max(0, start_time - search_range)
            search_end = start_time
            step = 1.0  # 1초 단위로 탐색
        else:
            search_start = end_time
            search_end = min(video_info['duration'], end_time + search_range)
            step = 1.0  # 1초 단위로 탐색
        
        best_time = start_time if is_start else end_time
        best_ssim = 0
        found_tri_split = False
        consecutive_non_tri_split = 0  # 연속된 비-삼분할 프레임 카운트
        last_tri_split_time = None  # 마지막으로 발견된 삼분할 시간
        
        # 탐색 범위 내의 모든 프레임 검사
        for current_time in np.arange(search_start, search_end, step):
            frame_idx = int(current_time * video_info['fps'])
            
            # ffmpeg를 사용하여 특정 프레임 추출
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(current_time),
                "-i", self.video_path,
                "-vframes", "1",
                "-f", "image2pipe",
                "-vcodec", "rawvideo",
                "-pix_fmt", "bgr24",
                "-"
            ]
            
            try:
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                frame_data, _ = process.communicate()
                
                if frame_data:
                    frame = np.frombuffer(frame_data, dtype=np.uint8)
                    frame = frame.reshape((video_info['height'], video_info['width'], 3))
                    
                    # 프레임 처리
                    height, width = frame.shape[:2]
                    if width == 3840:
                        frame = frame[:, :1920]
                        width = 1920
                    
                    section_width = width // 3
                    left = frame[:, :section_width]
                    center = frame[:, section_width:section_width*2]
                    right = frame[:, section_width*2:]
                    
                    # SSIM 계산
                    left_ssim = self.calculate_ssim(center, left)
                    right_ssim = self.calculate_ssim(center, right)
                    current_ssim = min(left_ssim, right_ssim)
                    
                    # 시작 시간 찾기: SSIM이 임계값을 넘는 첫 지점
                    if is_start:
                        if current_ssim >= self.ssim_threshold:
                            best_time = current_time
                            found_tri_split = True
                            break
                    # 종료 시간 찾기: 연속된 3개의 비-삼분할 프레임이 나오면 종료로 판단
                    else:
                        if current_ssim >= self.ssim_threshold:
                            consecutive_non_tri_split = 0
                            last_tri_split_time = current_time
                            best_ssim = max(best_ssim, current_ssim)
                        else:
                            consecutive_non_tri_split += 1
                            if consecutive_non_tri_split >= 3:  # 3초 연속으로 삼분할이 아니면 종료로 판단
                                if last_tri_split_time is not None:
                                    best_time = last_tri_split_time
                                else:
                                    best_time = current_time - 2  # 2초 전을 종료 시점으로 설정
                                found_tri_split = True
                                break
            
            except Exception as e:
                print(f"정밀 시간 탐색 중 오류: {e}")
                continue
        
        # 삼분할을 찾지 못한 경우 원래 시간 사용
        if not found_tri_split:
            return start_time if is_start else end_time
            
        return best_time

    def detect_tri_splits(self):
        start_total = time.time()
        timing_results = {}

        # 1. 비디오 정보 가져오기
        video_info = self.get_video_info()
        if self.debug:
            print("\n=== 비디오 정보 ===")
            print(f"해상도: {video_info['width']}x{video_info['height']}")
            print(f"FPS: {video_info['fps']}")
            print(f"길이: {video_info['duration']}초")
            print("==================\n")
        
        fps = video_info['fps']
        total_frames = int(video_info['duration'] * fps)
        
        # 10초당 한 프레임만 처리하도록 설정
        frame_interval = int(fps * 10)
        target_frames = total_frames // frame_interval
        
        if self.debug:
            print(f"처리할 총 프레임 수: {target_frames}")
            print(f"프레임 간격: {frame_interval} (10초)")
        
        tri_splits = []
        consecutive_count = 0
        start_time = None
        
        print("프레임 처리 중...")
        process_start = time.time()
        
        # 병렬 처리를 위한 프레임 인덱스 리스트 생성
        frame_indices = list(range(0, total_frames, frame_interval))
        
        with tqdm(total=target_frames, desc="프레임 처리 중") as pbar:
            # ThreadPoolExecutor를 사용하여 병렬 처리
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                # 각 프레임에 대한 작업 제출
                future_to_frame = {
                    executor.submit(self.process_frame_parallel, frame_idx, video_info): frame_idx 
                    for frame_idx in frame_indices
                }
                
                # 결과 수집
                for future in concurrent.futures.as_completed(future_to_frame):
                    frame_idx, is_tri_split = future.result()
                    current_time = frame_idx / fps
                    
                    if is_tri_split:
                        if consecutive_count == 0:
                            start_time = current_time
                            if self.debug:
                                print(f"\n삼분할 시작 감지: {self.format_time(current_time)}")
                        consecutive_count += 1
                    else:
                        if consecutive_count >= self.min_consecutive_frames:
                            end_time = current_time
                            if self.debug:
                                print(f"삼분할 종료 감지: {self.format_time(current_time)}")
                                print(f"연속 프레임 수: {consecutive_count}")
                            
                            # 정밀한 시작/종료 시간 찾기
                            print(f"\n구간 {len(tri_splits)+1} 정밀 시간 탐색 중...")
                            precise_start = self.find_precise_time(start_time, end_time, video_info, is_start=True)
                            precise_end = self.find_precise_time(start_time, end_time, video_info, is_start=False)
                            
                            # 시작 시간이 종료 시간보다 늦은 경우 처리
                            if precise_start > precise_end:
                                precise_end = precise_start + (end_time - start_time)
                            
                            tri_splits.append({
                                "start_time": self.format_time(precise_start),
                                "end_time": self.format_time(precise_end),
                                "duration": round(precise_end - precise_start, 2)
                            })
                        consecutive_count = 0
                        start_time = None
                    
                    pbar.update(1)
        
        timing_results['processing'] = time.time() - process_start

        # 마지막 구간 처리
        if consecutive_count >= self.min_consecutive_frames:
            end_time = frame_idx / fps
            
            # 정밀한 시작/종료 시간 찾기
            print(f"\n구간 {len(tri_splits)+1} 정밀 시간 탐색 중...")
            precise_start = self.find_precise_time(start_time, end_time, video_info, is_start=True)
            precise_end = self.find_precise_time(start_time, end_time, video_info, is_start=False)
            
            # 시작 시간이 종료 시간보다 늦은 경우 처리
            if precise_start > precise_end:
                precise_end = precise_start + (end_time - start_time)
            
            tri_splits.append({
                "start_time": self.format_time(precise_start),
                "end_time": self.format_time(precise_end),
                "duration": round(precise_end - precise_start, 2)
            })

        timing_results['total'] = time.time() - start_total

        # 시간 측정 결과 출력
        print("\n=== 처리 시간 요약 ===")
        print(f"프레임 처리: {timing_results['processing']:.2f}초")
        print(f"전체 처리 시간: {timing_results['total']:.2f}초")
        print("====================\n")

        return tri_splits

    def save_results(self, tri_splits):
        with open(self.output_json, 'w', encoding='utf-8') as f:
            json.dump({
                "video_path": self.video_path,
                "tri_splits": tri_splits
            }, f, ensure_ascii=False, indent=2)

    def extract_segments(self, output_dir="tri_split_segments"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(self.output_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        extract_start = time.time()
        segment_times = []
        
        for i, segment in enumerate(data["tri_splits"]):
            segment_start = time.time()
            output_path = os.path.join(output_dir, f"segment_{i+1}.mp4")
            cmd = [
                "ffmpeg", "-y",
                "-i", self.video_path,
                "-ss", str(segment["start_time"]),
                "-to", str(segment["end_time"]),
                "-c:v", "libx264",
                "-c:a", "aac",
                output_path
            ]
            subprocess.run(cmd)
            segment_times.append(time.time() - segment_start)
        
        total_extract_time = time.time() - extract_start
        
        # 구간 추출 시간 요약 출력
        print("\n=== 구간 추출 시간 요약 ===")
        for i, t in enumerate(segment_times, 1):
            print(f"구간 {i} 추출: {t:.2f}초")
        print(f"전체 구간 추출: {total_extract_time:.2f}초")
        print("========================\n")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="삼분할 영상 감지 프로그램")
    parser.add_argument("video_path", help="입력 비디오 파일 경로")
    parser.add_argument("--output-json", help="출력 JSON 파일 경로")
    parser.add_argument("--min-duration", type=float, default=20, help="최소 지속 시간(초)")
    parser.add_argument("--ssim-threshold", type=float, default=0.80, help="SSIM 임계값")
    parser.add_argument("--extract", action="store_true", help="감지된 구간 추출 여부")
    parser.add_argument("--debug", action="store_true", help="디버그 모드 활성화")
    
    args = parser.parse_args()
    
    detector = TriSplitDetector(
        args.video_path,
        args.output_json,
        args.min_duration,
        args.ssim_threshold,
        args.debug
    )
    
    print("삼분할 영상 감지 중...")
    tri_splits = detector.detect_tri_splits()
    detector.save_results(tri_splits)
    
    print(f"\n감지된 삼분할 구간: {len(tri_splits)}개")
    for i, split in enumerate(tri_splits, 1):
        print(f"구간 {i}: {split['start_time']} - {split['end_time']} (지속시간: {split['duration']}초)")
    
    if args.extract:
        print("\n구간 추출 중...")
        detector.extract_segments()
        print("구간 추출 완료!")

if __name__ == "__main__":
    main()
