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
    def __init__(self, video_path, output_json=None, min_duration=20, ssim_threshold=0.80, debug=False):
        self.video_path = video_path
        self.output_json = output_json or f"{os.path.splitext(video_path)[0]}_tri_splits.json"
        self.min_duration = min_duration
        self.ssim_threshold = ssim_threshold
        self.frame_rate = 1  # 1 FPS로 처리
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
        left = frame[:, :640]
        center = frame[:, 640:1280]
        right = frame[:, 1280:]
        
        # 디버그 이미지 생성
        debug_img = np.zeros((height, width * 2, 3), dtype=np.uint8)
        debug_img[:, :width] = frame
        
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
            # 프레임을 3등분
            left = frame[:, :640]
            center = frame[:, 640:1280]
            right = frame[:, 1280:]
            
            # SSIM 계산
            left_ssim = self.calculate_ssim(center, left)
            right_ssim = self.calculate_ssim(center, right)
            
            if self.debug:
                self.save_debug_frame(frame, left_ssim, right_ssim, frame_idx)
            
            return left_ssim >= self.ssim_threshold and right_ssim >= self.ssim_threshold
        except Exception as e:
            print(f"프레임 처리 중 오류 발생: {e}")
            return False

    def detect_tri_splits(self):
        start_total = time.time()
        video_info = self.get_video_info()
        frames_to_process = int(video_info['duration'] * self.frame_rate)

        tri_splits = []
        consecutive_count = 0
        start_time = None
        results = [None] * frames_to_process
        timing_results = {}

        # 1. 임시 폴더 생성
        temp_dir = "_temp_frames"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        # 2. ffmpeg로 프레임 이미지 일괄 추출
        ffmpeg_start = time.time()
        ffmpeg_cmd = [
            "ffmpeg",
            "-i", self.video_path,
            "-vf", f"fps={self.frame_rate}",
            os.path.join(temp_dir, "frame_%06d.png")
        ]
        print("ffmpeg로 프레임 이미지 추출 중...")
        subprocess.run(ffmpeg_cmd, check=True)
        timing_results['ffmpeg'] = time.time() - ffmpeg_start

        # 3. 이미지 파일 리스트
        frame_files = sorted(glob.glob(os.path.join(temp_dir, "frame_*.png")))

        def process_image(idx_file):
            idx, file = idx_file
            try:
                frame = cv2.imread(file)
                is_tri_split = self.process_frame(frame, idx)
                return (idx, is_tri_split)
            except Exception as e:
                print(f"프레임 {idx} 처리 중 오류 발생: {e}")
                return (idx, False)

        print("프레임 이미지 처리 중...")
        process_start = time.time()
        with tqdm(total=len(frame_files), desc="프레임 처리 중") as pbar:
            with ThreadPoolExecutor(max_workers=24) as executor:
                future_to_idx = {executor.submit(process_image, (idx, file)): idx for idx, file in enumerate(frame_files)}
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx, is_tri_split = future.result()
                    results[idx] = is_tri_split
                    pbar.update(1)
        timing_results['processing'] = time.time() - process_start

        # 4. 결과를 순차적으로 처리하여 구간 추출
        analysis_start = time.time()
        for frame_idx, is_tri_split in enumerate(results):
            current_time = frame_idx / self.frame_rate
            if is_tri_split:
                if consecutive_count == 0:
                    start_time = current_time
                consecutive_count += 1
            else:
                if consecutive_count >= self.min_consecutive_frames:
                    end_time = current_time
                    tri_splits.append({
                        "start_time": round(start_time, 2),
                        "end_time": round(end_time, 2),
                        "duration": round(end_time - start_time, 2)
                    })
                consecutive_count = 0
                start_time = None
        if consecutive_count >= self.min_consecutive_frames:
            end_time = frames_to_process / self.frame_rate
            tri_splits.append({
                "start_time": round(start_time, 2),
                "end_time": round(end_time, 2),
                "duration": round(end_time - start_time, 2)
            })
        timing_results['analysis'] = time.time() - analysis_start

        # 5. 임시 폴더 정리
        cleanup_start = time.time()
        shutil.rmtree(temp_dir)
        timing_results['cleanup'] = time.time() - cleanup_start

        timing_results['total'] = time.time() - start_total

        # 시간 측정 결과 출력
        print("\n=== 처리 시간 요약 ===")
        print(f"ffmpeg 프레임 추출: {timing_results['ffmpeg']:.2f}초")
        print(f"프레임 이미지 처리: {timing_results['processing']:.2f}초")
        print(f"구간 분석: {timing_results['analysis']:.2f}초")
        print(f"임시 파일 정리: {timing_results['cleanup']:.2f}초")
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
        print(f"구간 {i}: {split['start_time']}초 ~ {split['end_time']}초 (지속시간: {split['duration']}초)")
    
    if args.extract:
        print("\n구간 추출 중...")
        detector.extract_segments()
        print("구간 추출 완료!")

if __name__ == "__main__":
    main()
