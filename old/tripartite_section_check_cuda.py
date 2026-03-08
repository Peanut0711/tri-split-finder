"""
삼분할(Tripartite) 구간 검출기.

FHD 영상에서 "화면 정중앙 1/3을 양옆으로 복제한" 구간을 찾아
시작·종료 타임스탬프를 로그로 출력합니다.
상단 팝업(별풍선 등)을 피하기 위해 높이(y) 중앙부만 샘플하여 비교합니다.

사용: python tripartite_section_check.py input.ts [--workers N] [--cuda]
필수: ffmpeg, ffprobe (PATH)
패키지: pip install -r requirements_tripartite.txt  (opencv-python, numpy; GPU 시 cupy-cuda12x 등)
자원: 기본 CPU. --cuda 시 ffmpeg GPU 디코딩 + (CuPy 있으면) 판정 연산 GPU.
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

# GPU 판정 연산용 CuPy (선택). 없으면 --cuda 시에도 판정은 CPU.
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


# --- 설정 (필요 시 조정) ---
# 탐색: 코스 스캔 → 경계 구간만 이진 탐색 (3~6시간 영상에 효율적)
COARSE_INTERVAL = 30.0  # 코스 스캔 간격(초). 30초면 6시간에 약 720회 샘플
BOUNDARY_TOLERANCE = 0.5  # 경계 이진 탐색 정밀도(초). 지시사항: 0.5초 이내
Y_CROP_RATIO_START = 0.35  # 높이 상 35% 지점부터 샘플 (상단 팝업 제외)
Y_CROP_RATIO_END = 0.65  # 높이 상 65% 지점까지 샘플 (중앙부만 사용)
MSE_THRESHOLD = 500.0  # 이 값 이하면 좌=우 동일으로 간주 (튜닝 가능)
MIN_SEGMENT_DURATION = 20.0  # 최소 구간 길이(초), 이보다 짧으면 구간으로 인정 안 함(선택)
MERGE_GAP_SECONDS = 10.0  # 인접 구간이 이 시간(초) 이내면 하나로 병합
# 중앙에만 효과(꽃잎 등)가 있는 삼분할: 좌·우 동일성만 비교. strict면 좌=중앙=우 모두 비교.
LEFT_RIGHT_ONLY_DEFAULT = True
# 코스 스캔 병렬화: 0이면 순차, 1 이상이면 해당 수만큼 프로세스 사용 (논리 코어까지 활용 가능)
DEFAULT_WORKERS = os.cpu_count() or 8


def get_video_info(path: Path) -> tuple[float, int, int]:
    """ffprobe로 영상 길이(초)와 너비, 높이를 반환."""
    duration_cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ]
    size_cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path),
    ]
    try:
        out_d = subprocess.run(duration_cmd, capture_output=True, text=True, check=True, timeout=10)
        out_s = subprocess.run(size_cmd, capture_output=True, text=True, check=True, timeout=10)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise RuntimeError(f"ffprobe 실행 실패 (ffprobe가 PATH에 있는지 확인): {e}") from e

    duration = float(out_d.stdout.strip())
    # ffprobe 출력에 줄바꿈/여러 스트림이 섞일 수 있음 → 쉼표·줄바꿈 모두 구분해 첫 두 숫자만 사용
    raw = out_s.stdout.strip().replace("\n", ",")
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) < 2:
        raise RuntimeError(f"ffprobe 해상도 파싱 실패: {out_s.stdout!r}")
    width, height = int(parts[0]), int(parts[1])
    return duration, width, height


def extract_frame(
    path: Path, time_sec: float, width: int, height: int, use_gpu: bool = False
) -> np.ndarray | None:
    """지정 시점의 1프레임을 BGR numpy 배열로 반환. use_gpu=True 시 ffmpeg CUDA 디코딩. 실패 시 None."""
    cmd = ["ffmpeg", "-y"]
    if use_gpu:
        cmd.extend(["-hwaccel", "cuda"])
    cmd.extend([
        "-ss", str(time_sec), "-i", str(path),
        "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-an", "-sn", "pipe:1",
    ])
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=15)
        if out.returncode != 0 or not out.stdout:
            if use_gpu:
                # CUDA 실패 시 CPU로 재시도 (드라이버/코덱 미지원 등)
                cmd_cpu = [
                    "ffmpeg", "-y", "-ss", str(time_sec), "-i", str(path),
                    "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "bgr24",
                    "-an", "-sn", "pipe:1",
                ]
                out = subprocess.run(cmd_cpu, capture_output=True, timeout=15)
            if out.returncode != 0 or not out.stdout:
                return None
        frame = np.frombuffer(out.stdout, dtype=np.uint8).reshape((height, width, 3))
        return frame
    except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
        return None


def crop_middle_y(frame: np.ndarray) -> np.ndarray:
    """높이(y) 중앙부만 잘라 반환. 상단 팝업 제외."""
    h, w = frame.shape[:2]
    y0 = int(h * Y_CROP_RATIO_START)
    y1 = int(h * Y_CROP_RATIO_END)
    return frame[y0:y1, :].copy()


def split_third(region: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """세로로 3등분해 왼쪽, 중앙, 오른쪽 영역 반환."""
    w = region.shape[1]
    third = w // 3
    left = region[:, :third]
    center = region[:, third : 2 * third]
    right = region[:, 2 * third :]
    return left, center, right


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean Squared Error. 두 영역 크기가 같아야 함."""
    if a.shape != b.shape:
        return float("inf")
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def _is_tripartite_frame_gpu(frame: np.ndarray, left_right_only: bool) -> bool:
    """CuPy로 crop/split/MSE를 GPU에서 수행 (use_gpu 시 호출)."""
    if not CUPY_AVAILABLE:
        return _is_tripartite_frame_cpu(frame, left_right_only)
    h, w = frame.shape[:2]
    y0 = int(h * Y_CROP_RATIO_START)
    y1 = int(h * Y_CROP_RATIO_END)
    g = cp.asarray(frame[y0:y1, :])
    third = g.shape[1] // 3
    left = g[:, :third]
    center = g[:, third : 2 * third]
    right = g[:, 2 * third :]
    min_w = min(left.shape[1], center.shape[1], right.shape[1])
    left = left[:, :min_w].astype(cp.float64)
    right = right[:, :min_w].astype(cp.float64)
    if left_right_only:
        mse_val = float(cp.mean((left - right) ** 2).item())
        return mse_val <= MSE_THRESHOLD
    center = center[:, :min_w].astype(cp.float64)
    mse_lc = float(cp.mean((left - center) ** 2).item())
    mse_rc = float(cp.mean((right - center) ** 2).item())
    return mse_lc <= MSE_THRESHOLD and mse_rc <= MSE_THRESHOLD


def _is_tripartite_frame_cpu(frame: np.ndarray, left_right_only: bool) -> bool:
    """CPU(NumPy) 경로."""
    cropped = crop_middle_y(frame)
    left, center, right = split_third(cropped)
    min_w = min(left.shape[1], center.shape[1], right.shape[1])
    left = left[:, :min_w]
    center = center[:, :min_w]
    right = right[:, :min_w]
    if left_right_only:
        return mse(left, right) <= MSE_THRESHOLD
    return mse(left, center) <= MSE_THRESHOLD and mse(right, center) <= MSE_THRESHOLD


def is_tripartite_frame(
    frame: np.ndarray, left_right_only: bool = True, use_gpu: bool = False
) -> bool:
    """
    프레임이 삼분할인지 판정. 높이 중앙부만 사용.
    left_right_only=True: 중앙 무시, 좌·우 동일성만 비교.
    use_gpu=True이고 CuPy 있으면 GPU에서 crop/MSE 수행.
    """
    if use_gpu and CUPY_AVAILABLE:
        return _is_tripartite_frame_gpu(frame, left_right_only)
    return _is_tripartite_frame_cpu(frame, left_right_only)


def format_ts(sec: float) -> str:
    """초 단위 시간을 HH:MM:SS.mmm 형식으로."""
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _print_progress(
    current_t: float, duration: float, scan_start: float, done: int, total: int
) -> None:
    elapsed = time.perf_counter() - scan_start
    pct = (current_t / duration * 100) if duration > 0 else 0
    if elapsed >= 60:
        em, es = int(elapsed // 60), int(elapsed % 60)
        elapsed_str = f"{em}분 {es}초"
    else:
        elapsed_str = f"{int(elapsed)}초"
    print(
        f"  진행: {format_ts(current_t)} / {format_ts(duration)}  ({done}/{total}, {pct:.1f}%)  |  실행 경과: {elapsed_str}",
        flush=True,
    )


def _coarse_worker(args: tuple) -> tuple[float, bool]:
    """멀티프로세싱용: (path_str, t, width, height, left_right_only, use_gpu) → (t, is_tripartite)."""
    path_str, t, width, height, left_right_only, use_gpu = args
    path = Path(path_str)
    frame = extract_frame(path, t, width, height, use_gpu)
    if frame is None:
        return (t, False)
    return (t, is_tripartite_frame(frame, left_right_only, use_gpu))


def _check_tripartite_at(
    path: Path,
    t: float,
    width: int,
    height: int,
    left_right_only: bool = True,
    use_gpu: bool = False,
) -> bool:
    """시점 t에서 1프레임 추출 후 삼분할 여부 반환."""
    frame = extract_frame(path, t, width, height, use_gpu)
    return frame is not None and is_tripartite_frame(frame, left_right_only, use_gpu)


def _binary_search_first_tripartite(
    path: Path,
    t_low: float,
    t_high: float,
    width: int,
    height: int,
    left_right_only: bool = True,
    use_gpu: bool = False,
) -> float:
    """[t_low, t_high] 구간에서 삼분할이 처음 나오는 시점을 이진 탐색. (t_high에서 True인 전제)"""
    if _check_tripartite_at(path, t_low, width, height, left_right_only, use_gpu):
        return t_low
    while t_high - t_low > BOUNDARY_TOLERANCE:
        mid = (t_low + t_high) * 0.5
        if _check_tripartite_at(path, mid, width, height, left_right_only, use_gpu):
            t_high = mid
        else:
            t_low = mid
    return t_high


def _binary_search_last_tripartite(
    path: Path,
    t_low: float,
    t_high: float,
    width: int,
    height: int,
    left_right_only: bool = True,
    use_gpu: bool = False,
) -> float:
    """[t_low, t_high] 구간에서 삼분할이 마지막으로 나오는 시점을 이진 탐색. (t_low에서 True인 전제)"""
    if _check_tripartite_at(path, t_high, width, height, left_right_only, use_gpu):
        return t_high
    while t_high - t_low > BOUNDARY_TOLERANCE:
        mid = (t_low + t_high) * 0.5
        if _check_tripartite_at(path, mid, width, height, left_right_only, use_gpu):
            t_low = mid
        else:
            t_high = mid
    return t_low


def find_segments(
    path: Path,
    duration: float,
    width: int,
    height: int,
    min_duration_sec: float = MIN_SEGMENT_DURATION,
    coarse_interval_sec: float = COARSE_INTERVAL,
    left_right_only: bool = LEFT_RIGHT_ONLY_DEFAULT,
    n_workers: int = 0,
    use_gpu: bool = False,
) -> list[tuple[float, float]]:
    """
    삼분할 구간 탐색 (장편 영상 최적화).
    left_right_only=True: 좌·우만 비교 (중앙에만 효과가 있을 때). False: 좌=중앙=우 모두 비교.
    n_workers: 코스 스캔 시 병렬 프로세스 수. 0이면 순차, 1 이상이면 CPU 멀티코어 활용.
    use_gpu: True 시 ffmpeg CUDA 디코딩 + (CuPy 있으면) 판정 연산 GPU.
    """
    path_str = str(path.resolve())
    ts: list[float] = []
    t = 0.0
    while t < duration:
        ts.append(t)
        t += coarse_interval_sec
    if duration > 0 and (not ts or ts[-1] < duration - 0.5):
        ts.append(max(0.0, duration - 0.1))
    total_points = len(ts)
    worker_args = [(path_str, t, width, height, left_right_only, use_gpu) for t in ts]

    coarse_results: list[tuple[float, bool]] = []
    scan_start = time.perf_counter()

    if n_workers <= 1:
        # 순차 처리
        for i, (_path_s, t, w, h, lr, ug) in enumerate(worker_args):
            frame = extract_frame(path, t, w, h, ug)
            if frame is None:
                coarse_results.append((t, False))
            else:
                coarse_results.append((t, is_tripartite_frame(frame, lr, ug)))
            if (i + 1) % 30 == 0:
                _print_progress(ts[i], duration, scan_start, i + 1, total_points)
    else:
        # 병렬 처리 (멀티코어 CPU 활용)
        done = 0
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for res in executor.map(_coarse_worker, worker_args, chunksize=max(1, total_points // (n_workers * 4))):
                coarse_results.append(res)
                done += 1
                if done % 30 == 0:
                    _print_progress(ts[done - 1] if done <= len(ts) else duration, duration, scan_start, done, total_points)
        coarse_results.sort(key=lambda x: x[0])

    scan_elapsed = time.perf_counter() - scan_start
    if scan_elapsed >= 60:
        em, es = int(scan_elapsed // 60), int(scan_elapsed % 60)
        scan_str = f"{em}분 {es}초"
    else:
        scan_str = f"{scan_elapsed:.1f}초"
    print(f"  코스 스캔 완료: {total_points}개 시점, 소요 {scan_str}", flush=True)

    # 2) 전환 구간에서 경계 정밀화
    segments: list[tuple[float, float]] = []
    i = 0
    while i < len(coarse_results):
        t_cur, is_tri = coarse_results[i]
        if not is_tri:
            i += 1
            continue
        # 연속된 True 구간의 시작/끝 후보
        t_start_cand = t_cur
        t_end_cand = t_cur
        j = i
        while j < len(coarse_results) and coarse_results[j][1]:
            t_end_cand = coarse_results[j][0]
            j += 1
        # 다음 코스 샘플 시점
        t_next = coarse_results[j][0] if j < len(coarse_results) else duration

        t_prev = coarse_results[i - 1][0] if i > 0 else 0.0
        t_start = _binary_search_first_tripartite(
            path, t_prev, t_start_cand, width, height, left_right_only, use_gpu
        )
        t_end = _binary_search_last_tripartite(
            path, t_end_cand, min(t_next, duration), width, height, left_right_only, use_gpu
        )

        if t_end <= duration and (t_end - t_start) >= 0:
            segments.append((t_start, t_end))
            print(f"  [발견] {format_ts(t_start)} ~ {format_ts(t_end)}", flush=True)
        i = j

    # 3) 인접 구간 병합
    merged: list[tuple[float, float]] = []
    for start, end in sorted(segments):
        if merged and (start - merged[-1][1]) <= MERGE_GAP_SECONDS:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    # 4) 최소 길이 미만 구간 제거 (min_duration_sec이 0이면 제거 안 함)
    return [(a, b) for a, b in merged if (b - a) >= min_duration_sec]


def main() -> None:
    parser = argparse.ArgumentParser(description="영상에서 삼분할 구간 검출 (y 중앙부 샘플)")
    parser.add_argument("input", type=Path, help="입력 영상 파일 (예: input.ts)")
    parser.add_argument("--no-min-duration", action="store_true", help="최소 구간 길이(20초) 검사 없이 모두 출력")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="좌=중앙=우 모두 비교. 기본은 중앙 무시, 좌·우 동일성만 비교 (중앙에만 효과 있을 때)",
    )
    parser.add_argument(
        "--coarse-interval",
        type=float,
        default=COARSE_INTERVAL,
        metavar="SEC",
        help=f"코스 스캔 간격(초). 기본 %(default)s. 30이면 더 촘촘, 120이면 더 성기게",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        metavar="N",
        help=f"코스 스캔 병렬 프로세스 수 (CPU 멀티코어). 기본 %(default)s. 1이면 순차 처리",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="GPU 사용: ffmpeg CUDA 디코딩, CuPy 있으면 판정 연산도 GPU (NVIDIA 드라이버 필요)",
    )
    args = parser.parse_args()

    path = args.input.resolve()
    if not path.is_file():
        print(f"[오류] 파일을 찾을 수 없습니다: {path}", file=sys.stderr)
        sys.exit(1)

    print(f"입력: {path}")
    try:
        duration, width, height = get_video_info(path)
    except RuntimeError as e:
        print(f"[오류] {e}", file=sys.stderr)
        sys.exit(1)

    min_dur = 0.0 if args.no_min_duration else MIN_SEGMENT_DURATION
    coarse = max(1.0, args.coarse_interval)

    left_right_only = not args.strict
    workers = max(0, args.workers)
    use_gpu = getattr(args, "cuda", False)
    print(f"길이: {format_ts(duration)} ({duration:.1f}초), 해상도: {width}x{height}")
    print(f"비교 영역: 높이 {Y_CROP_RATIO_START*100:.0f}% ~ {Y_CROP_RATIO_END*100:.0f}% (상단 팝업 제외)")
    print(f"모드: {'좌·우만 비교 (중앙 무시)' if left_right_only else '좌=중앙=우 모두 비교 (strict)'}")
    gpu_desc = "GPU(CUDA 디코딩" + (", CuPy 판정" if CUPY_AVAILABLE else "") + ")" if use_gpu else "CPU"
    print(f"코스 스캔: 간격 {coarse}s, 병렬 프로세스 {workers}개, 자원 {gpu_desc}")
    print("-" * 50)

    run_start = time.perf_counter()
    segments = find_segments(
        path, duration, width, height,
        min_duration_sec=min_dur,
        coarse_interval_sec=coarse,
        left_right_only=left_right_only,
        n_workers=workers,
        use_gpu=use_gpu,
    )

    if not segments:
        print("삼분할 구간이 없습니다.")
        return

    run_elapsed = time.perf_counter() - run_start
    if run_elapsed >= 60:
        rm, rs = int(run_elapsed // 60), int(run_elapsed % 60)
        run_str = f"{rm}분 {rs}초"
    else:
        run_str = f"{run_elapsed:.1f}초"

    print("삼분할 구간:")
    for i, (start, end) in enumerate(segments, 1):
        print(f"  [{i}] {format_ts(start)} ~ {format_ts(end)}  (길이 {end - start:.1f}초)")
    print("-" * 50)
    print(f"총 {len(segments)}개 구간  (전체 소요: {run_str})")


if __name__ == "__main__":
    main()
