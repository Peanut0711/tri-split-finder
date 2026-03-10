"""
삼분할(Tripartite) 구간 검출기.

FHD 영상에서 "화면 정중앙 1/3을 양옆으로 복제한" 구간을 찾아
시작·종료 타임스탬프를 로그로 출력합니다.
상단 팝업을 피하기 위해 높이(y) 중앙부만 샘플하여 비교합니다.

사용: python tripartite_section_check.py input.ts [--workers N] [--cuda]
필수: ffmpeg, ffprobe (PATH)
패키지: pip install -r requirements_tripartite.txt  (opencv-python, numpy; GPU 시 cupy-cuda12x 등)
자원: 기본 CPU. --cuda 시 ffmpeg GPU 디코딩 + (CuPy 있으면) 판정 연산 GPU.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

# --verify-export 시 이미지 저장용. 없으면 해당 옵션 사용 시 안내만 출력.
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None  # type: ignore[assignment]
    CV2_AVAILABLE = False

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
BOUNDARY_TOLERANCE = 0.9375  # 경계 이진 탐색 정밀도(초). 기본 0.9375초(속도·정확도 균형). CLI --boundary-tolerance로 변경 가능
Y_CROP_RATIO_START = 0.35  # 높이 상 35% 지점부터 샘플 (상단 팝업 제외)
Y_CROP_RATIO_END = 0.65  # 높이 상 65% 지점까지 샘플 (중앙부만 사용)
MSE_THRESHOLD = 500.0  # 이 값 이하면 좌=우 동일으로 간주 (튜닝 가능)
# 송출자가 가상선을 640/1280 대신 647/1273 등으로 그린 경우 대비: 경계를 ±N픽셀 옮겨 보며 MSE 최소인 정렬 탐색
ALIGN_TOLERANCE_PX = 5  # 0이면 고정 1/3·2/3만 사용, 10이면 w/3±10 픽셀 범위에서 최적 좌/우 너비 탐색
# 빠른 경로(불연속성 검출 → 소패치 MSE): 경계 검색 ±픽셀, 검증용 패치 크기
BOUNDARY_SEARCH_TOLERANCE_PX = 5  # 637, 1273 등 정확히 1/3·2/3가 아닐 때도 검출 (최대 10픽셀 편차)
FAST_PATH_PATCH_SIZE = 40  # 40×40 패치로 좌/우 동일 위치 MSE 검증 (연산량 감소)
FAST_PATH_GRADIENT_PEAK_RATIO = 1.5  # 경계로 인정할 최소 peak/median 비율 (평탄한 영상 오탐 방지)
MIN_SEGMENT_DURATION = 20.0  # 최소 구간 길이(초), 이보다 짧으면 구간으로 인정 안 함(선택)
MERGE_GAP_SECONDS = 10.0  # 인접 구간이 이 시간(초) 이내면 하나로 병합
# 병합 시 임시 폴더: 사용자 Videos 아래 전용 폴더 사용(로컬/NVMe 체감용). 실행마다 그 안에 merge_xxx 생성 후 삭제.
MERGE_TMP_BASE = Path.home() / "Videos" / "tripartite_merge_tmp"
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


def _get_keyframe_times_from_file(path: Path, timeout: int = 120) -> list[float]:
    """
    단일 파일(보통 작은 조각)에서 키프레임 PTS(초) 목록 반환.
    packet 방식 → 0개면 frame 방식 폴백. 로컬 작은 파일용.
    """
    path_str = str(path)
    # 1) packet=pts_time,flags
    cmd_p = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "packet=pts_time,flags", "-of", "csv=p=0", path_str,
    ]
    try:
        out = subprocess.run(cmd_p, capture_output=True, text=True, check=True, timeout=timeout)
        times = []
        for line in out.stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 1)
            if len(parts) < 2 or "K" not in parts[1]:
                continue
            try:
                times.append(float(parts[0]))
            except ValueError:
                continue
        if times:
            return sorted(times)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    # 2) frame=key_frame,pkt_pts_time
    cmd_f = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "frame=key_frame,pkt_pts_time", "-of", "csv=p=0", path_str,
    ]
    try:
        out = subprocess.run(cmd_f, capture_output=True, text=True, check=True, timeout=timeout)
        times = []
        for line in out.stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 1)
            if len(parts) < 2:
                continue
            if parts[0] != "1":
                continue
            try:
                times.append(float(parts[1]))
            except ValueError:
                continue
        return sorted(times)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return []


def extract_frame(
    path: Path,
    time_sec: float,
    width: int,
    height: int,
    use_gpu: bool = False,
    scale_width: int | None = None,
) -> np.ndarray | None:
    """지정 시점의 1프레임을 BGR numpy 배열로 반환. use_gpu=True 시 ffmpeg CUDA 디코딩. 실패 시 None.
    scale_width가 지정되면 해당 폭으로 리사이즈 후 반환(코스 스캔 저해상도용). 비율 유지해 높이 계산."""
    out_w = scale_width if scale_width is not None else width
    out_h = round(height * out_w / width) if scale_width is not None else height
    base = ["-ss", str(time_sec), "-i", str(path)]
    scale_filters = ["-vf", f"scale={out_w}:-1"] if scale_width is not None else []
    tail = ["-vframes", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-an", "-sn", "pipe:1"]

    cmd = ["ffmpeg", "-y"]
    if use_gpu:
        cmd.extend(["-hwaccel", "cuda"])
    cmd.extend(base + scale_filters + tail)
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=15)
        if out.returncode != 0 or not out.stdout:
            if use_gpu:
                cmd_cpu = ["ffmpeg", "-y"] + base + scale_filters + tail
                out = subprocess.run(cmd_cpu, capture_output=True, timeout=15)
            if out.returncode != 0 or not out.stdout:
                return None
        frame = np.frombuffer(out.stdout, dtype=np.uint8).reshape((out_h, out_w, 3))
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


def split_third_with_left_width(
    region: np.ndarray, left_width: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """좌/우 너비를 left_width로 해서 왼쪽·중앙·오른쪽 영역 반환. (정렬 허용 시 사용)"""
    w = region.shape[1]
    left = region[:, :left_width]
    center = region[:, left_width : w - left_width]
    right = region[:, w - left_width :]
    return left, center, right


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean Squared Error. 두 영역 크기가 같아야 함."""
    if a.shape != b.shape:
        return float("inf")
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def _horizontal_gradient_strength(cropped: np.ndarray) -> np.ndarray:
    """가로 방향 픽셀 차이(불연속성) 강도. 반환 shape (w-1,). 각 열 경계에서 세로·채널 평균 절대차."""
    if cropped.size == 0 or cropped.shape[1] < 2:
        return np.array([])
    diff = np.abs(
        cropped[:, 1:].astype(np.float64) - cropped[:, :-1].astype(np.float64)
    )
    # 채널이 있으면(h,w,3) axis=(0,2)로, 그레이면(h,w) axis=0으로 → 항상 (w-1,)
    axes = (0, 2) if diff.ndim == 3 else 0
    return np.mean(diff, axis=axes)


def _detect_boundaries_near_third(
    cropped: np.ndarray, tolerance_px: int = BOUNDARY_SEARCH_TOLERANCE_PX
) -> tuple[int | None, int | None]:
    """
    w/3, 2w/3 근처(±tolerance_px)에서 수직 경계(그래디언트 피크) 검출.
    637, 1273 등 정확히 640/1280이 아닌 경우도 검출. 반환: (b1, b2) 열 인덱스 또는 (None, None).
    """
    strength = _horizontal_gradient_strength(cropped)
    if strength.size < 2:
        return (None, None)
    w = cropped.shape[1]
    # diff[i] = 열 i와 i+1 사이 경계 → 픽셀 인덱스로는 경계 위치 i+1
    base1 = w // 3
    base2 = (2 * w) // 3
    lo1 = max(0, base1 - tolerance_px)
    hi1 = min(strength.size, base1 + tolerance_px)
    lo2 = max(0, base2 - tolerance_px)
    hi2 = min(strength.size, base2 + tolerance_px)
    if lo1 >= hi1 or lo2 >= hi2:
        return (None, None)
    median_all = float(np.median(strength))
    min_peak = median_all * FAST_PATH_GRADIENT_PEAK_RATIO if median_all > 0 else 0.0
    idx1 = lo1 + int(np.argmax(strength[lo1:hi1]))
    idx2 = lo2 + int(np.argmax(strength[lo2:hi2]))
    peak1 = float(strength[idx1])
    peak2 = float(strength[idx2])
    if peak1 >= min_peak and peak2 >= min_peak:
        return (idx1 + 1, idx2 + 1)  # 경계는 열 idx와 idx+1 사이 → 오른쪽 열 인덱스 반환
    return (None, None)


def _mse_small_patches(
    cropped: np.ndarray,
    patch_size: int = FAST_PATH_PATCH_SIZE,
    b1: int | None = None,
    b2: int | None = None,
) -> float:
    """
    좌 1/3·우 1/3의 동일 상대 위치에서 patch_size×patch_size 패치 두 개 추출 후 MSE.
    b1, b2가 주어지면 경계 검출 결과(BOUNDARY_SEARCH_TOLERANCE_PX 기반)로 패치 중심 계산.
    없으면 고정 w/6, 5w/6 사용. 삼분할이면 같은 위치가 복제되어 있으므로 MSE가 낮음.
    """
    h, w = cropped.shape[:2]
    half = patch_size // 2
    if b1 is not None and b2 is not None:
        # 경계 기준: 좌 [0, b1) 중앙, 우 [b2, w) 중앙
        left_center_x = b1 // 2
        right_center_x = b2 + (w - b2) // 2
    else:
        left_center_x = w // 6
        right_center_x = 5 * w // 6
    cy = h // 2
    x0_l = left_center_x - half
    x1_l = left_center_x + half
    x0_r = right_center_x - half
    x1_r = right_center_x + half
    if x0_l < 0 or x1_l > w or x0_r < 0 or x1_r > w:
        return float("inf")
    if cy - half < 0 or cy + half > h:
        return float("inf")
    # b1/b2 사용 시 패치가 각 1/3 구간 안에 들어가야 함
    if b1 is not None and (x0_l < 0 or x1_l > b1):
        return float("inf")
    if b2 is not None and (x0_r < b2 or x1_r > w):
        return float("inf")
    left_patch = cropped[cy - half : cy + half, x0_l:x1_l]
    right_patch = cropped[cy - half : cy + half, x0_r:x1_r]
    if left_patch.shape != right_patch.shape:
        return float("inf")
    return mse(left_patch, right_patch)


def _is_tripartite_fast_path(frame: np.ndarray) -> bool | None:
    """
    1단계: w/3, 2w/3 근처(±BOUNDARY_SEARCH_TOLERANCE_PX) 수직 불연속성 검출.
    2단계: 경계 두 개 모두 발견 시, 좌/우 동일 상대 위치 40×40 패치 MSE로 검증.
    True=삼분할, False=아님, None=불명확(기존 전체 MSE 경로로 폴백).
    """
    cropped = crop_middle_y(frame)
    h, w = cropped.shape[:2]
    if w < 6 * BOUNDARY_SEARCH_TOLERANCE_PX or h < FAST_PATH_PATCH_SIZE:
        return None
    b1, b2 = _detect_boundaries_near_third(cropped)
    if b1 is None or b2 is None:
        return None
    patch_mse = _mse_small_patches(cropped, b1=b1, b2=b2)
    return patch_mse <= MSE_THRESHOLD


def _is_tripartite_frame_gpu(
    frame: np.ndarray, left_right_only: bool, align_tolerance_px: int | None = None
) -> bool:
    """CuPy로 crop/split/MSE를 GPU에서 수행 (use_gpu 시 호출)."""
    if not CUPY_AVAILABLE:
        return _is_tripartite_frame_cpu(frame, left_right_only, align_tolerance_px)
    h, w = frame.shape[:2]
    y0 = int(h * Y_CROP_RATIO_START)
    y1 = int(h * Y_CROP_RATIO_END)
    g = cp.asarray(frame[y0:y1, :])
    crop_w = g.shape[1]
    base = crop_w // 3
    tol = align_tolerance_px if align_tolerance_px is not None else ALIGN_TOLERANCE_PX
    if tol <= 0:
        L_vals = [base]
    else:
        L_vals = range(
            max(1, base - tol),
            min(crop_w // 2, base + tol + 1),
        )
    for L in L_vals:
        if crop_w - L < 1:
            continue
        left = g[:, :L].astype(cp.float64)
        right = g[:, crop_w - L :].astype(cp.float64)
        if left_right_only:
            mse_val = float(cp.mean((left - right) ** 2).item())
            if mse_val <= MSE_THRESHOLD:
                return True
        else:
            center = g[:, L : crop_w - L].astype(cp.float64)
            min_c = min(left.shape[1], center.shape[1], right.shape[1])
            if min_c < 1:
                continue
            lc = left[:, :min_c]
            cc = center[:, :min_c]
            rc = right[:, :min_c]
            mse_lc = float(cp.mean((lc - cc) ** 2).item())
            mse_rc = float(cp.mean((rc - cc) ** 2).item())
            if mse_lc <= MSE_THRESHOLD and mse_rc <= MSE_THRESHOLD:
                return True
    return False


def _is_tripartite_frame_cpu(
    frame: np.ndarray, left_right_only: bool, align_tolerance_px: int | None = None
) -> bool:
    """CPU(NumPy) 경로. ALIGN_TOLERANCE_PX > 0이면 경계를 ±N픽셀 옮겨 보며 최적 정렬 탐색."""
    cropped = crop_middle_y(frame)
    w = cropped.shape[1]
    base = w // 3
    tol = align_tolerance_px if align_tolerance_px is not None else ALIGN_TOLERANCE_PX
    if tol <= 0:
        L_vals = [base]
    else:
        L_vals = range(max(1, base - tol), min(w // 2, base + tol + 1))
    for L in L_vals:
        if w - L < 1:
            continue
        left, center, right = split_third_with_left_width(cropped, L)
        min_w = min(left.shape[1], center.shape[1], right.shape[1])
        if min_w < 1:
            continue
        left = left[:, :min_w]
        center = center[:, :min_w]
        right = right[:, :min_w]
        if left_right_only:
            if mse(left, right) <= MSE_THRESHOLD:
                return True
        else:
            if mse(left, center) <= MSE_THRESHOLD and mse(right, center) <= MSE_THRESHOLD:
                return True
    return False


def is_tripartite_frame(
    frame: np.ndarray,
    left_right_only: bool = True,
    use_gpu: bool = False,
    align_tolerance_px: int | None = None,
) -> bool:
    """
    프레임이 삼분할인지 판정. 높이 중앙부만 사용.
    left_right_only=True일 때: 1) 불연속성 검출(w/3·2w/3 ±10px) → 40×40 패치 MSE 검증을 먼저 시도하고,
    결론이 나면 즉시 반환. 불명확하면 2) 기존 전체 영역 MSE(및 GPU) 경로로 폴백.
    """
    if left_right_only:
        fast = _is_tripartite_fast_path(frame)
        if fast is not None:
            return fast
    if use_gpu and CUPY_AVAILABLE:
        return _is_tripartite_frame_gpu(frame, left_right_only, align_tolerance_px)
    return _is_tripartite_frame_cpu(frame, left_right_only, align_tolerance_px)


def get_tripartite_mse(
    frame: np.ndarray,
    left_right_only: bool = True,
    use_gpu: bool = False,
    align_tolerance_px: int | None = None,
) -> tuple[bool, dict[str, float]]:
    """
    프레임의 삼분할 판정 + MSE 값 반환 (--verify용).
    ALIGN_TOLERANCE_PX > 0이면 경계를 ±N픽셀 옮겨 보며 최적 정렬에서의 MSE 반환.
    반환: (is_tripartite, {"left_right": mse} 또는 {"left_center", "right_center"}, 필요 시 "best_L" 포함).
    """
    cropped = crop_middle_y(frame)
    w = cropped.shape[1]
    base = w // 3
    tol = align_tolerance_px if align_tolerance_px is not None else ALIGN_TOLERANCE_PX
    L_vals = (
        [base]
        if tol <= 0
        else list(range(max(1, base - tol), min(w // 2, base + tol + 1)))
    )

    if use_gpu and CUPY_AVAILABLE:
        g = cp.asarray(cropped)
        best_mse_lr = float("inf")
        best_mse_lc = best_mse_rc = float("inf")
        best_L = base
        for L in L_vals:
            if w - L < 1:
                continue
            left = g[:, :L].astype(cp.float64)
            right = g[:, w - L :].astype(cp.float64)
            if left_right_only:
                mse_lr = float(cp.mean((left - right) ** 2).item())
                if mse_lr < best_mse_lr:
                    best_mse_lr = mse_lr
                    best_L = L
            else:
                center = g[:, L : w - L].astype(cp.float64)
                min_c = min(left.shape[1], center.shape[1], right.shape[1])
                if min_c < 1:
                    continue
                lc, cc, rc = left[:, :min_c], center[:, :min_c], right[:, :min_c]
                mse_lc = float(cp.mean((lc - cc) ** 2).item())
                mse_rc = float(cp.mean((rc - cc) ** 2).item())
                if max(mse_lc, mse_rc) < max(best_mse_lc, best_mse_rc):
                    best_mse_lc, best_mse_rc = mse_lc, mse_rc
                    best_L = L
        if left_right_only:
            ok = best_mse_lr <= MSE_THRESHOLD
            return (ok, {"left_right": best_mse_lr, "best_L": float(best_L)})
        ok = best_mse_lc <= MSE_THRESHOLD and best_mse_rc <= MSE_THRESHOLD
        return (
            ok,
            {"left_center": best_mse_lc, "right_center": best_mse_rc, "best_L": float(best_L)},
        )

    # CPU
    best_mse_lr = float("inf")
    best_mse_lc = best_mse_rc = float("inf")
    best_L = base
    for L in L_vals:
        if w - L < 1:
            continue
        left, center, right = split_third_with_left_width(cropped, L)
        min_w = min(left.shape[1], center.shape[1], right.shape[1])
        if min_w < 1:
            continue
        left = left[:, :min_w]
        center = center[:, :min_w]
        right = right[:, :min_w]
        if left_right_only:
            mse_lr = mse(left, right)
            if mse_lr < best_mse_lr:
                best_mse_lr = mse_lr
                best_L = L
        else:
            mse_lc = mse(left, center)
            mse_rc = mse(right, center)
            if max(mse_lc, mse_rc) < max(best_mse_lc, best_mse_rc):
                best_mse_lc, best_mse_rc = mse_lc, mse_rc
                best_L = L
    if left_right_only:
        return (best_mse_lr <= MSE_THRESHOLD, {"left_right": best_mse_lr, "best_L": float(best_L)})
    ok = best_mse_lc <= MSE_THRESHOLD and best_mse_rc <= MSE_THRESHOLD
    return (
        ok,
        {"left_center": best_mse_lc, "right_center": best_mse_rc, "best_L": float(best_L)},
    )


def format_ts(sec: float) -> str:
    """초 단위 시간을 HH:MM:SS.mmm 형식으로."""
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _ts_to_export_fname(sec: float) -> str:
    """타임스탬프를 export 파일명에 쓸 수 있는 문자열로 (콜론 제거)."""
    return format_ts(sec).replace(":", "-")


def get_tripartite_diagnostics(frame: np.ndarray) -> dict:
    """
    한 프레임에 대한 삼분할 판정 상세(디버깅용). 빠른 경로·전체 MSE·최종 판정과 사유 반환.
    반환: cropped, b1, b2, patch_mse, full_mse, fast_verdict(bool|None), final_verdict(bool), reason(str)
    """
    cropped = crop_middle_y(frame)
    h, w = cropped.shape[:2]
    y0 = int(frame.shape[0] * Y_CROP_RATIO_START)
    y1 = int(frame.shape[0] * Y_CROP_RATIO_END)
    b1, b2 = _detect_boundaries_near_third(cropped) if w >= 6 * BOUNDARY_SEARCH_TOLERANCE_PX and h >= FAST_PATH_PATCH_SIZE else (None, None)
    patch_mse = _mse_small_patches(cropped, b1=b1, b2=b2) if (b1 is not None and b2 is not None) else float("inf")
    # 패치 중심(경계 기반): 디버그 그리기용
    if b1 is not None and b2 is not None:
        left_center_x = b1 // 2
        right_center_x = b2 + (w - b2) // 2
    else:
        left_center_x = w // 6
        right_center_x = 5 * w // 6
    # 전체 영역 MSE (고정 1/3 분할)
    left, center, right = split_third(cropped)
    min_w = min(left.shape[1], right.shape[1])
    full_mse = mse(left[:, :min_w], right[:, :min_w]) if min_w >= 1 else float("inf")
    fast_verdict = None
    if b1 is not None and b2 is not None:
        fast_verdict = patch_mse <= MSE_THRESHOLD
    final_verdict = bool(is_tripartite_frame(frame, left_right_only=True, use_gpu=False))
    if fast_verdict is None:
        reason = f"boundaries not found (b1={b1}, b2={b2}) -> fallback full_mse={full_mse:.0f}"
    elif fast_verdict:
        reason = f"fast: b1={b1} b2={b2} patch_mse={patch_mse:.0f}<=500"
    else:
        reason = f"fast: b1={b1} b2={b2} patch_mse={patch_mse:.0f}>{MSE_THRESHOLD}"
    return {
        "cropped": cropped,
        "b1": b1, "b2": b2,
        "left_center_x": left_center_x,
        "right_center_x": right_center_x,
        "patch_mse": patch_mse,
        "full_mse": full_mse,
        "fast_verdict": fast_verdict,
        "final_verdict": final_verdict,
        "reason": reason,
        "y0": y0, "y1": y1,
        "w": w, "h_crop": h,
    }


def _draw_boundary_debug_frame(
    frame: np.ndarray, diag: dict, t_sec: float, label_prefix: str
) -> np.ndarray:
    """디버그용: 프레임에 크롭 영역·경계선·40x40 패치·판정 문구를 그려 반환. opencv 필요."""
    if not CV2_AVAILABLE:
        return frame
    img = frame.copy()
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h_full, w_full = img.shape[:2]
    y0, y1 = diag["y0"], diag["y1"]
    w = diag["w"]
    b1, b2 = diag.get("b1"), diag.get("b2")
    patch_mse = diag.get("patch_mse", 0)
    full_mse = diag.get("full_mse", 0)
    final_verdict = diag.get("final_verdict", False)
    reason = diag.get("reason", "")
    # 크롭 영역 네모 (녹색)
    cv2.rectangle(img, (0, y0), (w_full - 1, y1 - 1), (0, 255, 0), 2)
    cv2.putText(img, "crop (40-50%h)", (5, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    # 경계선: 검출됐으면 빨강, 없으면 회색 w/3, 2w/3
    if b1 is not None and b2 is not None:
        cv2.line(img, (b1, 0), (b1, h_full - 1), (0, 0, 255), 2)
        cv2.line(img, (b2, 0), (b2, h_full - 1), (0, 0, 255), 2)
        cv2.putText(img, f"b1={b1}", (b1 + 4, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(img, f"b2={b2}", (b2 + 4, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    else:
        t1, t2 = w_full // 3, 2 * w_full // 3
        cv2.line(img, (t1, 0), (t1, h_full - 1), (128, 128, 128), 1)
        cv2.line(img, (t2, 0), (t2, h_full - 1), (128, 128, 128), 1)
    # 40x40 패치 위치 (노랑): b1/b2 있으면 경계 기반 중심, 없으면 고정 w/6, 5w/6
    half = FAST_PATH_PATCH_SIZE // 2
    cy_crop = diag["h_crop"] // 2
    left_cx = diag.get("left_center_x", w // 6)
    right_cx = diag.get("right_center_x", 5 * w // 6)
    ly0, ly1 = y0 + cy_crop - half, y0 + cy_crop + half
    cv2.rectangle(img, (left_cx - half, ly0), (left_cx + half, ly1), (0, 255, 255), 2)
    cv2.rectangle(img, (right_cx - half, ly0), (right_cx + half, ly1), (0, 255, 255), 2)
    cv2.putText(img, "40x40 patch", (left_cx - half, ly0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    # 상단 텍스트: 시각, 판정, MSE, 사유 (에러/판정 관련은 빨간색 원색으로 가독성 확보)
    ts_str = format_ts(t_sec)
    verdict_str = "Tripartite O" if final_verdict else "Tripartite X"
    color_red = (0, 0, 255)   # BGR 빨강 (에러·불일치 시)
    color_ok = (0, 255, 0)    # BGR 녹색 (일치 시)
    color_white = (255, 255, 255)
    line1 = f"{label_prefix} t={ts_str}  {verdict_str}"
    cv2.putText(img, line1, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_red if not final_verdict else color_ok, 2)
    line2 = f"patch_mse={patch_mse:.0f}  full_mse={full_mse:.0f}  (thr=500)"
    cv2.putText(img, line2, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_white, 2)
    cv2.putText(img, reason[:90], (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_red if not final_verdict else color_white, 2)
    return img


def run_debug_boundary(
    path: Path,
    start: float,
    end: float,
    margin_sec: float,
    width: int,
    height: int,
    use_gpu: bool,
    left_right_only: bool,
    duration: float,
) -> Path | None:
    """
    구간 [start, end] 전후 margin_sec 초를 샘플링해 각 시점의 판정 근거를 스크린샷으로 저장.
    반환: 저장한 디렉터리 경로 또는 실패 시 None. opencv-python 필요.
    """
    if not CV2_AVAILABLE:
        print("[경고] --debug-boundary를 쓰려면 opencv-python이 필요합니다. pip install opencv-python", flush=True)
        return None
    out_dir = Path.home() / "Videos" / "tripartite_temp"
    out_dir.mkdir(parents=True, exist_ok=True)
    seg_name = f"{format_ts(start).replace(':', '-')}_{format_ts(end).replace(':', '-')}"
    seg_dir = out_dir / seg_name
    seg_dir.mkdir(parents=True, exist_ok=True)
    interval = 2.0
    t_start_lo = max(0.0, start - margin_sec)
    t_start_hi = min(duration, start + margin_sec)
    t_end_lo = max(0.0, end - margin_sec)
    t_end_hi = min(duration, end + margin_sec)
    times = []
    t = t_start_lo
    while t <= t_start_hi:
        times.append(("start", t))
        t += interval
    t = t_end_lo
    while t <= t_end_hi:
        times.append(("end", t))
        t += interval
    def _save_png(p: Path, img: np.ndarray) -> bool:
        ret, buf = cv2.imencode(".png", img)
        if not ret or buf is None:
            return False
        try:
            p.write_bytes(buf.tobytes())
            return True
        except OSError:
            return False
    saved = 0
    for label, t in times:
        frame = extract_frame(path, t, width, height, use_gpu)
        if frame is None:
            continue
        diag = get_tripartite_diagnostics(frame)
        drawn = _draw_boundary_debug_frame(frame, diag, t, label)
        fname = f"{label}_{_ts_to_export_fname(t)}.png"
        if _save_png(seg_dir / fname, drawn):
            saved += 1
    print(f"경계 디버그 스크린샷: {saved}장 → {seg_dir}", flush=True)
    return seg_dir if saved else None


def _export_verify_frames(
    frame: np.ndarray,
    t_sec: float,
    ok: bool,
    mse_dict: dict[str, float],
    left_right_only: bool,
    export_dir: Path,
) -> bool:
    """
    검증용: 프로그램이 보는 영역(크롭)과 좌/중/우 구간 이미지를 저장.
    반환: 두 장 모두 성공 시 True, 하나라도 실패 시 False.
    """
    if not CV2_AVAILABLE:
        return False
    export_dir = export_dir.resolve()

    def _save_png(path: Path, img: np.ndarray) -> bool:
        """cv2.imencode + Path.write_bytes 로 저장 (한글 경로 등 Windows Unicode 호환)."""
        ret, buf = cv2.imencode(".png", img)
        if not ret or buf is None:
            return False
        try:
            path.write_bytes(buf.tobytes())
            return True
        except OSError:
            return False

    crop = crop_middle_y(frame)
    best_L = mse_dict.get("best_L")
    if best_L is not None:
        left, center, right = split_third_with_left_width(crop, int(round(best_L)))
    else:
        left, center, right = split_third(crop)
    fname = _ts_to_export_fname(t_sec)
    # 1) 크롭 영역 저장 (프로그램이 보는 영역)
    crop_path = export_dir / f"crop_{fname}.png"
    ok1 = _save_png(crop_path, crop)
    if not ok1:
        print(f"[경고] 이미지 저장 실패: {crop_path}", file=sys.stderr, flush=True)
    # 2) 좌|중|우 나란히 + (좌·우만 모드 시) 차이 열지도 + 판정 이유 문구
    left_h, c_h, right_h = left.shape[0], center.shape[0], right.shape[0]
    h_min = min(left_h, c_h, right_h)
    left = left[:h_min, :]
    center = center[:h_min, :]
    right = right[:h_min, :]
    combined = np.hstack([left, center, right])
    # cv2.putText does not support Unicode (CJK); use English for image labels
    verdict = "Tripartite OK" if ok else "Tripartite NG"
    mse_val = next((v for k, v in mse_dict.items() if k != "best_L"), 0)
    label = f"{verdict}  MSE={mse_val:.1f} (threshold {MSE_THRESHOLD})"
    cv2.putText(
        combined, label, (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if ok else (0, 0, 255), 2,
    )
    if not ok and left_right_only:
        reason = "Left vs right pixels differ; MSE exceeds threshold (500)."
        cv2.putText(
            combined, reason, (10, 56),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
        )
    w1, w2 = left.shape[1], left.shape[1] + center.shape[1]
    cv2.line(combined, (w1, 0), (w1, h_min), (128, 128, 128), 2)
    cv2.line(combined, (w2, 0), (w2, h_min), (128, 128, 128), 2)
    # 좌·우 차이 열지도 (빨강=차이 큼) — 어디가 다른지 시각화
    if left_right_only and left.shape == right.shape:
        diff_sq = (left.astype(np.float64) - right.astype(np.float64)) ** 2
        mse_map = np.mean(diff_sq, axis=2)
        mse_max = max(mse_map.max(), 1.0)
        mse_map_u8 = (np.clip(mse_map / mse_max, 0, 1) * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(mse_map_u8, cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (left.shape[1], h_min))
        cv2.putText(heatmap, "Diff (red=more different)", (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.line(heatmap, (0, 0), (heatmap.shape[1], 0), (200, 200, 200), 1)
        combined = np.hstack([combined, heatmap])
    lcr_path = export_dir / f"left_center_right_{fname}_mse{mse_val:.0f}.png"
    ok2 = _save_png(lcr_path, combined)
    if not ok2:
        print(f"[경고] 이미지 저장 실패: {lcr_path}", file=sys.stderr, flush=True)
    return ok1 and ok2


def parse_ts_to_sec(s: str) -> float:
    """HH:MM:SS.mmm 또는 초 단위 문자열을 초(float)로 변환."""
    s = s.strip()
    if ":" in s:
        parts = s.split(":")
        if len(parts) == 3:
            h, m, sec = float(parts[0]), float(parts[1]), float(parts[2])
            return h * 3600 + m * 60 + sec
    try:
        return float(s)
    except ValueError:
        raise ValueError(f"타임스탬프 형식이 올바르지 않습니다: {s!r} (예: 02:21:48.357 또는 초)") from None


def verify_segment(
    path: Path,
    start_sec: float,
    end_sec: float,
    width: int,
    height: int,
    interval_sec: float,
    left_right_only: bool,
    use_gpu: bool,
    export_dir: Path | None = None,
    export_max: int = 20,
    export_only_x: bool = False,
    align_tolerance_px: int | None = None,
) -> None:
    """
    지정 구간을 일정 간격으로 샘플링해 각 시점의 삼분할 여부와 MSE를 출력.
    export_dir이 있으면 해당 시점의 크롭 영역·좌/중/우 이미지를 저장 (프로그램이 보는 영역 확인용).
    """
    print(f"검증 구간: {format_ts(start_sec)} ~ {format_ts(end_sec)}  (길이 {end_sec - start_sec:.1f}초)")
    print(f"샘플 간격: {interval_sec}초, 기준 MSE 한계: {MSE_THRESHOLD}, 모드: {'좌·우만' if left_right_only else '좌=중앙=우'}")
    if export_dir is not None:
        export_dir = export_dir.resolve()
        if not CV2_AVAILABLE:
            print("[경고] --verify-export를 쓰려면 opencv-python이 필요합니다. pip install opencv-python", flush=True)
        else:
            try:
                export_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                print(f"[오류] 저장 폴더 생성 실패: {export_dir}  ({e})", file=sys.stderr, flush=True)
                export_dir = None
            else:
                cap = "전체" if export_max <= 0 else f"최대 {export_max}개"
                only_x = " (삼분할 X만)" if export_only_x else ""
                print(f"이미지 저장: {export_dir}  (샘플당 2장, {cap} 시점{only_x})", flush=True)
                print(f"  - crop_*.png: 비교에 쓰는 영역 (높이 {Y_CROP_RATIO_START*100:.0f}%~{Y_CROP_RATIO_END*100:.0f}%)")
                print(f"  - left_center_right_*.png: 좌|중|우 구간 + 차이 열지도(빨강=다름) + 판정 이유")
    print("-" * 60)
    samples: list[tuple[float, bool, dict[str, float]]] = []
    export_count = 0
    t = start_sec
    while t <= end_sec:
        frame = extract_frame(path, t, width, height, use_gpu)
        if frame is None:
            print(f"  {format_ts(t)}  [프레임 추출 실패]", flush=True)
            t += interval_sec
            continue
        ok, mse_dict = get_tripartite_mse(
            frame, left_right_only, use_gpu, align_tolerance_px=align_tolerance_px
        )
        samples.append((t, ok, mse_dict))
        mse_str = ", ".join(f"{k}={v:.1f}" for k, v in mse_dict.items())
        verdict = "삼분할 O" if ok else "삼분할 X"
        print(f"  {format_ts(t)}  {verdict}  |  {mse_str}  (기준 {MSE_THRESHOLD})", flush=True)
        if export_dir is not None and CV2_AVAILABLE and (export_max <= 0 or export_count < export_max):
            if not export_only_x or not ok:
                if _export_verify_frames(frame, t, ok, mse_dict, left_right_only, export_dir):
                    export_count += 1
        t += interval_sec
    if export_dir is not None and CV2_AVAILABLE and export_count > 0:
        print(f"저장 완료: {export_count}개 시점 (총 {export_count * 2}장) → {export_dir}", flush=True)
    # 요약
    n_ok = sum(1 for _, ok, _ in samples if ok)
    n_total = len(samples)
    print("-" * 60)
    print(f"요약: {n_ok}/{n_total}개 시점이 삼분할로 판정됨.")
    if n_total > 0 and n_ok < n_total:
        all_mse = [v for _, _, d in samples for v in d.values()]
        print(f"  MSE 범위: {min(all_mse):.1f} ~ {max(all_mse):.1f}  (기준 이하면 O)")
        print(f"  → 기준({MSE_THRESHOLD})보다 MSE가 크면 삼분할이 아님으로 판정됩니다.")
        print(f"  → 파인튜닝: MSE_THRESHOLD를 올리면 더 넓게 삼분할로 인정되나, 오탐이 늘 수 있습니다.")


def _format_elapsed(sec: float) -> str:
    """소요 시간 요약용: 60초 미만이면 X.X초, 이상이면 N분 N초."""
    if sec >= 60:
        m, s = int(sec // 60), int(sec % 60)
        return f"{m}분 {s}초"
    return f"{sec:.2f}초"


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
    """멀티프로세싱용: (path_str, t, width, height, left_right_only, use_gpu, scale_width?, cache_dir?, align_tolerance_px?) → (t, is_tripartite).
    scale_width가 None이 아니면 코스 스캔만 저해상도. cache_dir이 있으면 해당 프레임을 .npy로 저장."""
    path_str, t, width, height, left_right_only, use_gpu, scale_width, cache_dir = args[:8]
    align_tol_px = int(args[8]) if len(args) > 8 and args[8] is not None else None
    path = Path(path_str)
    frame = extract_frame(path, t, width, height, use_gpu, scale_width=scale_width)
    if frame is None:
        return (t, False)
    if cache_dir:
        try:
            out = Path(cache_dir) / f"frame_{t:.3f}.npy"
            np.save(out, frame)
        except (OSError, ValueError):
            pass
    return (t, is_tripartite_frame(frame, left_right_only, use_gpu, align_tolerance_px=align_tol_px))


def _get_frame_for_check(
    path: Path,
    t: float,
    width: int,
    height: int,
    use_gpu: bool,
    cache_dir: Path | None,
    cached_times: list[float] | None,
) -> np.ndarray | None:
    """캐시가 있으면 0.5초 이내 캐시 프레임 반환, 없으면 extract_frame. 경계 정밀화용."""
    if cache_dir is not None and cached_times:
        best = None
        best_d = float("inf")
        for tc in cached_times:
            d = abs(t - tc)
            if d <= BOUNDARY_TOLERANCE and d < best_d:
                best_d = d
                best = tc
        if best is not None:
            fpath = cache_dir / f"frame_{best:.3f}.npy"
            if fpath.is_file():
                try:
                    frame = np.load(fpath)
                    if frame.ndim == 3 and frame.shape[2] == 3:
                        return frame
                except (OSError, ValueError):
                    pass
    return extract_frame(path, t, width, height, use_gpu)


def _check_tripartite_at(
    path: Path,
    t: float,
    width: int,
    height: int,
    left_right_only: bool = True,
    use_gpu: bool = False,
    cache_dir: Path | None = None,
    cached_times: list[float] | None = None,
    cache_usable: bool = False,
    align_tolerance_px: int | None = None,
) -> bool:
    """시점 t에서 1프레임 추출(또는 캐시 로드) 후 삼분할 여부 반환."""
    if cache_usable and cache_dir is not None and cached_times is not None:
        frame = _get_frame_for_check(path, t, width, height, use_gpu, cache_dir, cached_times)
    else:
        frame = extract_frame(path, t, width, height, use_gpu)
    return frame is not None and is_tripartite_frame(
        frame, left_right_only, use_gpu, align_tolerance_px=align_tolerance_px
    )


def _binary_search_first_tripartite(
    path: Path,
    t_low: float,
    t_high: float,
    width: int,
    height: int,
    left_right_only: bool = True,
    use_gpu: bool = False,
    cache_dir: Path | None = None,
    cached_times: list[float] | None = None,
    cache_usable: bool = False,
    tolerance_sec: float | None = None,
    align_tolerance_px: int | None = None,
) -> float:
    """[t_low, t_high] 구간에서 삼분할이 처음 나오는 시점을 이진 탐색. (t_high에서 True인 전제)"""
    tol = tolerance_sec if tolerance_sec is not None else BOUNDARY_TOLERANCE
    if _check_tripartite_at(
        path, t_low, width, height, left_right_only, use_gpu,
        cache_dir=cache_dir, cached_times=cached_times, cache_usable=cache_usable,
        align_tolerance_px=align_tolerance_px,
    ):
        return t_low
    while t_high - t_low > tol:
        mid = (t_low + t_high) * 0.5
        if _check_tripartite_at(
            path, mid, width, height, left_right_only, use_gpu,
            cache_dir=cache_dir, cached_times=cached_times, cache_usable=cache_usable,
            align_tolerance_px=align_tolerance_px,
        ):
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
    cache_dir: Path | None = None,
    cached_times: list[float] | None = None,
    cache_usable: bool = False,
    tolerance_sec: float | None = None,
    align_tolerance_px: int | None = None,
) -> float:
    """[t_low, t_high] 구간에서 삼분할이 마지막으로 나오는 시점을 이진 탐색. (t_low에서 True인 전제)"""
    tol = tolerance_sec if tolerance_sec is not None else BOUNDARY_TOLERANCE
    if _check_tripartite_at(
        path, t_high, width, height, left_right_only, use_gpu,
        cache_dir=cache_dir, cached_times=cached_times, cache_usable=cache_usable,
        align_tolerance_px=align_tolerance_px,
    ):
        return t_high
    while t_high - t_low > tol:
        mid = (t_low + t_high) * 0.5
        if _check_tripartite_at(
            path, mid, width, height, left_right_only, use_gpu,
            cache_dir=cache_dir, cached_times=cached_times, cache_usable=cache_usable,
            align_tolerance_px=align_tolerance_px,
        ):
            t_low = mid
        else:
            t_high = mid
    return t_low


def _boundary_worker(args: tuple) -> tuple[float, float] | None:
    """
    멀티프로세싱용: 한 후보 구간의 시작·끝을 이진 탐색으로 정밀화.
    인자 끝에 cache_dir, cached_times, cache_usable, boundary_tolerance_sec, align_tolerance_px 추가 가능.
    """
    n = len(args)
    path_str, t_prev, t_start_cand, t_end_cand, t_next, duration, width, height, left_right_only, use_gpu = args[:10]
    cache_dir = Path(args[10]) if n > 10 and args[10] else None
    cached_times = list(args[11]) if n > 11 and args[11] else None
    cache_usable = bool(args[12]) if n > 12 else False
    tolerance_sec = float(args[13]) if n > 13 and args[13] is not None else None
    align_tol_px = int(args[14]) if n > 14 and args[14] is not None else None
    path = Path(path_str)
    t_start = _binary_search_first_tripartite(
        path, t_prev, t_start_cand, width, height, left_right_only, use_gpu,
        cache_dir=cache_dir, cached_times=cached_times, cache_usable=cache_usable,
        tolerance_sec=tolerance_sec, align_tolerance_px=align_tol_px,
    )
    t_end = _binary_search_last_tripartite(
        path, t_end_cand, min(t_next, duration), width, height, left_right_only, use_gpu,
        cache_dir=cache_dir, cached_times=cached_times, cache_usable=cache_usable,
        tolerance_sec=tolerance_sec, align_tolerance_px=align_tol_px,
    )
    if t_end <= duration and (t_end - t_start) >= 0:
        return (t_start, t_end)
    return None


# 코스 스캔 저해상도: 허용 폭 (640, 960, 1280). 높이는 원본 비율로 계산.
COARSE_SCALE_WIDTHS = (640, 960, 1280)


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
    coarse_scale_width: int | None = None,
    use_coarse_cache: bool = False,
    boundary_tolerance_sec: float | None = None,
    align_tolerance_px: int | None = None,
) -> tuple[list[tuple[float, float]], dict[str, float]]:
    """
    삼분할 구간 탐색 (장편 영상 최적화).
    coarse_scale_width가 640/960/1280이면 코스 스캔만 해당 폭으로 저해상도 추출(속도 실험용). 경계 정밀화는 원본 해상도.
    use_coarse_cache=True면 코스 스캔 시 각 프레임을 임시 폴더에 저장하고, 경계 정밀화에서 원본 해상도일 때만 재사용 후 삭제.
    boundary_tolerance_sec: 경계 이진 탐색 정밀도(초). None이면 BOUNDARY_TOLERANCE 사용. 크게 주면 경계 단계가 빨라지나 정확도 완화.
    반환: (최종 구간 목록, 단계별 소요 시간 dict). 파이프라인 분석·효율화용.
    """
    timings: dict[str, float] = {}
    path_str = str(path.resolve())
    scale_w = coarse_scale_width if coarse_scale_width in COARSE_SCALE_WIDTHS else None

    cache_dir: Path | None = None
    if use_coarse_cache:
        try:
            cache_dir = Path(tempfile.mkdtemp(prefix="tripartite_cache_", dir=str(path.parent)))
        except (OSError, PermissionError):
            cache_dir = Path(tempfile.mkdtemp(prefix="tripartite_cache_"))
        print(f"  캐시 임시 폴더: {cache_dir}  (작업 후 삭제)", flush=True)
    cache_dir_str = str(cache_dir) if cache_dir else None

    try:
        # 1) 시점 목록 생성
        t0 = time.perf_counter()
        ts: list[float] = []
        t = 0.0
        while t < duration:
            ts.append(t)
            t += coarse_interval_sec
        if duration > 0 and (not ts or ts[-1] < duration - 0.5):
            ts.append(max(0.0, duration - 0.1))
        total_points = len(ts)
        worker_args = [
            (path_str, t, width, height, left_right_only, use_gpu, scale_w, cache_dir_str, align_tolerance_px)
            for t in ts
        ]
        timings["시점 목록"] = time.perf_counter() - t0

        coarse_results: list[tuple[float, bool]] = []
        scan_start = time.perf_counter()

        if n_workers <= 1:
            for i, (_path_s, t, w, h, lr, ug, sw, _cd, align_tol) in enumerate(worker_args):
                frame = extract_frame(path, t, w, h, ug, scale_width=sw)
                if frame is None:
                    coarse_results.append((t, False))
                else:
                    if cache_dir is not None:
                        try:
                            np.save(cache_dir / f"frame_{t:.3f}.npy", frame)
                        except (OSError, ValueError):
                            pass
                    coarse_results.append((t, is_tripartite_frame(frame, lr, ug, align_tolerance_px=align_tol)))
                if (i + 1) % 30 == 0:
                    _print_progress(ts[i], duration, scan_start, i + 1, total_points)
        else:
            done = 0
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                for res in executor.map(_coarse_worker, worker_args, chunksize=max(1, total_points // (n_workers * 4))):
                    coarse_results.append(res)
                    done += 1
                    if done % 30 == 0:
                        _print_progress(ts[done - 1] if done <= len(ts) else duration, duration, scan_start, done, total_points)
            sort_start = time.perf_counter()
            coarse_results.sort(key=lambda x: x[0])
            timings["코스 정렬"] = time.perf_counter() - sort_start

        scan_elapsed = time.perf_counter() - scan_start
        timings["코스 스캔"] = scan_elapsed
        if scan_elapsed >= 60:
            em, es = int(scan_elapsed // 60), int(scan_elapsed % 60)
            scan_str = f"{em}분 {es}초"
        else:
            scan_str = f"{scan_elapsed:.1f}초"
        print(f"  코스 스캔 완료: {total_points}개 시점, 소요 {scan_str}", flush=True)

        # 캐시된 시점 목록 (경계 정밀화에서 재사용용). 원본 해상도일 때만 경계에서 사용.
        cached_times: list[float] = []
        if cache_dir is not None and cache_dir.is_dir():
            for f in cache_dir.glob("frame_*.npy"):
                try:
                    cached_times.append(float(f.stem.replace("frame_", "")))
                except ValueError:
                    continue
            cached_times.sort()
        cache_usable_in_boundary = (cache_dir is not None and scale_w is None and len(cached_times) > 0)
        boundary_tol = boundary_tolerance_sec if boundary_tolerance_sec is not None else BOUNDARY_TOLERANCE

        # 2) 후보 구간 수집
        cand_start = time.perf_counter()
        candidates: list[tuple] = []
        i = 0
        while i < len(coarse_results):
            t_cur, is_tri = coarse_results[i]
            if not is_tri:
                i += 1
                continue
            t_start_cand = t_cur
            t_end_cand = t_cur
            j = i
            while j < len(coarse_results) and coarse_results[j][1]:
                t_end_cand = coarse_results[j][0]
                j += 1
            t_next = coarse_results[j][0] if j < len(coarse_results) else duration
            t_prev = coarse_results[i - 1][0] if i > 0 else 0.0
            candidates.append(
                (
                    path_str,
                    t_prev,
                    t_start_cand,
                    t_end_cand,
                    t_next,
                    duration,
                    width,
                    height,
                    left_right_only,
                    use_gpu,
                    cache_dir_str,
                    cached_times,
                    cache_usable_in_boundary,
                    boundary_tol,
                    align_tolerance_px,
                )
            )
            i = j
        timings["후보 구간 수집"] = time.perf_counter() - cand_start

        # 3) 경계 정밀화
        boundary_start = time.perf_counter()
        segments = []
        if n_workers <= 1 or len(candidates) == 0:
            for c in candidates:
                (_path_s, t_prev, t_start_cand, t_end_cand, t_next, _dur, w, h, lr, ug, _cd, _ct, cache_ok, tol, align_tol) = c
                t_start = _binary_search_first_tripartite(
                    path, t_prev, t_start_cand, w, h, lr, ug,
                    cache_dir=cache_dir, cached_times=cached_times if cache_ok else None, cache_usable=cache_ok,
                    tolerance_sec=tol, align_tolerance_px=align_tol,
                )
                t_end = _binary_search_last_tripartite(
                    path, t_end_cand, min(t_next, duration), w, h, lr, ug,
                    cache_dir=cache_dir, cached_times=cached_times if cache_ok else None, cache_usable=cache_ok,
                    tolerance_sec=tol, align_tolerance_px=align_tol,
                )
                if t_end <= duration and (t_end - t_start) >= 0:
                    segments.append((t_start, t_end))
                    print(f"  [발견] {format_ts(t_start)} ~ {format_ts(t_end)}", flush=True)
        else:
            n_boundary = min(n_workers, len(candidates))
            with ProcessPoolExecutor(max_workers=n_boundary) as executor:
                results = list(
                    executor.map(
                        _boundary_worker,
                        candidates,
                        chunksize=max(1, len(candidates) // (n_boundary * 2)),
                    )
                )
            for res in results:
                if res is not None:
                    segments.append(res)
            segments.sort(key=lambda x: x[0])
            for t_start, t_end in segments:
                print(f"  [발견] {format_ts(t_start)} ~ {format_ts(t_end)}", flush=True)

        boundary_elapsed = time.perf_counter() - boundary_start
        timings["경계 정밀화"] = boundary_elapsed
        if boundary_elapsed >= 60:
            bm, bs = int(boundary_elapsed // 60), int(boundary_elapsed % 60)
            boundary_str = f"{bm}분 {bs}초"
        else:
            boundary_str = f"{boundary_elapsed:.1f}초"
        print(f"  경계 정밀화(발견) 완료: {len(segments)}개, 소요 {boundary_str}", flush=True)

        # 4) 인접 구간 병합
        merge_start = time.perf_counter()
        merged: list[tuple[float, float]] = []
        for start, end in sorted(segments):
            if merged and (start - merged[-1][1]) <= MERGE_GAP_SECONDS:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        timings["병합"] = time.perf_counter() - merge_start

        # 5) 최소 길이 미만 구간 제거 (min_duration_sec이 0이면 제거 안 함)
        filter_start = time.perf_counter()
        result = [(a, b) for a, b in merged if (b - a) >= min_duration_sec]
        timings["최소 길이 필터"] = time.perf_counter() - filter_start
        return (result, timings)
    finally:
        if cache_dir is not None and cache_dir.is_dir():
            try:
                shutil.rmtree(cache_dir, ignore_errors=True)
            except OSError:
                pass


def load_segments_from_file(file_path: Path) -> list[tuple[float, float]]:
    """
    구간 목록 텍스트 파일을 읽어 (start_sec, end_sec) 리스트로 반환.
    한 줄에 하나의 구간: "START END" (공백/탭 구분). START/END는 HH:MM:SS.mmm 또는 초.
    빈 줄, # 으로 시작하는 줄은 무시. 원하는 구간만 남기고 나머지 줄을 삭제하거나 # 처리하면 됨.
    """
    segments: list[tuple[float, float]] = []
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"구간 목록 파일을 찾을 수 없습니다: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            start = parse_ts_to_sec(parts[0])
            end = parse_ts_to_sec(parts[1])
            if end > start:
                segments.append((start, end))
        except ValueError:
            continue
    return segments


def write_segments_to_file(segments: list[tuple[float, float]], file_path: Path) -> None:
    """구간 목록을 텍스트 파일로 저장. 한 줄에 '시작 시각 끝 시각 # 길이(초)'. # 으로 시작하는 줄은 주석."""
    path = Path(file_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    comment = "# 시작 시각 끝 시각 (한 줄에 한 구간. 해당 줄 삭제 또는 # 붙이면 병합에서 제외)"
    lines = [comment] + [f"{format_ts(s)} {format_ts(e)} # {e - s:.1f}초" for s, e in segments]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _seg_output_path_for_input(input_path: Path) -> Path:
    """기본 구간 목록 파일명(원본_seg.txt). 이미 존재하면 _seg(1).txt, _seg(2).txt ... 중 존재하지 않는 이름 반환."""
    base = input_path.parent / (input_path.stem + "_seg.txt")
    if not base.exists():
        return base
    n = 1
    while True:
        candidate = input_path.parent / (input_path.stem + f"_seg({n}).txt")
        if not candidate.exists():
            return candidate
        n += 1


def _merge_segment_worker(args: tuple) -> tuple:
    """
    병렬용: 한 구간을 cut → 키프레임 탐색 → 앞·뒤 키프레임에 맞춰 트림해 segment_xxxx.ts 생성.
    args = (path_str, start, end, index_0based, tmpdir_str, use_gpu, leading_buffer_sec, trailing_buffer_sec)
    반환: (index_0based, success, elapsed_sec, first_kf, last_kf, duration_cut, buffer_lead, buffer_trail, cut_sec, trim_sec)
    """
    leading_sec = args[6]
    trailing_sec = args[7] if len(args) > 7 else 0.0
    path_str, start, end, idx, tmpdir_str, use_gpu = args[0], args[1], args[2], args[3], args[4], args[5]
    t0 = time.perf_counter()
    tmpdir = Path(tmpdir_str)
    seg_path = tmpdir / f"segment_{idx + 1:04d}.ts"
    temp_seg = tmpdir / f"segment_{idx + 1:04d}_temp.ts"
    start_cut = max(0.0, start - leading_sec)
    end_cut = end + trailing_sec
    duration_cut = end_cut - start_cut
    cmd_cut = ["ffmpeg", "-y"]
    if use_gpu:
        cmd_cut.extend(["-hwaccel", "cuda"])
    cmd_cut.extend([
        "-ss", str(start_cut), "-i", path_str,
        "-t", str(duration_cut), "-c", "copy",
        "-avoid_negative_ts", "make_zero", str(temp_seg),
    ])
    t_cut_start = time.perf_counter()
    ret = subprocess.run(cmd_cut, capture_output=True, timeout=3600)
    cut_sec = time.perf_counter() - t_cut_start
    if ret.returncode != 0 or not temp_seg.is_file():
        return (idx, False, time.perf_counter() - t0, None, None, None, None, None, cut_sec, None)
    keyframes = _get_keyframe_times_from_file(temp_seg)
    # temp 파일 기준: 0 ~ duration_cut. 원본 구간 끝은 (end - start_cut)
    end_limit_in_file = end - start_cut
    first_kf = keyframes[0] if keyframes else 0.0
    kf_before_end = [k for k in keyframes if k <= end_limit_in_file] if keyframes else []
    last_kf = max(kf_before_end, default=end_limit_in_file)
    last_kf = min(last_kf, duration_cut)
    last_kf = max(first_kf, last_kf)
    trim_dur = last_kf - first_kf
    trim_sec = 0.0
    if trim_dur > 0.5:
        cmd_trim = [
            "ffmpeg", "-y", "-ss", str(first_kf), "-i", str(temp_seg),
            "-t", str(trim_dur), "-c", "copy", "-avoid_negative_ts", "make_zero", str(seg_path),
        ]
        t_trim_start = time.perf_counter()
        ret2 = subprocess.run(cmd_trim, capture_output=True, timeout=300)
        trim_sec = time.perf_counter() - t_trim_start
        temp_seg.unlink(missing_ok=True)
        if ret2.returncode != 0 or not seg_path.is_file():
            return (idx, False, time.perf_counter() - t0, None, None, None, None, None, cut_sec, trim_sec)
    else:
        temp_seg.rename(seg_path)
    elapsed = time.perf_counter() - t0
    buffer_lead = start - start_cut
    buffer_trail = end_cut - end
    return (idx, True, elapsed, first_kf, last_kf, duration_cut, buffer_lead, buffer_trail, cut_sec, trim_sec)


def _start_background_copy(src: str, dst: str) -> None:
    """
    스테이징 파일(src)을 최종 경로(dst)로 복사한 뒤 src를 삭제하는 프로세스를 백그라운드로 기동.
    부모 프로세스는 대기하지 않고 반환한다.
    """
    cmd = [
        sys.executable, "-c",
        "import shutil,sys; from pathlib import Path;"
        " shutil.copy2(sys.argv[1], sys.argv[2]);"
        " Path(sys.argv[1]).unlink(missing_ok=True)",
        src, dst,
    ]
    kwargs = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen(cmd, **kwargs)


def _merge_output_path_for_input(input_path: Path) -> Path:
    """기본 병합 파일명(원본_merged.mp4). 이미 존재하면 _merged(1).mp4, _merged(2).mp4 ... 중 존재하지 않는 이름 반환."""
    base = input_path.parent / (input_path.stem + "_merged.mp4")
    if not base.exists():
        return base
    n = 1
    while True:
        candidate = input_path.parent / (input_path.stem + f"_merged({n}).mp4")
        if not candidate.exists():
            return candidate
        n += 1


def merge_segments(
    path: Path,
    segments: list[tuple[float, float]],
    output_path: Path,
    use_gpu: bool = False,
    merge_workers: int = 2,
) -> bool:
    """
    구간 목록을 ffmpeg로 코덱 카피(-c copy)해 잘라 concat demuxer로 이어붙여 output_path에 저장.
    모든 구간: 시작·끝 각 3초 여유를 두고 잘라, 그 조각에서 첫 키프레임~마지막 키프레임만 사용해 이어붙임(키프레임 경계에서 잘림 방지).
    """
    LEADING_BUFFER_SEC = 3.0   # 각 구간 자를 때 시작점 이전 3초까지 포함
    TRAILING_BUFFER_SEC = 3.0  # 각 구간 자를 때 끝점 이후 3초까지 포함 → 뒤로부터 마지막 키프레임에서 자름
    if not segments:
        return False
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    path_str = str(path.resolve())
    # 임시 작업은 로컬 고정 경로(Videos 내 전용 폴더)에서 수행해 concat/쓰기 속도 체감
    merge_base = MERGE_TMP_BASE
    merge_base.mkdir(parents=True, exist_ok=True)
    tmpdir = Path(tempfile.mkdtemp(prefix="merge_", dir=str(merge_base)))
    print(f"  병합 임시 폴더: {tmpdir}  (작업 후 삭제)", flush=True)
    merge_start = time.perf_counter()
    try:
        list_path = tmpdir / "list.txt"
        for i, (start, end) in enumerate(segments):
            if (end - start) <= 0:
                print(f"[오류] 구간 [{i + 1}] 길이 <= 0: {format_ts(start)} ~ {format_ts(end)}", file=sys.stderr)
                return False
        n_seg = len(segments)
        n_workers = max(1, merge_workers)
        if n_seg >= 2:
            worker_args = [
                (path_str, start, end, i, str(tmpdir), use_gpu, LEADING_BUFFER_SEC, TRAILING_BUFFER_SEC)
                for i, (start, end) in enumerate(segments)
            ]
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                results = list(executor.map(_merge_segment_worker, worker_args, chunksize=1))
            for idx, ok, elapsed, first_kf, last_kf, duration_cut, buffer_lead, buffer_trail, cut_sec, trim_sec in results:
                if not ok:
                    print(f"[오류] 구간 [{idx + 1}] 추출 또는 트림 실패", file=sys.stderr)
                    return False
                cut_str = f" cut {cut_sec:.1f}초" if cut_sec is not None else ""
                trim_str = f" trim {trim_sec:.1f}초" if trim_sec is not None and trim_sec > 0 else ""
                print(f"  구간 [{idx + 1}/{n_seg}]  추출 완료 (앞 {buffer_lead:.1f}초·뒤 {buffer_trail:.1f}초 여유, 길이 {duration_cut:.1f}초){cut_str}{trim_str}  소요 {elapsed:.1f}초", flush=True)
                if first_kf is not None and first_kf > 0.01:
                    print(f"  구간 [{idx + 1}] 앞부분 트림: 첫 키프레임({first_kf:.2f}초)부터 사용 (앞 {first_kf:.1f}초 제거)", flush=True)
                if last_kf is not None and duration_cut is not None and (duration_cut - last_kf) > 0.01:
                    print(f"  구간 [{idx + 1}] 뒤부분 트림: 마지막 키프레임({last_kf:.2f}초)까지 사용 (뒤 {duration_cut - last_kf:.1f}초 제거)", flush=True)
        else:
            (start, end) = segments[0]
            seg_path = tmpdir / "segment_0001.ts"
            temp_seg = tmpdir / "segment_0001_temp.ts"
            start_cut = max(0.0, start - LEADING_BUFFER_SEC)
            end_cut = end + TRAILING_BUFFER_SEC
            duration_cut = end_cut - start_cut
            cmd_cut = ["ffmpeg", "-y"]
            if use_gpu:
                cmd_cut.extend(["-hwaccel", "cuda"])
            cmd_cut.extend([
                "-ss", str(start_cut), "-i", path_str,
                "-t", str(duration_cut), "-c", "copy",
                "-avoid_negative_ts", "make_zero", str(temp_seg),
            ])
            ret = subprocess.run(cmd_cut, capture_output=True, timeout=3600)
            if ret.returncode != 0 or not temp_seg.is_file():
                print(f"[오류] 구간 [1] 추출 실패: {format_ts(start_cut)} ~ {format_ts(end_cut)}", file=sys.stderr)
                return False
            print(f"  구간 [1/1]  추출 완료 (앞 {start - start_cut:.1f}초·뒤 {end_cut - end:.1f}초 여유, 길이 {duration_cut:.1f}초)", flush=True)
            keyframes = _get_keyframe_times_from_file(temp_seg)
            end_limit_in_file = end - start_cut
            first_kf = keyframes[0] if keyframes else 0.0
            kf_before_end = [k for k in keyframes if k <= end_limit_in_file] if keyframes else []
            last_kf = max(kf_before_end, default=end_limit_in_file)
            last_kf = min(last_kf, duration_cut)
            last_kf = max(first_kf, last_kf)
            trim_dur = last_kf - first_kf
            if trim_dur > 0.5:
                cmd_trim = [
                    "ffmpeg", "-y", "-ss", str(first_kf), "-i", str(temp_seg),
                    "-t", str(trim_dur), "-c", "copy", "-avoid_negative_ts", "make_zero", str(seg_path),
                ]
                ret2 = subprocess.run(cmd_trim, capture_output=True, timeout=300)
                temp_seg.unlink(missing_ok=True)
                if ret2.returncode != 0 or not seg_path.is_file():
                    print(f"[오류] 구간 [1] 키프레임 트림 실패", file=sys.stderr)
                    return False
                if first_kf > 0.01:
                    print(f"  구간 [1] 앞부분 트림: 첫 키프레임({first_kf:.2f}초)부터 사용 (앞 {first_kf:.1f}초 제거)", flush=True)
                if (duration_cut - last_kf) > 0.01:
                    print(f"  구간 [1] 뒤부분 트림: 마지막 키프레임({last_kf:.2f}초)까지 사용 (뒤 {duration_cut - last_kf:.1f}초 제거)", flush=True)
            else:
                temp_seg.rename(seg_path)
        with open(list_path, "w", encoding="utf-8") as f:
            for i in range(len(segments)):
                seg_name = f"segment_{i + 1:04d}.ts"
                f.write(f"file '{seg_name}'\n")
        # concat은 항상 로컬 임시 파일에 먼저 써서 속도 확보, 이후 최종 경로로 복사
        local_merged = tmpdir / "merged.mp4"
        concat_start = time.perf_counter()
        cmd_concat = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(list_path), "-c", "copy", str(local_merged),
        ]
        ret = subprocess.run(cmd_concat, capture_output=True, timeout=3600, cwd=str(tmpdir))
        concat_elapsed = time.perf_counter() - concat_start
        if ret.returncode != 0 or not local_merged.is_file():
            print("[오류] 구간 합치기(concat) 실패", file=sys.stderr)
            return False
        print(f"  concat 합치기(로컬)  소요 {concat_elapsed:.1f}초", flush=True)
        # 최종 경로가 로컬 임시와 다르면 복사. 비동기: 로컬 스테이징 후 백그라운드에서 최종 경로로 복사.
        if local_merged.resolve() != output_path.resolve():
            staging_path = merge_base / f"merged_staging_{time.time_ns()}.mp4"
            shutil.copy2(str(local_merged), str(staging_path))
            _start_background_copy(str(staging_path), str(output_path.resolve()))
            print(f"  최종 파일 복사 → 백그라운드 진행 중: {output_path}", flush=True)
            print(f"병합 완료(로컬). 네트워크/최종 경로로 복사는 백그라운드에서 진행 중.  (총 {len(segments)}개 구간, 병합 소요 {time.perf_counter() - merge_start:.1f}초)", flush=True)
        else:
            total_elapsed = time.perf_counter() - merge_start
            print(f"병합 완료: {output_path}  (총 {len(segments)}개 구간, 병합 총 소요 {total_elapsed:.1f}초)", flush=True)
        return True
    finally:
        try:
            for f in tmpdir.iterdir():
                f.unlink(missing_ok=True)
            tmpdir.rmdir()
        except OSError:
            pass


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
        "--coarse-scale",
        type=int,
        default=None,
        metavar="W",
        choices=COARSE_SCALE_WIDTHS,
        help="코스 스캔만 저해상도로 수행 (폭 640/960/1280). 미지정 시 원본 해상도. 경계 정밀화는 항상 원본.",
    )
    parser.add_argument(
        "--coarse-cache",
        action="store_true",
        help="코스 스캔 시 각 시점 프레임을 임시 폴더에 캐시. 경계 정밀화에서 원본 해상도일 때만 재사용 후 작업 끝에 삭제. --coarse-scale과 함께 사용 가능(캐시는 저장만, 경계에서는 원본 해상도 사용).",
    )
    parser.add_argument(
        "--boundary-tolerance",
        type=float,
        default=None,
        metavar="SEC",
        help="경계 이진 탐색 정밀도(초). 미지정 시 기본값(0.9375). 크게 주면 경계 단계가 빨라지나 정확도 완화. 예: 0.5(더 정밀), 1.0",
    )
    parser.add_argument(
        "--align-tolerance",
        type=int,
        default=None,
        metavar="PX",
        help="경계 정렬 탐색 허용(픽셀). 0이면 고정 1/3·2/3만 사용. N이면 w/3±N 픽셀 범위에서 최적 좌/우 너비 탐색. 미지정 시 기본값(0). 예: 10, 30",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="GPU 사용: ffmpeg CUDA 디코딩, CuPy 있으면 판정 연산도 GPU (NVIDIA 드라이버 필요)",
    )
    parser.add_argument(
        "--verify",
        nargs=2,
        metavar=("START", "END"),
        help="지정 구간만 검증: 삼분할 여부와 MSE를 샘플별로 리포팅 (파인튜닝·미검출 원인 분석용). 예: --verify 02:21:48.357 02:30:00.679",
    )
    parser.add_argument(
        "--verify-interval",
        type=float,
        default=5.0,
        metavar="SEC",
        help="--verify 시 샘플 간격(초). 기본 5. 더 촘촘히 보려면 1~2",
    )
    parser.add_argument(
        "--verify-export",
        type=Path,
        default=None,
        metavar="DIR",
        help="--verify 시 프로그램이 보는 영역(크롭)·좌/중/우 이미지를 저장할 폴더. opencv-python 필요",
    )
    parser.add_argument(
        "--verify-export-max",
        type=int,
        default=20,
        metavar="N",
        help="--verify-export 시 저장할 시점 개수. 0이면 전부. 기본 20",
    )
    parser.add_argument(
        "--verify-export-only-x",
        action="store_true",
        help="--verify-export 시 삼분할 X(불일치)로 판정된 시점만 이미지 저장",
    )
    parser.add_argument(
        "--debug-boundary",
        nargs="?",
        type=float,
        const=10.0,
        default=None,
        metavar="SEC",
        help="검출된 구간(첫 번째)의 시작·끝 전후 SEC초를 샘플링해 판정 근거(경계·MSE) 스크린샷 저장. 기본 10초. opencv-python 필요. 예: --debug-boundary 또는 --debug-boundary 15",
    )
    parser.add_argument(
        "--merge",
        nargs="?",
        const=True,
        default=None,
        metavar="OUTPUT",
        help="검출된 구간만 코덱 카피로 잘라서 한 파일로 이어붙여 저장. 인자 없으면 원본과 같은 폴더에 원본파일명_merged.mp4 로 저장. 경로/파일명을 주면 그 위치에 저장. 예: --merge 또는 --merge F:\\세경\\result.mp4",
    )
    parser.add_argument(
        "--merge-workers",
        type=int,
        default=2,
        metavar="N",
        help="병합 시 구간 추출 병렬 워커 수. 기본 2(네트워크/HDD 권장). 로컬 NVMe면 4~8 시도 가능. 예: --merge-workers 6",
    )
    parser.add_argument(
        "--segments-out",
        nargs="?",
        const=True,
        default=None,
        metavar="FILE",
        help="검출된 구간 목록을 텍스트 파일로 저장. 인자 없으면 원본과 같은 폴더에 원본파일명_seg.txt 로 저장. 경로 지정 시 그 위치에 저장",
    )
    parser.add_argument(
        "--segments-in",
        nargs="?",
        const=True,
        default=None,
        metavar="FILE",
        help="구간 검출 대신 파일에서 구간 목록 읽기. 인자 없으면 원본과 같은 폴더의 원본파일명_seg.txt 사용. 빈 줄·# 줄 무시",
    )
    args = parser.parse_args()

    path = args.input.resolve()
    if not path.is_file():
        print(f"[오류] 파일을 찾을 수 없습니다: {path}", file=sys.stderr)
        sys.exit(1)

    use_gpu = getattr(args, "cuda", False)
    segments_in = getattr(args, "segments_in", None)
    segments_out = getattr(args, "segments_out", None)

    # --segments-in: 파일에서 구간 목록 읽어서 출력·병합만 (검출 생략)
    if segments_in is not None:
        seg_file = (
            path.parent / (path.stem + "_seg.txt")
            if segments_in is True
            else Path(segments_in).resolve()
        )
        if not seg_file.is_file():
            print(f"[오류] 구간 목록 파일을 찾을 수 없습니다: {seg_file}", file=sys.stderr)
            sys.exit(1)
        try:
            segments = load_segments_from_file(seg_file)
        except FileNotFoundError as e:
            print(f"[오류] {e}", file=sys.stderr)
            sys.exit(1)
        if not segments:
            print("[오류] 구간 목록 파일에 유효한 구간이 없습니다.", file=sys.stderr)
            sys.exit(1)
        print(f"입력: {path}")
        print(f"구간 목록: {seg_file}  (총 {len(segments)}개)")
        print("삼분할 구간:")
        total_length_sec = 0.0
        for i, (start, end) in enumerate(segments, 1):
            seg_len = end - start
            total_length_sec += seg_len
            print(f"  [{i}] {format_ts(start)} ~ {format_ts(end)}  (길이 {seg_len:.1f}초)")
        print("-" * 50)
        print(f"길이 총합: {total_length_sec:.1f}초  ({_format_elapsed(total_length_sec)})")
        if args.merge is not None:
            output_path = (
                Path(args.merge).resolve()
                if args.merge is not True
                else _merge_output_path_for_input(path)
            )
            print(f"구간 병합 중... (코덱 카피) → {output_path}", flush=True)
            if not merge_segments(path, segments, output_path, use_gpu, merge_workers=args.merge_workers):
                sys.exit(1)
        return

    print(f"입력: {path}")
    meta_start = time.perf_counter()
    try:
        duration, width, height = get_video_info(path)
    except RuntimeError as e:
        print(f"[오류] {e}", file=sys.stderr)
        sys.exit(1)
    meta_elapsed = time.perf_counter() - meta_start

    left_right_only = not args.strict

    # --verify: 지정 구간만 샘플링해 삼분할 여부·MSE 리포팅 후 종료
    if args.verify is not None:
        try:
            start_sec = parse_ts_to_sec(args.verify[0])
            end_sec = parse_ts_to_sec(args.verify[1])
        except ValueError as e:
            print(f"[오류] {e}", file=sys.stderr)
            sys.exit(1)
        interval = max(0.5, getattr(args, "verify_interval", 5.0))
        print(f"길이: {format_ts(duration)} ({duration:.1f}초), 해상도: {width}x{height}")
        print(f"비교 영역: 높이 {Y_CROP_RATIO_START*100:.0f}% ~ {Y_CROP_RATIO_END*100:.0f}%, 모드: {'좌·우만' if left_right_only else '좌=중앙=우'}")
        print("-" * 50)
        verify_segment(
            path, start_sec, end_sec, width, height,
            interval, left_right_only, use_gpu,
            export_dir=getattr(args, "verify_export", None),
            export_max=max(0, getattr(args, "verify_export_max", 20)),
            export_only_x=getattr(args, "verify_export_only_x", False),
            align_tolerance_px=args.align_tolerance,
        )
        return

    run_start = time.perf_counter()
    min_dur = 0.0 if args.no_min_duration else MIN_SEGMENT_DURATION
    coarse = max(1.0, args.coarse_interval)
    workers = max(0, args.workers)
    print(f"길이: {format_ts(duration)} ({duration:.1f}초), 해상도: {width}x{height}")
    print(f"비교 영역: 높이 {Y_CROP_RATIO_START*100:.0f}% ~ {Y_CROP_RATIO_END*100:.0f}% (상단 팝업 제외)")
    print(f"모드: {'좌·우만 비교 (중앙 무시)' if left_right_only else '좌=중앙=우 모두 비교 (strict)'}")
    gpu_desc = "GPU(CUDA 디코딩" + (", CuPy 판정" if CUPY_AVAILABLE else "") + ")" if use_gpu else "CPU"
    coarse_scale = getattr(args, "coarse_scale", None)
    coarse_scale = coarse_scale if coarse_scale in COARSE_SCALE_WIDTHS else None
    coarse_res_str = ""
    if coarse_scale is not None:
        coarse_h = round(height * coarse_scale / width)
        coarse_res_str = f", 코스 해상도 {coarse_scale}×{coarse_h} (저해상도)"
    print(f"코스 스캔: 간격 {coarse}s, 병렬 프로세스 {workers}개, 자원 {gpu_desc}{coarse_res_str}")
    print("-" * 50)

    use_coarse_cache = getattr(args, "coarse_cache", False)
    segments, phase_timings = find_segments(
        path, duration, width, height,
        min_duration_sec=min_dur,
        coarse_interval_sec=coarse,
        left_right_only=left_right_only,
        n_workers=workers,
        use_gpu=use_gpu,
        coarse_scale_width=coarse_scale,
        use_coarse_cache=use_coarse_cache,
        boundary_tolerance_sec=args.boundary_tolerance,
        align_tolerance_px=args.align_tolerance,
    )

    run_elapsed = time.perf_counter() - run_start
    # 파이프라인 단계별 소요 시간 요약 (효율화 분석용)
    all_phases = [
        "메타(ffprobe)",
        "시점 목록",
        "코스 스캔",
        "코스 정렬",
        "후보 구간 수집",
        "경계 정밀화",
        "병합",
        "최소 길이 필터",
    ]
    full_timings: dict[str, float] = {"메타(ffprobe)": meta_elapsed, **phase_timings}
    total_sec = sum(full_timings.get(p, 0.0) for p in all_phases)
    if total_sec <= 0:
        total_sec = run_elapsed
    name_width = max(len(p) for p in all_phases)
    time_width = 10  # "123.45초" 또는 "12분 34초" 수준
    print("소요 시간 요약 (파이프라인):")
    for name in all_phases:
        sec = full_timings.get(name, 0.0)
        if sec <= 0 and name != "메타(ffprobe)":
            continue
        pct = (sec / total_sec * 100) if total_sec > 0 else 0
        print(f"  {name:<{name_width}}  {_format_elapsed(sec):>{time_width}}  ({pct:>5.1f}%)", flush=True)
    print(f"  {'─ 합계':<{name_width}}  {_format_elapsed(total_sec):>{time_width}}  (전체 실행: {_format_elapsed(run_elapsed)})", flush=True)
    print("-" * 50)

    if not segments:
        print("삼분할 구간이 없습니다.")
        return

    print("삼분할 구간:")
    total_length_sec = 0.0
    for i, (start, end) in enumerate(segments, 1):
        seg_len = end - start
        total_length_sec += seg_len
        print(f"  [{i}] {format_ts(start)} ~ {format_ts(end)}  (길이 {seg_len:.1f}초)")
    print("-" * 50)
    print(f"총 {len(segments)}개 구간  (전체 소요: {_format_elapsed(run_elapsed)})")
    print(f"길이 총합: {total_length_sec:.1f}초  ({_format_elapsed(total_length_sec)})")

    if getattr(args, "debug_boundary", None) is not None and segments:
        margin_sec = float(args.debug_boundary)
        start, end = segments[0]
        run_debug_boundary(
            path, start, end, margin_sec,
            width, height, use_gpu, left_right_only, duration,
        )

    if segments_out is not None:
        out_path = (
            _seg_output_path_for_input(path)
            if segments_out is True
            else Path(segments_out).resolve()
        )
        write_segments_to_file(segments, out_path)
        print(f"구간 목록 저장: {out_path}  (편집 후 --segments-in으로 읽어 병합 가능)", flush=True)

    if args.merge is not None:
        output_path = (
            Path(args.merge).resolve()
            if args.merge is not True
            else _merge_output_path_for_input(path)
        )
        print(f"구간 병합 중... (코덱 카피) → {output_path}", flush=True)
        if not merge_segments(path, segments, output_path, use_gpu, merge_workers=args.merge_workers):
            sys.exit(1)


if __name__ == "__main__":
    main()
