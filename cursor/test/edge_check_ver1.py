"""
삼분할 영상 엣지 검출: 1/3, 2/3 위치의 수직 엣지 강도로 삼분할 여부 판정.
사용: python edge_check.py "image.jpg"
디버그 이미지: 입력 이미지와 동일 경로에 input_debug.jpg 로 저장.
"""

import argparse
import os
import time
import cv2
import numpy as np


def fast_edge_check(frame_gray, band_px=3, threshold_mult=5.0):
    """
    x방향 Sobel로 1/3, 2/3 위치의 수직 엣지 강도를 측정해 삼분할 여부 판정.
    :param frame_gray: 그레이스케일 이미지 (H, W)
    :param band_px: 엣지 강도 합산 시 사용할 수직 밴드 폭(픽셀). 좌우 각 band_px만큼.
    :param threshold_mult: 로컬 참조 대비 이 배수 이상이면 엣지 있음으로 간주.
    :return: (is_tripartite, edge_strength_left, edge_strength_right, avg_strength_ref, match_ratio)
    """
    h, w = frame_gray.shape
    slice_w = w // 3

    # [개선 1] 가우시안 블러로 미세 노이즈 제거 후 Sobel 실행
    blurred = cv2.GaussianBlur(frame_gray, (3, 3), 0)
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_x = np.abs(sobel_x)

    # [개선 2] 특정 지점의 강도를 '밴드 내 최댓값' 기반으로 분석 (단순 합산보다 정확)
    def get_peak_strength(center_x):
        lo = max(0, center_x - band_px)
        hi = min(w, center_x + band_px + 1)
        col_sums = np.sum(sobel_x[:, lo:hi], axis=0)
        return np.max(col_sums)

    edge_strength_left = get_peak_strength(slice_w)
    edge_strength_right = get_peak_strength(slice_w * 2)

    # [개선 3] 로컬 기준값 계산 (전체 평균이 아닌, 경계선 주변부의 평균과 비교)
    ref_area_w = 20
    ref_l_lo = max(0, slice_w - 100)
    ref_l_hi = min(w, slice_w - 100 + ref_area_w)
    ref_r_lo = max(0, slice_w * 2 + 100)
    ref_r_hi = min(w, slice_w * 2 + 100 + ref_area_w)
    # [수정] 로컬 참조 영역의 '평균' 수직 합으로 단위를 맞춤 (get_peak_strength와 동일 단위)
    ref_l = (np.sum(sobel_x[:, ref_l_lo:ref_l_hi]) / ref_area_w) if ref_area_w > 0 and ref_l_hi > ref_l_lo else 0.0
    ref_r = (np.sum(sobel_x[:, ref_r_lo:ref_r_hi]) / ref_area_w) if ref_area_w > 0 and ref_r_hi > ref_r_lo else 0.0
    # 이제 단위가 맞으므로, 일반적인 배경 엣지보다 threshold_mult배 강한지 비교 가능
    local_ref = max(ref_l, ref_r) * threshold_mult

    pass_left = edge_strength_left > local_ref
    pass_right = edge_strength_right > local_ref
    is_tripartite = pass_left and pass_right

    # 일치율: 두 엣지 모두 기준 통과면 100%, 하나만 통과면 50%, 둘 다 미통과면 0%
    if pass_left and pass_right:
        match_ratio = 1.0
    elif pass_left or pass_right:
        match_ratio = 0.5
    else:
        match_ratio = 0.0

    return is_tripartite, edge_strength_left, edge_strength_right, local_ref, match_ratio


def fast_edge_check_v3(frame_gray, search_range=15, threshold_mult=5.0):
    """
    640/1280 부근 피크 검색 + 간격 조건으로 삼분할 판정.
    :param frame_gray: 그레이스케일 이미지 (H, W)
    :param search_range: 640/1280 주변 검색 범위(픽셀).
    :param threshold_mult: 로컬 참조 대비 이 배수 이상이면 엣지 있음으로 간주.
    :return: (is_tripartite, peak_l_idx, peak_r_idx)
    """
    h, w = frame_gray.shape

    # 1. Sobel 처리 (기존과 동일)
    blurred = cv2.GaussianBlur(frame_gray, (3, 3), 0)
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_x = np.abs(sobel_x)

    # 수직 합(Column Sum) 계산
    col_sums = np.sum(sobel_x, axis=0)

    # 2. 첫 번째 경계선(640 부근) 피크 찾기
    l_min, l_max = 640 - search_range, 640 + search_range
    l_min, l_max = max(0, l_min), min(w, l_max)
    peak_l_idx = l_min + np.argmax(col_sums[l_min:l_max])
    peak_l_val = col_sums[peak_l_idx]

    # 3. 두 번째 경계선(1280 부근) 피크 찾기
    r_min, r_max = 1280 - search_range, 1280 + search_range
    r_min, r_max = max(0, r_min), min(w, r_max)
    peak_r_idx = r_min + np.argmax(col_sums[r_min:r_max])
    peak_r_val = col_sums[peak_r_idx]

    # 4. 로컬 참조값 계산 (단위 보정 포함)
    ref_area_w = 20
    ref_val = np.sum(sobel_x[:, 100:100 + ref_area_w]) / ref_area_w
    local_ref = ref_val * threshold_mult

    # 5. [핵심] 간격 조건 검증 (640 +- 10px)
    actual_gap = peak_r_idx - peak_l_idx
    gap_valid = abs(actual_gap - 640) <= 10

    # 6. 최종 판정
    pass_strength = (peak_l_val > local_ref) and (peak_r_val > local_ref)
    is_tripartite = pass_strength and gap_valid

    # 디버깅 정보
    if is_tripartite:
        print(f"🎯 경계선 탐지 성공! 위치: ({peak_l_idx}, {peak_r_idx}), 간격: {actual_gap}px")

    return is_tripartite, peak_l_idx, peak_r_idx


def run_edge_check(image_path, band_px=5, threshold_mult=3.0, use_v3=False, search_range=15):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[오류] 이미지를 불러올 수 없습니다: {image_path}")
        return

    frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = frame_gray.shape

    if use_v3:
        t0 = time.perf_counter()
        is_tripartite, peak_l_idx, peak_r_idx = fast_edge_check_v3(
            frame_gray, search_range=search_range, threshold_mult=threshold_mult
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        actual_gap = peak_r_idx - peak_l_idx

        # 로그 출력 (v3)
        print("-" * 50)
        print(f"입력: {image_path}")
        print(f"해상도: {w} x {h}")
        print(f"엣지 검출(v3) 소요 시간: {elapsed_ms:.2f} ms")
        print(f"경계선 위치: ({peak_l_idx}, {peak_r_idx}), 간격: {actual_gap}px")
        print(f"판정: {'삼분할 영상' if is_tripartite else '일반 영상'}")
        print("-" * 50)

        # 디버그 이미지 (v3): 탐지된 좌표에 수직선
        debug_img = img.copy()
        cv2.line(debug_img, (peak_l_idx, 0), (peak_l_idx, h), (0, 255, 0), 2)
        cv2.line(debug_img, (peak_r_idx, 0), (peak_r_idx, h), (0, 255, 0), 2)
        result_text = "삼분할" if is_tripartite else "일반"
        cv2.putText(debug_img, result_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0) if is_tripartite else (0, 0, 255), 2)
        cv2.putText(debug_img, f"L:{peak_l_idx} R:{peak_r_idx} gap:{actual_gap}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    else:
        t0 = time.perf_counter()
        is_tripartite, edge_l, edge_r, ref, match_ratio = fast_edge_check(
            frame_gray, band_px=band_px, threshold_mult=threshold_mult
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # 로그 출력
        print("-" * 50)
        print(f"입력: {image_path}")
        print(f"해상도: {w} x {h}")
        print(f"엣지 검출 소요 시간: {elapsed_ms:.2f} ms")
        print(f"엣지 강도 (1/3 위치): {edge_l:.0f}")
        print(f"엣지 강도 (2/3 위치): {edge_r:.0f}")
        print(f"기준값 (로컬참조*{threshold_mult}): {ref:.0f}")
        print(f"삼분할 일치율: {match_ratio*100:.0f}%")
        print(f"판정: {'삼분할 영상' if is_tripartite else '일반 영상'}")
        print("-" * 50)

        # 디버그 이미지: 동일 경로에 input_debug.jpg
        debug_img = img.copy()
        slice_w = w // 3
        cv2.line(debug_img, (slice_w, 0), (slice_w, h), (0, 255, 0), 2)
        cv2.line(debug_img, (slice_w * 2, 0), (slice_w * 2, h), (0, 255, 0), 2)
        result_text = "삼분할" if is_tripartite else "일반"
        rate_text = f"일치율: {match_ratio*100:.0f}%"
        cv2.putText(debug_img, result_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0) if is_tripartite else (0, 0, 255), 2)
        cv2.putText(debug_img, rate_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(debug_img, f"L:{edge_l:.0f} R:{edge_r:.0f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    out_dir = os.path.dirname(os.path.abspath(image_path))
    debug_path = os.path.join(out_dir, "input_debug.jpg")
    cv2.imwrite(debug_path, debug_img)
    print(f"디버그 이미지 저장: {debug_path}")


def run_edge_check_v3(image_path, search_range=15, threshold_mult=5.0):
    """v3 전용 실행: 경계선 검출 위치 출력 + _debug_edge.jpg 저장."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"[오류] 이미지를 불러올 수 없습니다: {image_path}")
        return

    frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. 유연한 엣지 체크 실행 (위치 정보 반환) + 소요 시간 측정
    t0 = time.perf_counter()
    is_tri, p_l, p_r = fast_edge_check_v3(frame_gray, search_range, threshold_mult)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # 2. 경계선 검출 위치 및 엣지 검출 시간 출력
    print("-" * 50)
    print(f"엣지 검출(v3) 소요 시간: {elapsed_ms:.2f} ms")
    print(f"경계선 검출 위치: 왼쪽 {p_l}px, 오른쪽 {p_r}px (간격: {p_r - p_l}px)")
    print("-" * 50)

    # 3. 디버그 이미지 생성
    debug_img = img.copy()
    h, w = img.shape[:2]

    # [표시] 검출된 경계선 그리기 (초록색: 성공, 빨간색: 실패/고정위치)
    line_color = (0, 255, 0) if is_tri else (0, 0, 255)

    # 왼쪽 경계선 표시
    cv2.line(debug_img, (p_l, 0), (p_l, h), line_color, 3)
    cv2.putText(debug_img, f"Peak L: {p_l}px", (p_l + 10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, line_color, 2)

    # 오른쪽 경계선 표시
    cv2.line(debug_img, (p_r, 0), (p_r, h), line_color, 3)
    cv2.putText(debug_img, f"Peak R: {p_r}px", (p_r + 10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, line_color, 2)

    # [표시] 간격 정보 출력
    gap = p_r - p_l
    status_text = f"Status: {'OK' if is_tri else 'FAIL'} | Gap: {gap}px"
    cv2.putText(debug_img, status_text, (50, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, line_color, 3)

    # 저장
    out_path = image_path.replace(".jpg", "_debug_edge.jpg")
    cv2.imwrite(out_path, debug_img)
    print(f"📍 경계선 좌표 확인 완료: {p_l}px, {p_r}px (저장: {out_path})")


def main():
    parser = argparse.ArgumentParser(description="삼분할 영상 엣지 검출 및 일치율 판정")
    parser.add_argument("image_path", help="판별할 이미지 파일 경로 (예: image.jpg)")
    parser.add_argument("--band", type=int, default=3, help="엣지 밴드 폭 픽셀 (기본: 3)")
    parser.add_argument("--threshold", type=float, default=5.0, help="로컬 참조 대비 엣지 기준 배수 (기본: 5.0)")
    parser.add_argument("--v3", action="store_true", help="fast_edge_check_v3 사용 (640/1280 피크+간격 검증)")
    parser.add_argument("--run-v3", action="store_true", help="run_edge_check_v3만 실행 (경계선 위치 출력, _debug_edge.jpg 저장)")
    parser.add_argument("--search-range", type=int, default=15, help="v3 사용 시 640/1280 주변 검색 범위 픽셀 (기본: 15)")
    args = parser.parse_args()

    if args.run_v3:
        run_edge_check_v3(
            args.image_path,
            search_range=args.search_range,
            threshold_mult=args.threshold,
        )
    else:
        run_edge_check(
            args.image_path,
            band_px=args.band,
            threshold_mult=args.threshold,
            use_v3=args.v3,
            search_range=args.search_range,
        )


if __name__ == "__main__":
    main()
