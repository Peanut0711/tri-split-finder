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


def run_edge_check(image_path, band_px=5, threshold_mult=3.0):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[오류] 이미지를 불러올 수 없습니다: {image_path}")
        return

    frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = frame_gray.shape

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
    # 수직선 그리기 (1/3, 2/3)
    cv2.line(debug_img, (slice_w, 0), (slice_w, h), (0, 255, 0), 2)
    cv2.line(debug_img, (slice_w * 2, 0), (slice_w * 2, h), (0, 255, 0), 2)
    # 결과 텍스트
    result_text = "삼분할" if is_tripartite else "일반"
    rate_text = f"일치율: {match_ratio*100:.0f}%"
    cv2.putText(debug_img, result_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0) if is_tripartite else (0, 0, 255), 2)
    cv2.putText(debug_img, rate_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(debug_img, f"L:{edge_l:.0f} R:{edge_r:.0f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    out_dir = os.path.dirname(os.path.abspath(image_path))
    debug_path = os.path.join(out_dir, "input_debug.jpg")
    cv2.imwrite(debug_path, debug_img)
    print(f"디버그 이미지 저장: {debug_path}")


def main():
    parser = argparse.ArgumentParser(description="삼분할 영상 엣지 검출 및 일치율 판정")
    parser.add_argument("image_path", help="판별할 이미지 파일 경로 (예: image.jpg)")
    parser.add_argument("--band", type=int, default=3, help="엣지 밴드 폭 픽셀 (기본: 3)")
    parser.add_argument("--threshold", type=float, default=5.0, help="로컬 참조 대비 엣지 기준 배수 (기본: 5.0)")
    args = parser.parse_args()

    run_edge_check(args.image_path, band_px=args.band, threshold_mult=args.threshold)


if __name__ == "__main__":
    main()
