"""
삼분할/이분할 영상 엣지 검출 (v3 전용).
- w*1/3, w*2/3 부근 삼분할 검사 후, 불만족 시 화면 절반(이분할) 검사.
사용: python edge_check_ver1.py "image.jpg"
디버그 이미지: 이미지가 있는 폴더 안 debug/ 에 *_debug_edge.jpg 로 저장.
"""

import argparse
import os
import time
import cv2
import numpy as np


def _format_time_sec(time_sec):
    """초 단위를 HH:MM:SS.mmm 문자열로."""
    if time_sec is None:
        return ""
    h = int(time_sec // 3600)
    m = int((time_sec % 3600) // 60)
    s = time_sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _edge_log_prefix(time_sec):
    """엣지 검출 로그용 접두어. time_sec 있으면 [HH:MM:SS.mmm] 없으면 빈 문자열."""
    if time_sec is None:
        return ""
    return f"[{_format_time_sec(time_sec)}] "


def fast_edge_check_v3(
    frame_gray,
    search_range=40,
    threshold_mult=5.0,
    time_sec=None,
    verify_duplicate=True,
    duplicate_similarity_threshold=0.92,
):
    """
    w*1/3, w*2/3 부근 피크 검색 + 간격 조건으로 삼분할 판정.
    verify_duplicate=True이면 좌·중·우 패치 유사도 검증 후 통과 시에만 삼분할 판정.
    :param frame_gray: 그레이스케일 이미지 (H, W)
    :param search_range: 1/3, 2/3 지점 주변 검색 범위(픽셀).
    :param threshold_mult: 로컬 참조 대비 이 배수 이상이면 엣지 있음으로 간주.
    :param time_sec: 영상 시각(초). 주어지면 로그에 시각 출력.
    :param verify_duplicate: True면 좌·중·우 9패치 검증 후 통과할 때만 삼분할 판정.
    :param duplicate_similarity_threshold: 패치 유사도 기준 (0~1). 좌==우 or 중==좌 or 중==우 중 하나 만족 시 통과.
    :return: (is_tripartite, peak_l_idx, peak_r_idx)
    """
    h, w = frame_gray.shape
    # 삼분할 기준: 1/3, 2/3 지점 (1920이면 640, 1280)
    pos_1_3 = w // 3
    pos_2_3 = w * 2 // 3
    expected_gap = w // 3  # 두 경계선 간 기대 간격

    # 1. Sobel 처리 (기존과 동일)
    blurred = cv2.GaussianBlur(frame_gray, (3, 3), 0)
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_x = np.abs(sobel_x)

    # 수직 합(Column Sum) 계산
    col_sums = np.sum(sobel_x, axis=0)

    # 2. 첫 번째 경계선(w*1/3 부근) 피크 찾기
    l_min, l_max = pos_1_3 - search_range, pos_1_3 + search_range
    l_min, l_max = max(0, l_min), min(w, l_max)
    peak_l_idx = l_min + np.argmax(col_sums[l_min:l_max])
    peak_l_val = col_sums[peak_l_idx]

    # 3. 두 번째 경계선(w*2/3 부근) 피크 찾기
    r_min, r_max = pos_2_3 - search_range, pos_2_3 + search_range
    r_min, r_max = max(0, r_min), min(w, r_max)
    peak_r_idx = r_min + np.argmax(col_sums[r_min:r_max])
    peak_r_val = col_sums[peak_r_idx]

    # 4. 로컬 참조값 계산 (단위 보정 포함)
    ref_area_w = 20
    ref_val = np.sum(sobel_x[:, 100:100 + ref_area_w]) / ref_area_w
    local_ref = ref_val * threshold_mult

    # 5. [핵심] 간격 조건 검증 (기대 간격 w/3 ± 허용오차)
    actual_gap = peak_r_idx - peak_l_idx
    gap_tolerance = max(15, expected_gap // 20)
    gap_valid = abs(actual_gap - expected_gap) <= gap_tolerance

    # 6. 엣지 강도 + 간격 통과 시, 선택적으로 좌·중·우 패치 검증
    pass_strength = (peak_l_val > local_ref) and (peak_r_val > local_ref)
    is_tripartite = pass_strength and gap_valid

    if is_tripartite and verify_duplicate:
        is_verified, detail = verify_left_center_right_tripartite(
            frame_gray,
            peak_l_idx,
            peak_r_idx,
            similarity_threshold=duplicate_similarity_threshold,
        )
        is_tripartite = is_verified
        if pass_strength and gap_valid and not is_verified:
            print(f"{_edge_log_prefix(time_sec)}삼분할 미판정: 경계선은 있으나 좌·중·우 동일 영상 아님 (유사도 검증 실패: {detail})")

    # 디버깅 정보
    if is_tripartite:
        print(f"{_edge_log_prefix(time_sec)}삼분할 탐지: 위치 ({peak_l_idx}, {peak_r_idx}), 간격 {actual_gap}px (기대: {expected_gap}px)")

    return is_tripartite, peak_l_idx, peak_r_idx


def _patch_similarity(patch_l, patch_r):
    """
    두 패치의 유사도 [0, 1]. 1에 가까울수록 동일.
    평균 절대 차이 기반: 1 - mean(|L-R|)/255 (그레이스케일 0~255 가정).
    """
    if patch_l.size == 0 or patch_r.size == 0:
        return 0.0
    diff = np.abs(patch_l.astype(np.float64) - patch_r.astype(np.float64))
    mean_diff = np.mean(diff)
    return max(0.0, 1.0 - mean_diff / 255.0)


def verify_left_center_right_tripartite(
    frame_gray,
    peak_l_idx,
    peak_r_idx,
    patch_size=50,
    similarity_threshold=0.92,
):
    """
    삼분할 영상에서 좌·중·우 구역이 동일(유사)한지 패치로 검증.
    화면을 삼등분(좌/중/우) × 상·중·하로 9개 위치에 패치를 두고,
    각 위치에서 (좌==우) or (중==좌) or (중==우) 중 하나만 만족하면 해당 위치 통과.
    9개 위치 중 하나라도 통과하면 삼분할(동일 영상 3연복)로 판단.

    :param frame_gray: 그레이스케일 (H, W)
    :param peak_l_idx: 좌측 경계 x (중앙 구역 시작)
    :param peak_r_idx: 우측 경계 x (우측 구역 시작)
    :param patch_size: 패치 한 변 길이
    :param similarity_threshold: 유사도 기준 (0~1)
    :return: (is_tripartite_verified, detail_str)
    """
    h, w = frame_gray.shape
    w_l = peak_l_idx
    w_c = peak_r_idx - peak_l_idx
    w_r = w - peak_r_idx
    if w_l < 10 or w_c < 10 or w_r < 10:
        return False, "section too narrow"

    # 상·중·하 y 중심, 좌/중/우 구역 내 x 상대 위치 (0.25, 0.5, 0.75)
    y_centers = [h // 4, h // 2, (3 * h) // 4]
    x_ratios = [0.25, 0.5, 0.75]
    ps = min(
        patch_size,
        w_l // 3,
        w_c // 3,
        w_r // 3,
        h // 4,
        50,
    )
    ps = max(10, ps)

    for row, yc in enumerate(y_centers):
        for col, x_ratio in enumerate(x_ratios):
            # 각 구역에서 같은 상대 위치의 패치 중심
            x_l = int(w_l * x_ratio)
            x_c = peak_l_idx + int(w_c * x_ratio)
            x_r = peak_r_idx + int(w_r * x_ratio)
            y0 = max(0, yc - ps // 2)
            y1 = min(h, y0 + ps)

            def extract(cx, x_start, x_end):
                px0 = max(x_start, cx - ps // 2)
                px1 = min(x_end, px0 + ps)
                if px1 <= px0 or px1 - px0 < 5:
                    return None
                return frame_gray[y0:y1, px0:px1]

            pl = extract(x_l, 0, peak_l_idx)
            pc = extract(x_c, peak_l_idx, peak_r_idx)
            pr = extract(x_r, peak_r_idx, w)
            if pl is None or pc is None or pr is None:
                continue
            # 크기 맞춰서 비교 (최소 공통 크기)
            min_h = min(pl.shape[0], pc.shape[0], pr.shape[0])
            min_w = min(pl.shape[1], pc.shape[1], pr.shape[1])
            if min_h < 5 or min_w < 5:
                continue
            pl = pl[:min_h, :min_w]
            pc = pc[:min_h, :min_w]
            pr = pr[:min_h, :min_w]

            sim_lr = _patch_similarity(pl, pr)
            sim_lc = _patch_similarity(pl, pc)
            sim_cr = _patch_similarity(pc, pr)
            # 좌==우 or 중==좌 or 중==우 중 하나만 만족하면 통과
            if sim_lr >= similarity_threshold or sim_lc >= similarity_threshold or sim_cr >= similarity_threshold:
                return True, f"patch row={row} col={col} (L-R={sim_lr:.2f} L-C={sim_lc:.2f} C-R={sim_cr:.2f})"

    return False, "no patch pair passed (L-R or L-C or C-R)"


def verify_left_right_duplicate(
    frame_gray,
    center,
    patch_size=50,
    similarity_threshold=0.92,
    use_half_compare=True,
):
    """
    좌측 절반과 우측 절반이 동일(또는 매우 유사)한 영상인지 검증.
    이분할 판정은 '중앙선 + 좌우 전체 일치'만 허용: 전체 좌반 vs 우반(half_compare) 유사도가
    기준 이상일 때만 통과. 일부 패치만 비슷한 경우(우연한 오탐)는 통과시키지 않음.

    - 필수: 전체 절반 픽셀 비교 (좌반 [0:center] vs 우반 [center:center+half_w]) 유사도 >= threshold
    - 패치(상·중·하)는 실패 시 로그용 best_patch만 계산, 통과 조건으로는 사용하지 않음

    :param frame_gray: 그레이스케일 (H, W)
    :param center: 중앙 x (우반 시작 열 인덱스)
    :param patch_size: 패치 한 변 길이 (실패 시 detail 로그용)
    :param similarity_threshold: 이 값 이상이면 유사로 인정 (0~1)
    :param use_half_compare: True면 전체 좌/우 절반 유사도로 통과 판단
    :return: (is_duplicate, detail_str)
    """
    h, w = frame_gray.shape
    if center <= 0 or center >= w:
        return False, "center out of range"

    half_w = min(center, w - center)  # 비교할 열 개수 (좌반/우반 중 짧은 쪽)
    left = frame_gray[:, 0:half_w]
    right = frame_gray[:, center : center + half_w]

    half_sim_for_detail = 0.0
    # ----- 필수: 전체 좌반 vs 우반 유사도 (이 조건만 통과 시 이분할 인정) -----
    if use_half_compare and left.size > 0 and right.size == left.size:
        half_sim = _patch_similarity(left, right)
        half_sim_for_detail = half_sim
        if half_sim >= similarity_threshold:
            return True, f"half_compare sim={half_sim:.3f}"

    # ----- 패치 유사도는 통과 조건 아님, 실패 시 로그(detail)용 -----
    patch_x_offsets = [center // 4, (3 * center) // 4]
    y_offsets = [h // 4, h // 2, (3 * h) // 4]
    ps = min(patch_size, half_w // 2, h // 2, 20)
    best_sim = 0.0
    if ps >= 10:
        for x_off in patch_x_offsets:
            for y_off in y_offsets:
                y0 = max(0, y_off - ps // 2)
                y1 = min(h, y0 + ps)
                x0_l = max(0, x_off - ps // 2)
                x1_l = min(center, x0_l + ps)
                width = x1_l - x0_l
                x0_r = center + x0_l
                if x0_r + width > w or width <= 0:
                    continue
                pl = frame_gray[y0:y1, x0_l:x1_l]
                pr = frame_gray[y0:y1, x0_r:x0_r + width]
                if pl.size == 0 or pr.size != pl.size:
                    continue
                sim = _patch_similarity(pl, pr)
                best_sim = max(best_sim, sim)

    detail = f"best_patch={best_sim:.3f}, half={half_sim_for_detail:.3f} (필수: half>={similarity_threshold})"
    return False, detail


def fast_edge_check_bipartite(
    frame_gray,
    search_range=40,
    threshold_mult=5.0,
    time_sec=None,
    verify_duplicate=True,
    duplicate_similarity_threshold=0.92,
):
    """
    화면 절반 지점(이분할) 경계선 검출.
    좌측 절반을 우측에 복사·붙여넣기한 영상에서 중앙(960 부근)의 수직 경계선을 찾고,
    verify_duplicate=True이면 좌/우 절반이 실제로 동일(유사)한지 검증해 우연한 중앙선 오탐을 걸러냄.
    :param frame_gray: 그레이스케일 이미지 (H, W)
    :param search_range: 중앙 주변 검색 범위(픽셀).
    :param threshold_mult: 로컬 참조 대비 이 배수 이상이면 엣지 있음으로 간주.
    :param time_sec: 영상 시각(초). 주어지면 로그에 시각 출력 (tripartite_section_check 연동용).
    :param verify_duplicate: True면 좌/우 절반 유사도 검증 후 통과할 때만 이분할로 판정.
    :param duplicate_similarity_threshold: 유사도 기준 (0~1). 이 이상이면 동일 영상 복붙으로 인정.
    :return: (is_bipartite, peak_center_idx)
    """
    h, w = frame_gray.shape
    center = w // 2  # 1920이면 960

    # 0. 좌/우 절반이 충분히 비슷하면 엣지 강도와 무관하게 이분할로 먼저 판정
    #    (엣지 검출은 보조 정보로 사용)
    if verify_duplicate:
        is_dup_first, detail_first = verify_left_right_duplicate(
            frame_gray,
            center=center,
            similarity_threshold=duplicate_similarity_threshold,
        )
        if is_dup_first:
            # 엣지 검출 없이 중앙 index는 이론상 center로 반환
            print(
                f"{_edge_log_prefix(time_sec)}이분할 탐지: 위치 {center}px (중앙 {center} 부근) (검증: {detail_first}, edge check skipped)"
            )
            return True, center

    blurred = cv2.GaussianBlur(frame_gray, (3, 3), 0)
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_x = np.abs(sobel_x)
    col_sums = np.sum(sobel_x, axis=0)

    # 중앙 부근 피크 찾기
    c_min, c_max = center - search_range, center + search_range
    c_min, c_max = max(0, c_min), min(w, c_max)
    peak_idx = c_min + np.argmax(col_sums[c_min:c_max])
    peak_val = col_sums[peak_idx]

    ref_area_w = 20
    ref_val = np.sum(sobel_x[:, 100:100 + ref_area_w]) / ref_area_w
    local_ref = ref_val * threshold_mult

    has_center_edge = peak_val > local_ref
    is_bipartite = has_center_edge
    detail_success = None  # 검증 통과 시 원인 분석용 (half_compare / patch row=N)

    if has_center_edge and verify_duplicate:
        is_dup, detail = verify_left_right_duplicate(
            frame_gray,
            center=peak_idx,
            similarity_threshold=duplicate_similarity_threshold,
        )
        is_bipartite = is_dup
        if has_center_edge and not is_dup:
            print(f"{_edge_log_prefix(time_sec)}이분할 미판정: 중앙 경계선은 있으나 좌/우 동일 영상 아님 (유사도 검증 실패: {detail})")
        else:
            detail_success = detail

    if is_bipartite:
        msg = f"{_edge_log_prefix(time_sec)}이분할 탐지: 위치 {peak_idx}px (중앙 {center} 부근)"
        if detail_success is not None:
            msg += f" (검증: {detail_success})"
        print(msg)

    return is_bipartite, peak_idx


def resize_with_preset(img, preset: str):
    """
    FHD(1920x1080) 기준 테스트용 다운스케일 프리셋.
    - "orig" : 원본 그대로
    - "960"  : 가로 960, 세로는 종횡비 유지 (예: 1920x1080 → 960x540)
    - "640"  : 가로 640, 세로는 종횡비 유지 (예: 1920x1080 → 640x360)
    - "480"  : 가로 480, 세로는 종횡비 유지 (예: 1920x1080 → 480x270)
    """
    if preset == "orig":
        return img

    try:
        target_width = int(preset)
    except ValueError:
        print(f"[경고] 지원하지 않는 프리셋입니다: {preset} (orig, 960, 640, 480 중 선택)")
        return img

    h, w = img.shape[:2]
    if w == 0 or h == 0:
        return img

    scale = target_width / w
    target_height = max(1, int(h * scale))

    return cv2.resize(img, (target_width, target_height))


def run_edge_check(
    image_path,
    search_range=40,
    threshold_mult=5.0,
    preset="orig",
    verify_tripartite_duplicate=True,
    verify_bipartite_duplicate=True,
    duplicate_similarity_threshold=0.92,
):
    """삼분할(w/3, 2w/3) → 불만족 시 이분할(중앙) 검사. 삼/이분할 모두 좌·중·우(또는 좌·우) 동일 영상 검증 통과 시에만 판정. 경계선 위치 출력 + debug/*_debug_edge.jpg 저장."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"[오류] 이미지를 불러올 수 없습니다: {image_path}")
        return

    # FHD(1920x1080) 기준 프리셋 다운스케일 적용 (종횡비 유지)
    img = resize_with_preset(img, preset)

    frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 다운스케일 시 리사이즈 보간으로 경계가 흐려져 엣지 강도가 약해짐 → 기준 완화
    effective_threshold = threshold_mult
    if preset == "960":
        effective_threshold = min(threshold_mult, 4.5)
    elif preset == "640":
        effective_threshold = min(threshold_mult, 4.0)
    elif preset == "480":
        effective_threshold = min(threshold_mult, 3.5)

    # 1. 삼분할(640/1280) 엣지 체크 + 선택적 좌·중·우 패치 검증
    t0 = time.perf_counter()
    is_tri, p_l, p_r = fast_edge_check_v3(
        frame_gray,
        search_range,
        effective_threshold,
        verify_duplicate=verify_tripartite_duplicate,
        duplicate_similarity_threshold=duplicate_similarity_threshold,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # 2. 삼분할 불만족 시 → 화면 절반(이분할) 경계선 체크
    is_bi, p_center = False, 0
    if not is_tri:
        t_bi = time.perf_counter()
        is_bi, p_center = fast_edge_check_bipartite(
            frame_gray,
            search_range,
            effective_threshold,
            verify_duplicate=verify_bipartite_duplicate,
            duplicate_similarity_threshold=duplicate_similarity_threshold,
        )
        elapsed_ms += (time.perf_counter() - t_bi) * 1000

    # 3. 경계선 검출 위치 및 엣지 검출 시간 출력
    print("-" * 50)
    print(f"엣지 검출(v3) 소요 시간: {elapsed_ms:.2f} ms")
    if is_tri:
        print(f"경계선 검출 위치: 왼쪽 {p_l}px, 오른쪽 {p_r}px (간격: {p_r - p_l}px)")
        print(f"판정: 삼분할 영상")
    elif is_bi:
        print(f"경계선 검출 위치: 중앙 {p_center}px (이분할)")
        print(f"판정: 이분할 영상 (좌측 복사·우측 붙여넣기)")
    else:
        print(f"경계선 검출 위치: 왼쪽 {p_l}px, 오른쪽 {p_r}px (간격: {p_r - p_l}px)")
        print(f"판정: 일반 영상")
    print("-" * 50)

    # 4. 디버그 이미지 생성
    debug_img = img.copy()
    h, w = img.shape[:2]

    if is_tri:
        line_color = (0, 255, 0)
        cv2.line(debug_img, (p_l, 0), (p_l, h), line_color, 3)
        cv2.putText(debug_img, f"Peak L: {p_l}px", (p_l + 10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, line_color, 2)
        cv2.line(debug_img, (p_r, 0), (p_r, h), line_color, 3)
        cv2.putText(debug_img, f"Peak R: {p_r}px", (p_r + 10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, line_color, 2)
        status_text = f"Status: OK (3-split) | Gap: {p_r - p_l}px"
    elif is_bi:
        line_color = (0, 255, 0)
        cv2.line(debug_img, (p_center, 0), (p_center, h), line_color, 3)
        cv2.putText(debug_img, f"Center: {p_center}px", (p_center + 10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, line_color, 2)
        status_text = f"Status: OK (2-split) | Center: {p_center}px"
    else:
        line_color = (0, 0, 255)
        cv2.line(debug_img, (p_l, 0), (p_l, h), line_color, 3)
        cv2.putText(debug_img, f"Peak L: {p_l}px", (p_l + 10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, line_color, 2)
        cv2.line(debug_img, (p_r, 0), (p_r, h), line_color, 3)
        cv2.putText(debug_img, f"Peak R: {p_r}px", (p_r + 10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, line_color, 2)
        status_text = f"Status: FAIL | Gap: {p_r - p_l}px"

    cv2.putText(debug_img, status_text, (50, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, line_color, 3)

    # 저장: 이미지가 있는 폴더 안 debug/ 에 *_debug_edge.jpg 로 저장. 중복 시 _debug_edge_(1).jpg, _(2).jpg ... 넘버링
    img_dir = os.path.dirname(os.path.abspath(image_path))
    debug_dir = os.path.join(img_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    suffix_stem = f"_{preset}_debug_edge" if preset != "orig" else "_debug_edge"
    out_path = os.path.join(debug_dir, base_name + suffix_stem + ".jpg")
    n = 0
    while os.path.exists(out_path):
        n += 1
        out_path = os.path.join(debug_dir, base_name + suffix_stem + f"_({n}).jpg")
    cv2.imwrite(out_path, debug_img)
    if is_tri:
        print(f"경계선 좌표 확인 완료: {p_l}px, {p_r}px (저장: {out_path})")
    elif is_bi:
        print(f"이분할 경계선 좌표 확인 완료: {p_center}px (저장: {out_path})")
    else:
        print(f"경계선 미충족 (L:{p_l}, R:{p_r}) (저장: {out_path})")


def main():
    parser = argparse.ArgumentParser(description="삼분할/이분할 영상 엣지 검출 (v3)")
    parser.add_argument("image_path", help="판별할 이미지 파일 경로 (예: image.jpg)")
    parser.add_argument("--threshold", type=float, default=5.0, help="로컬 참조 대비 엣지 기준 배수 (기본: 5.0)")
    parser.add_argument("--search-range", type=int, default=40, help="1/3·2/3 지점 주변 검색 범위 픽셀 (기본: 40, FHD에서 666/1311 수준까지 검사)")
    parser.add_argument(
        "--preset",
        choices=["orig", "960", "640", "480"],
        default="orig",
        help="입력 FHD 영상을 다운스케일할 가로 해상도 프리셋 (orig, 960, 640, 480)",
    )
    parser.add_argument(
        "--no-verify-tripartite",
        action="store_true",
        help="삼분할 시 좌·중·우 패치 유사도 검증 생략",
    )
    parser.add_argument(
        "--no-verify-duplicate",
        action="store_true",
        help="이분할 시 좌/우 동일 영상 유사도 검증 생략 (중앙선만 보면 이분할로 판정)",
    )
    parser.add_argument(
        "--duplicate-threshold",
        type=float,
        default=0.92,
        help="삼/이분할 패치 유사도 기준 (0~1). 좌==우 or 중==좌 or 중==우 등 하나만 만족해도 통과 (기본: 0.92)",
    )
    args = parser.parse_args()

    run_edge_check(
        args.image_path,
        search_range=args.search_range,
        threshold_mult=args.threshold,
        preset=args.preset,
        verify_tripartite_duplicate=not args.no_verify_tripartite,
        verify_bipartite_duplicate=not args.no_verify_duplicate,
        duplicate_similarity_threshold=args.duplicate_threshold,
    )


if __name__ == "__main__":
    main()
