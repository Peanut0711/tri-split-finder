import os
import cv2
import numpy as np
import random

def _imread_unicode(path):
    """Windows에서 한글 등 비ASCII 경로를 안전하게 로드 (OpenCV imread는 Windows에서 유니코드 경로 미지원)."""
    with open(path, "rb") as f:
        buf = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def _imwrite_unicode(path, img):
    """Windows에서 한글 등 비ASCII 경로로 이미지 저장 (OpenCV imwrite는 Windows에서 유니코드 경로 미지원)."""
    ext = os.path.splitext(path)[1].lower() or ".png"
    success, buf = cv2.imencode(ext, img)
    if not success:
        return False
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    return True


def test_tripartite_image(
    image_path,
    num_patches=30,
    patch_size=64,
    score_threshold=0.70,
    pass_ratio=0.4,
    position_tol=7,
    exclude_bottom_ratio=0.22,
    exclude_top_ratio=0.12,
    exclude_h_margin_ratio=0.08,
    min_patch_variance=80,
):
    """
    삼분할 판별 (양쪽 UI가 다른 경우 적합: 왼쪽 하단 채팅, 우측 상단 팝업 등).
    - 패치: 전체 영역(좌/중/우 슬라이스)에서 추출, 같은 슬라이스 내에서는 겹치지 않게 선택.
    - 좌/우 일치율 각각 pass_ratio 이상이면 삼분할 인정.
    """
    # 1. 이미지 로드 (한글 경로 지원: Windows에서 cv2.imread는 유니코드 경로 미지원)
    img = _imread_unicode(image_path)
    if img is None:
        print(f"Error: '{image_path}' 이미지를 불러올 수 없습니다. 경로를 확인해주세요.")
        return

    h, w = img.shape[:2]
    slice_w = w // 3  # 정확히 3등분 (1920 기준 640)

    # 연산을 위한 그레이스케일 변환 및 3등분
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    left_img = gray[:, 0:slice_w]
    center_img = gray[:, slice_w:slice_w*2]
    right_img = gray[:, slice_w*2:w]

    # 디버깅(시각화)을 위해 원본 복사
    debug_img = img.copy()
    
    left_valid_count = 0
    right_valid_count = 0
    left_checks = 0
    right_checks = 0
    margin = patch_size // 2
    tol = position_tol
    slices_gray = [left_img, center_img, right_img]

    # UI 회피: 상단·하단·가장자리 제외한 "안전 영역" (각 슬라이스 공통, 로컬 좌표)
    y_min = margin + int(h * exclude_top_ratio)
    y_max = h - margin - 1 - int(h * exclude_bottom_ratio)
    x_margin = max(margin, int(slice_w * exclude_h_margin_ratio))
    x_min, x_max = x_margin, slice_w - x_margin - 1 - margin
    if y_max <= y_min or x_max <= x_min:
        y_min, y_max = margin, h - margin - 1
        x_min, x_max = margin, slice_w - margin - 1

    def overlaps_any(cx, cy, used):
        for (ux, uy) in used:
            if abs(cx - ux) < patch_size and abs(cy - uy) < patch_size:
                return True
        return False

    used_centers = [[], [], []]  # 슬라이스별 이미 쓴 패치 중심 (로컬 cx,cy)
    tried = 0
    max_tries = num_patches * 8
    drawn = 0

    print(f"[{image_path}] 분석 시작... (해상도: {w}x{h}, 전체 영역·슬라이스 내 비겹침)")

    while drawn < num_patches and tried < max_tries:
        tried += 1
        # 전체 영역: 좌(0)/중(1)/우(2) 중 랜덤 선택
        sid = random.randint(0, 2)
        cx = random.randint(x_min, x_max) if x_max >= x_min else random.randint(margin, slice_w - margin - 1)
        cy = random.randint(y_min, y_max) if y_max >= y_min else random.randint(margin, h - margin - 1)

        if overlaps_any(cx, cy, used_centers[sid]):
            continue

        patch = slices_gray[sid][cy-margin:cy+margin, cx-margin:cx+margin]
        if patch.size == 0 or patch.shape[0] != patch_size or patch.shape[1] != patch_size:
            continue
        if min_patch_variance > 0 and np.var(patch) < min_patch_variance:
            continue

        used_centers[sid].append((cx, cy))
        drawn += 1
        expected_x = cx - margin
        expected_y = cy - margin

        def match_at(slice_idx, px, py):
            src = slices_gray[slice_idx]
            res = cv2.matchTemplate(src, patch, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            ok = (max_val >= score_threshold and
                  abs(max_loc[0] - px) <= tol and abs(max_loc[1] - py) <= tol)
            return ok, max_loc

        # 출처가 아닌 두 슬라이스에서 같은 위치(cx,cy)로 매칭
        left_matched, right_matched = False, False
        loc_left, loc_right = (0, 0), (0, 0)

        if sid == 0:
            # 패치 출처: 좌 → 중·우에서 검사 (좌측 검사 없음)
            c_ok, loc_c = match_at(1, expected_x, expected_y)
            r_ok, loc_r = match_at(2, expected_x, expected_y)
            right_checks += 1
            if r_ok:
                right_valid_count += 1
            right_matched = r_ok
            loc_right = loc_r
            # 중앙도 시각화용
            center_matched = c_ok
            loc_center = loc_c
        elif sid == 1:
            # 패치 출처: 중 → 좌·우에서 검사
            l_ok, loc_l = match_at(0, expected_x, expected_y)
            r_ok, loc_r = match_at(2, expected_x, expected_y)
            left_checks += 1
            right_checks += 1
            if l_ok:
                left_valid_count += 1
            if r_ok:
                right_valid_count += 1
            left_matched, right_matched = l_ok, r_ok
            loc_left, loc_right = loc_l, loc_r
            center_matched, loc_center = True, (expected_x, expected_y)
        else:
            # 패치 출처: 우 → 좌·중에서 검사 (우측 검사 없음)
            l_ok, loc_l = match_at(0, expected_x, expected_y)
            c_ok, loc_c = match_at(1, expected_x, expected_y)
            left_checks += 1
            if l_ok:
                left_valid_count += 1
            left_matched = l_ok
            loc_left = loc_l
            center_matched = c_ok
            loc_center = loc_c

        # 시각화: 출처 슬라이스 파란색, 비교한 슬라이스는 매칭 시 녹색 / 실패 빨간색
        def draw_slice(slice_idx, lx, ly, color):
            gx = slice_w * slice_idx + lx
            cv2.rectangle(debug_img, (gx, ly), (gx + patch_size, ly + patch_size), color, 2)

        draw_slice(sid, cx - margin, cy - margin, (255, 0, 0))
        if sid == 0:
            draw_slice(1, loc_center[0], loc_center[1], (0, 255, 0) if center_matched else (0, 0, 255))
            draw_slice(2, loc_right[0], loc_right[1], (0, 255, 0) if right_matched else (0, 0, 255))
        elif sid == 1:
            draw_slice(0, loc_left[0], loc_left[1], (0, 255, 0) if left_matched else (0, 0, 255))
            draw_slice(2, loc_right[0], loc_right[1], (0, 255, 0) if right_matched else (0, 0, 255))
        else:
            draw_slice(0, loc_left[0], loc_left[1], (0, 255, 0) if left_matched else (0, 0, 255))
            draw_slice(1, loc_center[0], loc_center[1], (0, 255, 0) if center_matched else (0, 0, 255))

    # 최종 결과: 좌/우 검사 횟수가 0이면 1로 나누어 0% 처리
    actual_patches = max(1, drawn)
    left_success_rate = left_valid_count / max(1, left_checks)
    right_success_rate = right_valid_count / max(1, right_checks)
    is_tripartite = (left_success_rate >= pass_ratio) and (right_success_rate >= pass_ratio)

    print("-" * 30)
    print(f"총 추출 패치: {actual_patches}개 (전체 영역, 슬라이스별 비겹침)")
    print(f"좌측 일치: {left_valid_count}/{max(1, left_checks)} ({left_success_rate*100:.1f}%)")
    print(f"우측 일치: {right_valid_count}/{max(1, right_checks)} ({right_success_rate*100:.1f}%)")
    print(f"통과 기준: 각 {pass_ratio*100:.0f}% 이상")
    print(f"최종 판별: {'[OK] 삼분할 영상입니다.' if is_tripartite else '[X] 일반 영상입니다.'}")
    print("-" * 30)

    # 스캔 결과를 참고 이미지와 동일 경로에 저장 (파일명_tripartite_result.확장자)
    base_dir = os.path.dirname(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    ext = os.path.splitext(image_path)[1].lower() or ".jpg"
    result_path = os.path.join(base_dir, f"{base_name}_tripartite_result{ext}")
    if _imwrite_unicode(result_path, debug_img):
        print(f"결과 이미지 저장: {result_path}")
    else:
        print(f"결과 이미지 저장 실패: {result_path}")

    # 결과 이미지 화면 출력
    cv2.imshow("Tripartite Test (Blue: source, Green: match, Red: fail, L/R separate)", debug_img)
    print("아무 키나 누르면 창이 닫힙니다.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --- 실행 부분 ---
# 테스트하실 이미지의 실제 경로로 수정해주세요.
# 한글 경로: raw 문자열(r"...") 또는 슬래시 사용. OpenCV는 Windows에서 유니코드 경로를 _imread_unicode로 처리.
test_image_path = r"image\[2025-02-14] 고라니율 해피발렌타인~! 1080p60 (2).ts_20260312_001846.802.jpg"  # 업로드 하신 이미지 파일명 또는 경로

# 함수 실행 (UI 있는 삼분할은 기본값으로 일치도 향상 적용됨)
test_tripartite_image(
    image_path=test_image_path,
    num_patches=30,               # 패치 개수 (많을수록 안정, 기본 30)
    patch_size=64,
    score_threshold=0.70,         # 유사도 임계 (0.70: UI/프레임차이 완화)
    pass_ratio=0.4,
    position_tol=7,               # 위치 오차 허용 픽셀 (5→7 완화)
    exclude_bottom_ratio=0.22,    # 하단 22% 제외 (왼쪽 채팅창)
    exclude_top_ratio=0.12,       # 상단 12% 제외 (우측 팝업)
    exclude_h_margin_ratio=0.08,  # 중앙 슬라이스 좌우 8% 제외
    min_patch_variance=80,        # 평탄 패치(문자/단색) 제외, 0이면 비활성
)