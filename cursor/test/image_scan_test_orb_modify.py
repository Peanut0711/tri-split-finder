import cv2
import numpy as np


def _to_edge_silhouette(img, blur_ksize=5, low=50, high=150):
    """실루엣(엣지) 강조: 채팅/팝업 텍스트보다 인물·배경 윤곽이 두드러지게."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    edges = cv2.Canny(blurred, low, high)
    return edges


def _overlay_mask(shape, left_exclude_ratio=0.35, right_exclude_ratio=0.25):
    """
    좌측: 왼쪽 일부(채팅), 우측: 오른쪽 상단(팝업) 제외용 마스크.
    반환: (left_mask, right_mask) 각각 (H,W) uint8, 사용할 영역=255.
    """
    h, w = shape[:2]
    # 좌측 슬라이스: 왼쪽 left_exclude_ratio 비율 제외 (채팅)
    left_mask = np.ones((h, w), dtype=np.uint8) * 255
    ex = int(w * left_exclude_ratio)
    left_mask[:, :ex] = 0
    # 우측 슬라이스: 오른쪽 + 상단 일부 제외 (팝업)
    right_mask = np.ones((h, w), dtype=np.uint8) * 255
    ex_w = int(w * right_exclude_ratio)
    ex_h = int(h * 0.2)
    right_mask[:ex_h, -ex_w:] = 0
    right_mask[:, -ex_w:] = 0
    return left_mask, right_mask


def test_tripartite_orb(
    image_path,
    tol=15,
    min_matches=30,
    use_silhouette=False,
    use_overlay_mask=False,
    relaxed_criterion=False,
    min_matches_any=20,
    min_matches_sum=50,
    min_weak_side=5,
):
    """
    ORB 특징점 매칭을 이용해 삼분할 영상 여부를 판별합니다.

    오버레이 무시 옵션:
    - use_silhouette: 엣지(실루엣) 이미지에서 ORB → 채팅/팝업 위치와 무관하게 실루엣이 매칭에 기여 (위치 변동 시 추천)
    - use_overlay_mask: 좌/우 고정 비율 영역 마스크 (채팅·팝업 위치가 랜덤이면 비추천)
    - relaxed_criterion: 한쪽이 낮아도 삼분할 인정 가능. min_weak_side로 약한 쪽 최소 개수 요구 (한쪽만 2개 같은 건 방지)
    """
    # 1. 이미지 로드 및 3분할
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: '{image_path}' 이미지를 불러올 수 없습니다.")
        return

    h, w = img.shape[:2]
    slice_w = w // 3  # 1920 기준 640

    left_img = img[:, :slice_w]
    center_img = img[:, slice_w:slice_w*2]
    right_img = img[:, slice_w*2:]

    # 실루엣 모드: 엣지 이미지에서 특징점 추출 (오버레이 텍스트보다 인물·배경 실루엣이 지배적)
    if use_silhouette:
        left_img = _to_edge_silhouette(left_img)
        center_img = _to_edge_silhouette(center_img)
        right_img = _to_edge_silhouette(right_img)

    # 2. ORB 초기화 (특징점을 넉넉하게 1000개 추출)
    orb = cv2.ORB_create(nfeatures=1000)

    # 3. 각 영역에서 특징점 추출 (오버레이 마스크 사용 시 해당 구역 제외)
    mask_left, mask_right = None, None
    if use_overlay_mask:
        sh = left_img.shape[:2] if left_img.ndim >= 2 else (h, slice_w)
        mask_left, mask_right = _overlay_mask((sh[0], sh[1]))
    left_src = left_img if left_img.ndim == 2 else cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    center_src = center_img if center_img.ndim == 2 else cv2.cvtColor(center_img, cv2.COLOR_BGR2GRAY)
    right_src = right_img if right_img.ndim == 2 else cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    kp_l, des_l = orb.detectAndCompute(left_src, mask_left)
    kp_c, des_c = orb.detectAndCompute(center_src, None)
    kp_r, des_r = orb.detectAndCompute(right_src, mask_right)

    if des_c is None:
        print("중앙 영상에서 특징점을 찾을 수 없습니다.")
        return

    # 4. 특징점 매칭 (해밍 거리 사용, 상호 일치하는 것만 허용)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches_left = bf.match(des_c, des_l) if des_l is not None else []
    matches_right = bf.match(des_c, des_r) if des_r is not None else []

    # 5. 기하학적 필터링 (가장 중요한 부분)
    # 특징점이 단순히 비슷하게 생긴 것을 넘어, 좌/우 영상의 '동일한 높이와 비율 위치'에 있는지 검사
    valid_left_matches = []
    for m in matches_left:
        pt_c = kp_c[m.queryIdx].pt
        pt_l = kp_l[m.trainIdx].pt
        
        # 중앙 이미지 내의 로컬 좌표(pt_c)와 좌측 이미지 내의 로컬 좌표(pt_l)가 오차(tol) 내에 있는지 확인
        if abs(pt_c[0] - pt_l[0]) <= tol and abs(pt_c[1] - pt_l[1]) <= tol:
            valid_left_matches.append(m)

    valid_right_matches = []
    for m in matches_right:
        pt_c = kp_c[m.queryIdx].pt
        pt_r = kp_r[m.trainIdx].pt
        if abs(pt_c[0] - pt_r[0]) <= tol and abs(pt_c[1] - pt_r[1]) <= tol:
            valid_right_matches.append(m)

    # 6. 결과 출력 및 판별
    n_left, n_right = len(valid_left_matches), len(valid_right_matches)
    if relaxed_criterion:
        # 한쪽이 오버레이로 막혀도 삼분할 인정 가능하되, 약한 쪽도 min_weak_side 이상이어야 함 (한쪽만 2개 같은 과도한 완화 방지)
        is_tripartite = (
            (n_left >= min_matches_any or n_right >= min_matches_any)
            and (n_left + n_right >= min_matches_sum)
            and min(n_left, n_right) >= min_weak_side
        )
    else:
        is_tripartite = n_left >= min_matches and n_right >= min_matches

    print("-" * 40)
    if use_silhouette or use_overlay_mask or relaxed_criterion:
        modes = []
        if use_silhouette:
            modes.append("실루엣(엣지)")
        if use_overlay_mask:
            modes.append("오버레이마스크")
        if relaxed_criterion:
            modes.append(f"기준완화(약한쪽>={min_weak_side})")
        print(f"[오버레이 무시 모드: {', '.join(modes)}]")
    print(f"중앙 영역 추출 특징점: {len(kp_c)}개")
    print(f"좌측 일치 특징점: {n_left}개 (통과 기준: {min_matches}개)")
    print(f"우측 일치 특징점: {n_right}개 (통과 기준: {min_matches}개)")
    if relaxed_criterion:
        print(f"  → 약한 쪽 최소: {min(n_left, n_right)}개 (요구: {min_weak_side}개)")
    print(f"최종 판별: {'[OK] 삼분할 영상입니다.' if is_tripartite else '[X] 일반 영상입니다.'}")
    print("-" * 40)

    # 7. 디버그용 이미지 생성 (원본 이미지 위에 선 그리기)
    debug_img = img.copy()

    # 중앙 -> 좌측 유효 매칭점 그리기 (녹색 선)
    for m in valid_left_matches:
        # 전체 화면 좌표계로 변환
        pt_c_global = (int(kp_c[m.queryIdx].pt[0] + slice_w), int(kp_c[m.queryIdx].pt[1]))
        pt_l_global = (int(kp_l[m.trainIdx].pt[0]), int(kp_l[m.trainIdx].pt[1]))
        
        cv2.circle(debug_img, pt_c_global, 4, (0, 255, 0), -1)
        cv2.circle(debug_img, pt_l_global, 4, (0, 255, 0), -1)
        cv2.line(debug_img, pt_c_global, pt_l_global, (0, 255, 0), 1)

    # 중앙 -> 우측 유효 매칭점 그리기 (파란색 선)
    for m in valid_right_matches:
        pt_c_global = (int(kp_c[m.queryIdx].pt[0] + slice_w), int(kp_c[m.queryIdx].pt[1]))
        pt_r_global = (int(kp_r[m.trainIdx].pt[0] + slice_w * 2), int(kp_r[m.trainIdx].pt[1]))
        
        cv2.circle(debug_img, pt_c_global, 4, (255, 150, 0), -1) # 파란/주황색 계열
        cv2.circle(debug_img, pt_r_global, 4, (255, 150, 0), -1)
        cv2.line(debug_img, pt_c_global, pt_r_global, (255, 150, 0), 1)

    # 화면에 결과 띄우기
    cv2.imshow("ORB Match Result", debug_img)
    print("아무 키나 누르면 창이 닫힙니다.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --- 실행 부분 ---
test_image_path = r"image\input_5.jpg"

# 1) 기본 실행 (오버레이 무시 없음)
# test_tripartite_orb(
#     image_path=test_image_path,
#     tol=15,
#     min_matches=30,
# )

# 2) 채팅/팝업 위치가 변동(좌 40~90%, 우 랜덤)일 때: 실루엣만 사용, 고정 마스크 비권장
#    - use_silhouette: 위치와 무관하게 실루엣으로 매칭 → ORB 그대로 사용 가능
#    - min_weak_side: 좌 2개·우 372개처럼 한쪽만 거의 0이면 삼분할로 안 함 (과도한 완화 방지)
test_tripartite_orb(
    image_path=test_image_path,
    tol=15,
    min_matches=30,
    use_silhouette=True,       # 실루엣 기반 (오버레이 위치 변동해도 괜찮음)
    use_overlay_mask=False,    # 위치 랜덤이면 고정 마스크 의미 없음
    relaxed_criterion=True,
    min_matches_any=20,
    min_matches_sum=50,
    min_weak_side=5,           # 약한 쪽도 5개 이상 있어야 삼분할 인정
)