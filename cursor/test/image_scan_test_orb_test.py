import cv2
import numpy as np

def test_tripartite_orb_v2(image_path, tol=15, min_matches=30):
    """
    ORB 특징점 매칭을 이용해 삼분할 영상 여부를 판별합니다. (v2: ROI, nfeatures 5000, 비대칭 판별)
    
    :param image_path: 테스트할 이미지 경로
    :param tol: 좌우 매칭 시 허용할 픽셀 위치 오차 (채팅/팝업 밀림 방지)
    :param min_matches: 좌/우 각각 최소 몇 개의 특징점이 일치해야 통과시킬지 기준
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: '{image_path}' 이미지를 불러올 수 없습니다.")
        return

    h, w = img.shape[:2]
    slice_w = w // 3

    # [개선 1] 채팅창 방지를 위해 상단 60% 영역만 사용 (ROI 설정)
    roi_h = int(h * 0.6)
    left_img = img[:roi_h, :slice_w]
    center_img = img[:roi_h, slice_w:slice_w*2]
    right_img = img[:roi_h, slice_w*2:]

    # [개선 2] 특징점 개수 5000개로 상향 (9800X3D 성능 활용)
    orb = cv2.ORB_create(nfeatures=5000)

    kp_l, des_l = orb.detectAndCompute(left_img, None)
    kp_c, des_c = orb.detectAndCompute(center_img, None)
    kp_r, des_r = orb.detectAndCompute(right_img, None)

    if des_c is None:
        print("중앙 영상에서 특징점을 찾을 수 없습니다.")
        return

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches_l = bf.match(des_c, des_l) if des_l is not None else []
    matches_r = bf.match(des_c, des_r) if des_r is not None else []

    # 기하학적 필터링 (동일 로직)
    valid_l = [m for m in matches_l if abs(kp_c[m.queryIdx].pt[0]-kp_l[m.trainIdx].pt[0]) <= tol and abs(kp_c[m.queryIdx].pt[1]-kp_l[m.trainIdx].pt[1]) <= tol]
    valid_r = [m for m in matches_r if abs(kp_c[m.queryIdx].pt[0]-kp_r[m.trainIdx].pt[0]) <= tol and abs(kp_c[m.queryIdx].pt[1]-kp_r[m.trainIdx].pt[1]) <= tol]

    cnt_l, cnt_r = len(valid_l), len(valid_r)

    # [개선 3] 비대칭 판별 로직
    # 한쪽이 매우 확실(예: 100개 이상)하다면 다른 쪽은 기준을 낮춰줌 (합산 130개 이상 등)
    is_tripartite = (cnt_l >= min_matches and cnt_r >= min_matches) or \
                    (cnt_l + cnt_r >= 150 and (cnt_l > 5 or cnt_r > 5))

    print("-" * 40)
    print(f"결과 - 좌: {cnt_l} / 우: {cnt_r} | 판별: {is_tripartite}")
    print(f"최종: {'✅ 삼분할 영상입니다.' if is_tripartite else '❌ 일반 영상입니다.'}")
    print("-" * 40)

    # 디버그용 이미지 생성 (원본 이미지 위에 선 그리기)
    debug_img = img.copy()

    # 중앙 -> 좌측 유효 매칭점 그리기 (녹색 선)
    for m in valid_l:
        # 전체 화면 좌표계로 변환
        pt_c_global = (int(kp_c[m.queryIdx].pt[0] + slice_w), int(kp_c[m.queryIdx].pt[1]))
        pt_l_global = (int(kp_l[m.trainIdx].pt[0]), int(kp_l[m.trainIdx].pt[1]))
        
        cv2.circle(debug_img, pt_c_global, 4, (0, 255, 0), -1)
        cv2.circle(debug_img, pt_l_global, 4, (0, 255, 0), -1)
        cv2.line(debug_img, pt_c_global, pt_l_global, (0, 255, 0), 1)

    # 중앙 -> 우측 유효 매칭점 그리기 (파란색 선)
    for m in valid_r:
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
test_image_path = r"image\input_4.jpg"

# 실행
test_tripartite_orb_v2(
    image_path=test_image_path,
    tol=15,           # 픽셀 오차 허용치 (영상 편집 시 미세하게 틀어짐 감안)
    min_matches=30    # 좌/우 각각 최소 30개의 점이 일치하면 삼분할로 인정
)