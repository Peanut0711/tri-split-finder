import cv2
import numpy as np

def test_tripartite_orb(image_path, tol=15, min_matches=30):
    """
    ORB 특징점 매칭을 이용해 삼분할 영상 여부를 판별합니다.
    
    :param image_path: 테스트할 이미지 경로
    :param tol: 좌우 매칭 시 허용할 픽셀 위치 오차 (채팅/팝업 밀림 방지)
    :param min_matches: 좌/우 각각 최소 몇 개의 특징점이 일치해야 통과시킬지 기준
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

    # 2. ORB 초기화 (특징점을 넉넉하게 1000개 추출)
    orb = cv2.ORB_create(nfeatures=1000)

    # 3. 각 영역에서 특징점(Keypoints)과 디스크립터(Descriptors) 추출
    kp_l, des_l = orb.detectAndCompute(left_img, None)
    kp_c, des_c = orb.detectAndCompute(center_img, None)
    kp_r, des_r = orb.detectAndCompute(right_img, None)

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

    # 6. 결과 출력 및 시각화 준비
    is_tripartite = len(valid_left_matches) >= min_matches and len(valid_right_matches) >= min_matches

    print("-" * 40)
    print(f"중앙 영역 추출 특징점: {len(kp_c)}개")
    print(f"좌측 일치 특징점: {len(valid_left_matches)}개 (통과 기준: {min_matches}개)")
    print(f"우측 일치 특징점: {len(valid_right_matches)}개 (통과 기준: {min_matches}개)")
    print(f"최종 판별: {'✅ 삼분할 영상입니다.' if is_tripartite else '❌ 일반 영상입니다.'}")
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
test_image_path = r"image\input_4.jpg"

# 실행
test_tripartite_orb(
    image_path=test_image_path,
    tol=15,           # 픽셀 오차 허용치 (영상 편집 시 미세하게 틀어짐 감안)
    min_matches=30    # 좌/우 각각 최소 30개의 점이 일치하면 삼분할로 인정
)