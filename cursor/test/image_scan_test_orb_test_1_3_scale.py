"""
삼분할 영상 판별 (ORB 특징점 매칭) — 1/3 해상도 버전

이미지를 1/3로 축소한 뒤 판별하여 연산 부하를 줄입니다.

동작 방식:
  1) 중앙 기준 비교: 중앙(C) vs 좌(L), 중앙(C) vs 우(R) 로 매칭하여 삼분할 여부 판별.
  2) 실패 시 폴백: 위 판별이 부족하면 좌(L) vs 우(R) 직접 비교.
     - 좌·우가 동일 소스 복사본이면 로컬 좌표 (x,y)가 거의 일치하므로, L==R 일치점이 많으면 삼분할로 인정.
"""

import argparse
import time
import cv2
import numpy as np

def test_tripartite_orb_v2(image_path, tol=5, min_matches=15):
    """
    ORB 특징점 매칭을 이용해 삼분할 영상 여부를 판별합니다.
    (v2: ROI, 1/3 해상도·nfeatures 1500, 비대칭 판별)
    
    :param image_path: 테스트할 이미지 경로
    :param tol: 좌우 매칭 시 허용할 픽셀 위치 오차 (1/3 해상도 기준, 기본 5)
    :param min_matches: 좌/우 각각 최소 일치 특징점 개수 (1/3 해상도 기준, 기본 15)
    """
    img_raw = cv2.imread(image_path)
    if img_raw is None:
        print(f"Error: '{image_path}' 이미지를 불러올 수 없습니다.")
        return

    # [1/3 해상도] 부하 감소를 위해 축소 후 판별
    img = cv2.resize(img_raw, (0, 0), fx=1/3.0, fy=1/3.0, interpolation=cv2.INTER_AREA)

    t_start = time.perf_counter()
    h, w = img.shape[:2]
    slice_w = w // 3

    # [개선 1] 채팅창 방지를 위해 상단 60% 영역만 사용 (ROI 설정)
    roi_h = int(h * 0.6)
    left_img = img[:roi_h, :slice_w]
    center_img = img[:roi_h, slice_w:slice_w*2]
    right_img = img[:roi_h, slice_w*2:]

    # [1/3 해상도] 특징점 수 비례 (1/4보다 해상도 높음 → 1000)
    orb = cv2.ORB_create(nfeatures=1000)

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

    valid_lr = []
    # [Fallback] 중앙 기준 매칭 실패 시 → 좌/우 직접 비교 (L vs R)
    # (중앙만 줌/이펙트가 있고 좌·우가 동일 소스인 경우 대응)
    if not is_tripartite and des_l is not None and des_r is not None:
        print("  [알림] 중앙 기준 매칭 부족 -> 좌/우 직접 비교(L vs R) 시작")
        matches_lr = bf.match(des_l, des_r)
        for m in matches_lr:
            pt_l = kp_l[m.queryIdx].pt
            pt_r = kp_r[m.trainIdx].pt
            # 좌·우는 복사본이므로 로컬 좌표 (x,y)가 거의 같아야 함
            if abs(pt_l[0] - pt_r[0]) <= tol and abs(pt_l[1] - pt_r[1]) <= tol:
                valid_lr.append(m)
        cnt_lr = len(valid_lr)
        print(f"  [결과] 좌/우 직접 일치점: {cnt_lr}개")
        if cnt_lr >= min_matches:
            is_tripartite = True
            print("  [확정] ✅ 좌/우 일치 확인으로 삼분할 판정!")

    elapsed = time.perf_counter() - t_start
    print("-" * 40)
    print(f"결과 - 좌: {cnt_l} / 우: {cnt_r}" + (f" / L-R: {len(valid_lr)}" if valid_lr else "") + f" | 판별: {is_tripartite}")
    print(f"판별 소요: {elapsed:.3f}초")
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

    # [Fallback 사용 시] 좌측 <-> 우측 직접 매칭점 그리기 (보라색)
    for m in valid_lr:
        pt_l_global = (int(kp_l[m.queryIdx].pt[0]), int(kp_l[m.queryIdx].pt[1]))
        pt_r_global = (int(kp_r[m.trainIdx].pt[0] + slice_w * 2), int(kp_r[m.trainIdx].pt[1]))
        cv2.circle(debug_img, pt_l_global, 4, (255, 0, 255), -1)
        cv2.circle(debug_img, pt_r_global, 4, (255, 0, 255), -1)
        cv2.line(debug_img, pt_l_global, pt_r_global, (255, 0, 255), 1)

    # 화면에 결과 띄우기
    cv2.imshow("ORB Match Result", debug_img)
    print("아무 키나 누르면 창이 닫힙니다.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --- 실행 부분 ---
def main():
    parser = argparse.ArgumentParser(description="ORB 삼분할 판별 (1/3 해상도, 중앙 기준 + L vs R 폴백)")
    parser.add_argument("image_path", help="판별할 이미지 파일 경로")
    parser.add_argument("--tol", type=int, default=5,
                        help="픽셀 오차 허용치, 1/3 해상도 기준 (기본: 5)")
    parser.add_argument("--min-matches", type=int, default=15, dest="min_matches",
                        help="최소 일치 특징점 개수, 1/3 해상도 기준 (기본: 15)")
    args = parser.parse_args()

    test_tripartite_orb_v2(
        image_path=args.image_path,
        tol=args.tol,
        min_matches=args.min_matches,
    )


if __name__ == "__main__":
    main()