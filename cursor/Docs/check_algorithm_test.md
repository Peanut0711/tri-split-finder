# 분할 영상(이분할/삼분할) 검출 시스템 설계 문서

## 1. 목적

본 시스템은 입력 이미지 또는 영상 프레임이 다음 중 어떤 구조인지 판별한다.

* 단일 영상
* 이분할 영상 (좌/우)
* 삼분할 영상 (좌/중앙/우)

추가 조건:

* 각 영역은 완전히 동일하지 않을 수 있음 (팝업, UI 존재)
* 영역 간 일부 겹침 존재 가능
* 분할 비율은 고정되지 않음 (정확히 1/2, 1/3 아님)
* 경계 위치는 약간의 오차 존재 (±5~10%)

---

## 2. 핵심 접근 전략

본 문제는 다음 두 가지 특징을 기반으로 해결한다:

1. **경계 검출 (Structural discontinuity)**

   * 분할 영상은 수직 방향에서 강한 edge를 생성함

2. **콘텐츠 유사성 검증 (Content similarity)**

   * 분할된 영역은 서로 유사한 콘텐츠를 포함함

---

## 3. 전체 처리 파이프라인

### Stage 1: Edge 기반 분할 후보 검출 (필수)

### Stage 2: SSIM 기반 구조 유사도 검증 (핵심)

### Stage 3: Feature 기반 보강 검증 (ORB)

### Stage 4: Deep Feature 기반 최종 검증 (CNN)

각 단계는 순차적으로 수행되며, 조건 만족 시 즉시 종료한다.

---

## 4. Stage 1: Edge Projection 기반 경계 검출

### 목적

영상 내 분할 경계를 후보로 검출

### 처리 과정

1. 입력 이미지를 grayscale 변환
2. Sobel 필터 (x 방향) 적용
3. 절대값 후 column-wise sum 수행
4. Gaussian smoothing 적용
5. peak detection 수행

### 출력

* 분할 후보 위치 리스트 (x 좌표)

### 조건

* peak prominence ≥ 전체 평균의 2~3배
* peak 간 최소 거리 ≥ width * 0.1

---

## 5. Stage 2: SSIM 기반 유사도 검증 (핵심 단계)

### 목적

분할된 영역 간 구조적 유사성 확인

### 전처리

* 각 영역 resize (속도 최적화)

  * 좌/우: 128x256
  * 중앙: 256x256

### 핵심 기법: Sliding Window SSIM

중앙 영역이 더 클 경우, 다음 방식 적용:

* 작은 영역(L 또는 R)을 기준으로
* 큰 영역(C) 내부에서 sliding 하면서 SSIM 최대값 탐색

### 계산 방식

* SSIM(L, sliding(C))
* SSIM(R, sliding(C))

### threshold

* SSIM ≥ 0.6 → 유사한 영역으로 판단

---

## 6. Stage 3: ORB Feature Matching (보강 단계)

### 목적

SSIM 실패 시 특징 기반 유사도 검증

### 처리 과정

1. ORB keypoint 추출
2. BFMatcher로 매칭
3. ratio test 적용 (0.75)

### 판단 기준

* good match ≥ 50 → 유사 영역

---

## 7. Stage 4: CNN Feature 기반 검증 (최종 fallback)

### 목적

최종적으로 의미 기반 유사도 판단

### 모델

* ResNet18 (pretrained)
* 마지막 FC layer 제거 후 feature vector 사용

### 처리

1. 각 영역 feature 추출
2. cosine similarity 계산

### 기준

* similarity ≥ 0.8 → 동일 콘텐츠

---

## 8. 분할 판정 로직

### Case 1: 삼분할

조건:

* Stage1에서 peak ≥ 2개
* 좌/중앙, 중앙/우 유사도 모두 threshold 이상

### Case 2: 이분할

조건:

* peak = 1개
* 좌/우 유사도 threshold 이상

### Case 3: 단일 영상

조건:

* 위 조건 모두 미충족

---

## 9. 추가 최적화 전략

### ROI 제한

* 상하 10% 제거 후 중앙 80% 영역만 사용
* 팝업 및 UI 영향 감소

### 속도 최적화

* Stage2까지만으로 대부분 판정 가능
* Stage3, 4는 fallback으로만 사용

### GPU 활용 (선택)

* CNN 단계에서만 사용 권장

---

## 10. 예외 처리

* peak가 과도하게 많을 경우 → noise로 간주
* SSIM 계산 실패 시 → ORB 단계로 fallback
* feature 추출 실패 시 → single로 처리

---

## 11. 출력 형식

다음 중 하나 반환:

* "single"
* "2-split"
* "3-split"

옵션:

* confidence score 추가 가능

---

## 12. 시스템 특성 요약

* 팝업 및 UI에 강건함
* 비대칭 분할 지원
* 부분 겹침 대응 가능
* 실시간 처리 가능 (고성능 환경 기준)

---

## 13. 핵심 요약

본 시스템은 다음 구조를 따른다:

"Edge 기반 분할 위치 탐지 → Sliding SSIM 기반 구조 검증 → Feature 기반 보정"

이 접근은 정확도와 속도의 균형을 최적으로 유지한다.




-------------------

# 14. 메모

좋다. 요구사항 + 환경 보면
👉 “정확도 우선 + 라이브러리 적극 활용 + 단계적 fallback” 구조로 가는 게 최적이다.

아래는 실전용 파이프라인 (Python 기준)이다.
단순 샘플이 아니라 실제 현업 수준 구조로 짰다.

🔥 전체 구조 (우선순위 기반 파이프라인)
Stage 1: Edge Projection (fast, high hit-rate)
Stage 2: SSIM (robust, popup 대응)
Stage 3: Feature Matching (ORB / CNN)
Stage 4: Deep Embedding (CLIP / ResNet fallback)
✅ 사용 라이브러리
pip install opencv-python numpy scipy scikit-image torch torchvision timm


# 예시 코드

pip install faiss-cpu  # 빠른 similarity
pip install clip-anytorch  # CLIP
1️⃣ Stage 1: Edge Projection (경계 검출)
import cv2
import numpy as np
from scipy.signal import find_peaks

def find_split_candidates(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Sobel X
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gx = np.abs(gx)

    # column projection
    col_sum = gx.mean(axis=0)

    # smoothing
    col_sum = cv2.GaussianBlur(col_sum, (51,1), 0)

    # peak detection
    peaks, props = find_peaks(
        col_sum,
        prominence=np.mean(col_sum)*2,
        distance=img.shape[1]//10
    )

    return peaks, col_sum
2️⃣ Stage 2: Sliding SSIM (핵심)
from skimage.metrics import structural_similarity as ssim

def sliding_ssim(a, b):
    h, w = a.shape[:2]
    max_score = 0

    for x in range(0, b.shape[1] - w, w//10):
        patch = b[:, x:x+w]
        score = ssim(a, patch, channel_axis=2)
        max_score = max(max_score, score)

    return max_score
3️⃣ Stage 3: ORB Feature Matching
def orb_similarity(a, b):
    orb = cv2.ORB_create(1000)

    kp1, des1 = orb.detectAndCompute(a, None)
    kp2, des2 = orb.detectAndCompute(b, None)

    if des1 is None or des2 is None:
        return 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    return len(good)
4️⃣ Stage 4: CNN Feature (ResNet)
import torch
import torchvision.transforms as T
import torchvision.models as models

device = "cuda" if torch.cuda.is_available() else "cpu"

model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval().to(device)

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224,224)),
    T.ToTensor(),
])

def extract_feature(img):
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        f = model(x).flatten()
    return f.cpu().numpy()

def cosine_sim(a, b):
    return np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b))
🔥 최종 통합 로직
def detect_split(img):
    H, W = img.shape[:2]

    peaks, score = find_split_candidates(img)

    # 후보 개수로 1차 판단
    if len(peaks) >= 2:
        # 3분할 후보
        peaks = sorted(peaks)[:2]
        x1, x2 = peaks

        L = img[:, :x1]
        C = img[:, x1:x2]
        R = img[:, x2:]

        # resize for speed
        Lr = cv2.resize(L, (128,256))
        Cr = cv2.resize(C, (256,256))
        Rr = cv2.resize(R, (128,256))

        # --- Stage 2: SSIM ---
        s1 = sliding_ssim(Lr, Cr)
        s2 = sliding_ssim(Rr, Cr)

        if s1 > 0.6 and s2 > 0.6:
            return "3-split (SSIM)"

        # --- Stage 3: ORB ---
        f1 = orb_similarity(Lr, Cr)
        f2 = orb_similarity(Rr, Cr)

        if f1 > 50 and f2 > 50:
            return "3-split (ORB)"

        # --- Stage 4: CNN ---
        fL = extract_feature(Lr)
        fC = extract_feature(Cr)
        fR = extract_feature(Rr)

        c1 = cosine_sim(fL, fC)
        c2 = cosine_sim(fR, fC)

        if c1 > 0.8 and c2 > 0.8:
            return "3-split (CNN)"

    elif len(peaks) == 1:
        x = peaks[0]

        L = img[:, :x]
        R = img[:, x:]

        Lr = cv2.resize(L, (128,256))
        Rr = cv2.resize(R, (128,256))

        s = ssim(Lr, Rr, channel_axis=2)

        if s > 0.7:
            return "2-split"

    return "single"