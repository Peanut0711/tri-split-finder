## `tripartite_section_check.py`가 `tri_split_detector.py`보다 빠른 이유

이 문서는 같은 TS 파일을 분석했을 때
`tripartite_section_check.py`가 약 **50초**, `tri_split_detector.py --all-segments`가 약 **157초**가 걸린 상황을 기준으로,
두 스크립트의 구조 차이 때문에 생기는 **성능 차이의 원인**을 정리한 것입니다.

---

### 1. 프레임 추출 방식의 차이 (가장 큰 차이)

- **`tripartite_section_check.py`**
  - `extract_frame()`에서 FFmpeg를 이렇게 사용합니다.
    - `-ss t -i input.ts -vframes 1 -f rawvideo -pix_fmt bgr24 pipe:1`
  - 즉, **표준 출력(stdout)으로 raw BGR 바이트를 받아서 바로 `np.frombuffer`로 배열을 만듭니다.
  - **디스크에 이미지를 쓰지 않습니다.**

- **`tri_split_detector.py`**
  - `_extract_frame_ffmpeg()`에서
    - `-ss t -i input.ts -vframes 1 -q:v 2 tmp.png`
  - FFmpeg가 매번 **PNG 파일을 디스크에 저장**하고,
    - 그 뒤에 `cv2.imread(tmp.png)`로 다시 읽어옵니다.
  - 매 샘플마다 **파일 생성 + 디스크 쓰기 + 디스크 읽기 + 파일 삭제**가 들어가므로,
    - HDD 환경에서는 특히 **I/O 비용이 매우 크게** 작용합니다.

**→ 같은 샘플 수라면, rawvideo 파이프 방식인 `tripartite_section_check.py`가 훨씬 빠를 수밖에 없습니다.**

---

### 2. 병렬 처리와 자원 활용

- **`tripartite_section_check.py`**
  - 코스 스캔 단계에서 `ProcessPoolExecutor`를 이용해
    - `_coarse_worker()`를 **멀티프로세스**로 실행할 수 있습니다. (`--workers N`)
  - CPU 코어 수만큼 FFmpeg + 판정 연산을 병렬로 돌리며,
    - **SSD/빠른 디스크**일수록 이 병렬성이 잘 살아납니다.
  - 선택적으로 `--cuda` 옵션으로
    - FFmpeg 하드웨어 디코딩(`-hwaccel cuda`)과
    - CuPy 기반 GPU 판정 연산도 사용할 수 있습니다.

- **`tri_split_detector.py`**
  - 항상 **단일 프로세스/단일 스레드**로 동작합니다.
  - FFmpeg 호출과 OpenCV 연산이 모두 **직렬**로만 수행됩니다.

**→ CPU·GPU·SSD 환경에서는 `tripartite_section_check.py`가 병렬성을 활용해 훨씬 더 많은 샘플을 같은 시간에 처리할 수 있습니다.**

> 다만, 질문에서처럼 **7200rpm HDD에 영상이 있는 경우** 디스크 랜덤 읽기가 병목이 되기 때문에,
> 병렬화를 해도 HDD가 감당 못 하면 속도 이득이 제한될 수 있습니다.

---

### 3. 해상도/영역 처리 전략

- 두 스크립트 모두 **Y축 중앙부만 사용**하고, **가로 해상도를 줄이거나(스케일) 템플릿 영역만 사용**해서
  - 완전한 FHD 전체를 매번 처리하지는 않습니다.

- **추가로 `tripartite_section_check.py`가 갖는 이점**
  - 코스 스캔 시 `--coarse-scale` 옵션을 통해
    - 예: 640/960/1280 폭으로 **더 낮은 해상도만 사용**해 빨리 판정할 수 있습니다.
  - 판정 로직은 픽셀 평균 제곱 오차(MSE) 기반으로 구현되어 있어,
    - CuPy가 있을 경우 GPU에서 벡터 연산으로 빠르게 처리할 수 있게 설계되어 있습니다.

- **`tri_split_detector.py`**
  - Y 안전영역 절단 + 1/4 축소를 이미 적용했지만,
  - 프레임 추출 단계(FFmpeg→PNG→디스크 I/O)가 병목이라,
    - 해상도 축소 효과가 전체 시간에는 덜 반영됩니다.

**→ 연산 측면에서는 두 스크립트 모두 최적화가 되어 있지만,
현재 구조에서는 `tri_split_detector.py`는 “연산보다 디스크 I/O”가 더 큰 병목입니다.**

---

### 4. 캐시와 파이프라인 설계

- **`tripartite_section_check.py`**
  - `use_coarse_cache=True`일 때
    - 코스 스캔 중 추출한 프레임을 `.npy`로 저장해 두었다가,
    - 경계 정밀화(이진 탐색)에서 **같은 시점 근처 프레임을 재사용**할 수 있습니다.
  - 또한, 함수 단위로
    - **코스 스캔 → 경계 이진 탐색 → 키프레임 보정 → 추출/병합**
    - 각 단계의 소요 시간을 `timings` dict로 분리하여
      - **어느 단계가 병목인지 분석/튜닝**하기 쉽게 설계되어 있습니다.

- **`tri_split_detector.py`**
  - 현재는
    - 코스 스캔(30초 간격) + 첫 구간 시작 이진 탐색 + 디버그 이미지 저장,
    - 또는 `--all-segments` 시 단순한 전체 코스 스캔(경계 정밀화 없음) 정도만 구현되어 있습니다.
  - 프레임 재사용 캐시나 단계별 타이밍 계측은 없는 상태입니다.

**→ `tripartite_section_check.py`는 “한 번 추출한 프레임을 재활용”하고,
어느 단계가 느린지 측정할 수 있게 설계되어 있어, 같은 기능을 더 낮은 비용으로 수행할 수 있습니다.**

---

### 5. 요약: 속도 차이의 핵심 포인트

`tripartite_section_check.py`가 같은 파일을 약 50초에 처리하고,
`tri_split_detector.py --all-segments`가 약 157초가 걸린 상황을 요약하면 다음과 같습니다.

1. **프레임 추출 경로**  
   - `tripartite_section_check.py`: FFmpeg → **rawvideo 파이프(in-memory)** → `numpy`  
   - `tri_split_detector.py`: FFmpeg → **PNG 파일로 디스크 저장** → `cv2.imread` → `numpy`  
   → 디스크 I/O, PNG 인코딩/디코딩 비용 때문에 `tri_split_detector.py`가 훨씬 느림.

2. **병렬 처리와 자원 활용**  
   - `tripartite_section_check.py`: 코스 스캔을 **멀티프로세스 + (선택) GPU**로 처리 가능.  
   - `tri_split_detector.py`: 단일 프로세스, 직렬 처리.

3. **캐시 및 파이프라인 설계**  
   - `tripartite_section_check.py`: 코스 스캔 프레임 캐시, 단계별 타이밍 측정으로 병목 분석/튜닝 용이.  
   - `tri_split_detector.py`: 현재는 단순 코스 스캔 + 1회 이진 탐색 수준.

실질적으로 가장 큰 차이는 **“프레임을 어떻게 가져오느냐(파이프 vs PNG+디스크)”**이며,
이 부분을 `tri_split_detector.py`에서 `tripartite_section_check.py`와 비슷한 구조(FFmpeg rawvideo 파이프)로 바꾸면,
같은 30초 점프 조건에서도 전체 시간이 크게 줄어들 여지가 있습니다.

