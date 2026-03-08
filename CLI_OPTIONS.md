# tripartite_section_check.py 명령줄 옵션

FHD 영상에서 삼분할(좌·중·우 동일) 구간을 검출하는 스크립트의 실행 옵션 정리입니다.

---

## 사용법

```bash
python tripartite_section_check.py <input> [옵션...]
```

---

## 인자

| 인자 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `input` | 경로 | ✅ | 입력 영상 파일 (예: `input.ts`, `F:\영상\video.mp4`) |

---

## 옵션

### 기본 동작

| 옵션 | 설명 |
|------|------|
| `--no-min-duration` | 최소 구간 길이(20초) 검사 없이 발견된 구간을 모두 출력 |
| `--strict` | 좌=중앙=우 모두 비교. 기본은 **좌·우만** 비교 (중앙에만 효과 있을 때용) |

### 탐색 설정

| 옵션 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `--coarse-interval SEC` | float | 30.0 | 코스 스캔 간격(초). 작을수록 촘촘, 클수록 성김 |
| `--workers N` | int | CPU 코어 수 | 코스 스캔에 쓸 병렬 프로세스 수. 1이면 순차 처리 |

### GPU

| 옵션 | 설명 |
|------|------|
| `--cuda` | ffmpeg CUDA 디코딩 사용. CuPy가 있으면 판정 연산도 GPU (NVIDIA 드라이버 필요) |

### 검증 모드 (`--verify`)

지정 구간만 샘플링해 삼분할 여부·MSE를 출력합니다. 전체 탐색 없이 파인튜닝·원인 분석용입니다.

| 옵션 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `--verify START END` | 시각 2개 | - | 검증할 구간. 예: `--verify 02:21:48.357 02:24:00.679` (HH:MM:SS.mmm 또는 초) |
| `--verify-interval SEC` | float | 5.0 | 검증 구간 내 샘플 간격(초). 1~2면 더 촘촘 |

### 검증 시 이미지 저장 (`--verify-export`)

`--verify`와 함께 쓰면, 샘플 시점의 크롭·좌/중/우 이미지를 지정 폴더에 저장합니다. **opencv-python** 필요.

| 옵션 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `--verify-export DIR` | 경로 | - | 이미지를 저장할 폴더 |
| `--verify-export-max N` | int | 20 | 저장할 시점 개수. **0**이면 제한 없이 전부 저장 |
| `--verify-export-only-x` | 플래그 | - | 삼분할 **X(불일치)**로 판정된 시점만 저장 |

---

## 실행 예시

```bash
# 기본: 전체 영상에서 삼분할 구간 탐색
python tripartite_section_check.py "F:\영상\video.ts"

# GPU 사용
python tripartite_section_check.py "video.ts" --cuda

# 최소 구간 길이 무시하고 모든 구간 출력
python tripartite_section_check.py "video.ts" --no-min-duration

# 특정 구간만 검증 (삼분할 O/X, MSE 로그만)
python tripartite_section_check.py "video.ts" --verify 02:21:48.357 02:24:00.679

# 검증 + 이미지 저장 (OK/NG 모두, 최대 20개)
python tripartite_section_check.py "video.ts" --verify 02:21:48 02:24:00 --verify-export "F:\export"

# 검증 + NG 시점만 이미지 무제한 저장
python tripartite_section_check.py "video.ts" --cuda --verify 02:21:48.357 02:24:00.679 --verify-export "F:\export" --verify-export-only-x --verify-export-max 0

# 검증 구간을 1초 간격으로 촘촘히
python tripartite_section_check.py "video.ts" --verify 02:21:00 02:22:00 --verify-interval 1
```

---

## 참고

- **필수 외부 도구**: `ffmpeg`, `ffprobe` (PATH에 등록)
- **패키지**: `opencv-python`, `numpy`; GPU 시 `cupy-cuda12x` 등 (선택)
- **정렬 허용**: 가상선이 640/1280이 아닌 647/1273처럼 약간 어긋난 경우도 인식하도록, 코드 내 `ALIGN_TOLERANCE_PX`(기본 30)로 경계를 ±N픽셀 탐색합니다. 명령줄 옵션은 없고, 필요 시 스크립트 상단 상수만 수정하면 됩니다.
