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

### 구간 병합 (코덱 카피)

| 옵션 | 설명 |
|------|------|
| `--merge` | 검출된 구간만 코덱 카피로 잘라 한 파일로 이어붙여 저장. **인자 없으면** 원본과 같은 폴더에 **원본파일명_merged.mp4** 로 저장. |
| `--merge OUTPUT` | 위와 동일하되, **OUTPUT**에 절대경로·파일명을 주면 그 위치에 저장. 예: `--merge "F:\세경\result.mp4"` |

### 구간 목록 파일 (원하는 구간만 편집 후 병합)

| 옵션 | 설명 |
|------|------|
| `--segments-out FILE` | 검출된 구간 목록을 **텍스트 파일**로 저장. 한 줄에 `시작시각 끝시각`. 나중에 줄 삭제 또는 `#` 붙여서 제외할 구간 정한 뒤 `--segments-in`으로 사용. |
| `--segments-in FILE` | 구간 **검출 생략**, 파일에서 구간 목록만 읽음. `--merge` 와 함께 쓰면 해당 목록만 잘라서 병합. (예: 12개 중 10번만 제외하고 병합) |

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

# 검출 후 구간만 병합: 인자 없으면 같은 폴더에 video_merged.mp4 생성
python tripartite_section_check.py "F:\세경\video.ts" --cuda --merge

# 병합 결과를 지정 경로/파일명으로 저장
python tripartite_section_check.py "F:\세경\video.ts" --cuda --merge "F:\세경\result.mp4"

# 구간 목록을 txt로 저장 → 편집(제외할 구간 줄 삭제 또는 #) → 그 목록만 병합
python tripartite_section_check.py "F:\세경\video.ts" --cuda --segments-out "F:\세경\segments.txt"
# (segments.txt 에서 10번 구간 줄 삭제 또는 앞에 # 붙인 뒤)
python tripartite_section_check.py "F:\세경\video.ts" --segments-in "F:\세경\segments.txt" --merge

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
