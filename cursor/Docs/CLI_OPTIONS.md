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
| `--coarse-scale W` | int | (없음) | 코스 스캔만 저해상도로 수행. 폭 640/960/1280 중 하나. 미지정 시 원본 해상도. 경계 정밀화는 항상 원본. |
| `--coarse-cache` | 플래그 | - | 코스 스캔 시 각 시점 프레임을 임시 폴더에 캐시. 경계 정밀화에서 원본 해상도일 때만 재사용 후 작업 끝에 삭제. `--coarse-scale`과 함께 사용 가능(캐시는 저장만, 경계에서는 원본 사용). |
| `--boundary-tolerance SEC` | float | 0.9375 | 경계 이진 탐색 정밀도(초). 크게 주면 경계 단계가 빨라지나 정확도 완화. 예: 0.5(더 정밀), 1.0 |
| `--align-tolerance PX` | int | 5 | 경계 정렬 탐색 허용(픽셀). 0이면 고정 1/3·2/3만 사용. N이면 w/3±N 픽셀 범위에서 최적 좌/우 너비 탐색. 미지정 시 5. 예: 10, 30 |

### 구간 병합 (코덱 카피)

| 옵션 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `--merge` | - | - | 검출된 구간만 코덱 카피로 잘라 한 파일로 이어붙여 저장. **인자 없으면** 원본과 같은 폴더에 **원본파일명_merged.mp4** 로 저장. |
| `--merge OUTPUT` | - | - | 위와 동일하되, **OUTPUT**에 절대경로·파일명을 주면 그 위치에 저장. 예: `--merge "F:\세경\result.mp4"` |
| `--merge-workers N` | int | 2 | 병합 시 구간 추출 병렬 워커 수. 네트워크/HDD 원본이면 2 권장. **로컬 NVMe** 원본이면 4~8 시도 가능. 예: `--merge-workers 6` |
| `--merge-notify MODE` | 선택 | print | 병합·복사 완료 시 알림 방식. `print`=콘솔만, `sound`=비프음, `toast`=Windows 토스트(win11toast 권장·Win11 호환, 클릭 시 폴더 열기; 없으면 win10toast-click→plyer), `all`=sound+toast. 예: `--merge-notify toast` |

### 구간 목록 파일 (원하는 구간만 편집 후 병합)

| 옵션 | 설명 |
|------|------|
| `--segments-out` | 검출된 구간 목록을 **텍스트 파일**로 저장. **인자 없으면** 원본과 같은 폴더에 **원본파일명_seg.txt** 로 저장. |
| `--segments-out FILE` | 위와 동일하되, **FILE**에 경로를 지정하면 그 위치에 저장. |
| `--segments-in` | 구간 **검출 생략**. **인자 없으면** 원본과 같은 폴더의 **원본파일명_seg.txt** 가 있는지 확인 후 해당 파일에서 구간 목록 읽음. 없으면 오류. |
| `--segments-in FILE` | 위와 동일하되, **FILE**에 경로 지정 시 그 파일에서 읽음. `--merge` 와 함께 쓰면 해당 목록만 잘라서 병합. |

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

## 네트워크 HDD / NAS 환경 추천 옵션

홈 네트워크의 다른 PC에 연결된 HDD(예: `\\192.168.x.x\share\video.ts`)를 스캔할 때는 **디스크·네트워크가 병목**이 되기 쉽습니다.  
아래 조합을 기본으로 쓰는 것을 권장합니다.

- **부하 최소 + 안정성 위주**
  - `--workers 1`
  - `--coarse-scale 640`
  - `--coarse-interval 30` (또는 45~60)
  - `--merge-workers 1`
  - 설명:
    - `--workers 1`: 코스 스캔을 순차 처리해 **동시 읽기 1회**만 발생 → 네트워크/HDD 100% 부하 완화.
    - `--coarse-scale 640`: 코스 스캔을 저해상도로만 디코딩 → 프레임당 처리량과 디코딩 부하 감소.
    - `--coarse-interval`: 30초는 기본값, **60초 이상**으로 키우면 읽기 횟수가 줄어들어 더 가볍지만, 탐색이 성김.
    - `--merge-workers 1`: 병합 단계에서도 동시 접근 1개만 유지.

- **속도·부하 균형**
  - `--workers 1`
  - `--coarse-scale 960`
  - `--coarse-interval 30`
  - `--merge-workers 1` 또는 `2`

> 참고: 동일 네트워크 HDD에서의 실험 로그 기준으로,  
> `--workers 2`나 `--cuda`를 켰을 때 **총 실행 시간은 오히려 늘고**, 검출 결과는 동일했습니다.  
> 이 환경에서는 **I/O(네트워크/HDD)가 병목**이라, 디코딩을 GPU로 옮겨도 시간 단축 효과가 거의 없고,  
> 동시 접근 수를 줄이는 것이 더 중요합니다.

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

# 병합 완료 시 Windows 토스트 알림 (클릭 시 폴더 열기)
python tripartite_section_check.py "F:\세경\video.ts" --merge --merge-notify toast

# 구간 목록 저장: 인자 없으면 같은 폴더에 video_seg.txt 생성
python tripartite_section_check.py "F:\세경\video.ts" --cuda --segments-out
# (video_seg.txt 에서 제외할 구간 줄 삭제 또는 # 붙인 뒤) 인자 없이 같은 폴더의 video_seg.txt 로 병합
python tripartite_section_check.py "F:\세경\video.ts" --segments-in --merge

# 검증 + NG 시점만 이미지 무제한 저장
python tripartite_section_check.py "video.ts" --cuda --verify 02:21:48.357 02:24:00.679 --verify-export "F:\export" --verify-export-only-x --verify-export-max 0

# 검증 구간을 1초 간격으로 촘촘히
python tripartite_section_check.py "video.ts" --verify 02:21:00 02:22:00 --verify-interval 1
```

---

## 참고

- **필수 외부 도구**: `ffmpeg`, `ffprobe` (PATH에 등록)
- **패키지**: `opencv-python`, `numpy`; GPU 시 `cupy-cuda12x` 등 (선택)
- **정렬 허용**: 가상선이 640/1280이 아닌 647/1273처럼 약간 어긋난 경우도 인식하도록 `--align-tolerance PX`로 경계를 ±N픽셀 탐색합니다. 미지정 시 기본 5. 0이면 고정 1/3·2/3만 사용. 예: `--align-tolerance 30`
- **메모**: `--boundary-tolerance` 기본값은 **0.9375초**. 경계 이진 탐색 정밀도를 이 값으로 두어 속도·정확도 균형을 맞추고, 더 정밀히 하려면 `--boundary-tolerance 0.5` 등으로 줄이면 됩니다.
