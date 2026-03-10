# Tri-Split Finder (삼분할 구간 검출기)

FHD 영상에서 **"화면 정중앙 1/3을 양옆으로 복제한"** 삼분할 구간을 자동으로 찾아 시작·종료 타임스탬프를 출력하는 도구입니다. 상단 팝업을 피하기 위해 높이(y) 중앙부만 샘플하여 비교합니다.

## 주요 기능

- **코스 스캔 + 경계 이진 탐색**: 긴 영상(3~6시간)에서도 효율적으로 구간 검출
- **CPU 멀티코어 병렬화**: 코스 스캔 시 `--workers`로 프로세스 수 지정
- **선택적 GPU 가속**: `--cuda`로 ffmpeg CUDA 디코딩, CuPy 설치 시 판정 연산도 GPU
- **저해상도 코스 스캔**: `--coarse-scale`(640/960/1280)로 스캔만 저해상도, 경계는 원본 해상도
- **구간 검증 모드**: `--verify`로 지정 구간만 샘플링해 삼분할 여부·MSE 리포팅
- **구간 병합**: 검출된 구간만 코덱 카피로 잘라 한 파일로 저장 (`--merge`)
- **구간 목록 입출력**: `--segments-out`으로 저장, `--segments-in`으로 읽어 병합만 수행 가능

## 요구 사항

- **시스템**: [ffmpeg](https://ffmpeg.org/), [ffprobe](https://ffmpeg.org/ffprobe.html)가 PATH에 있어야 합니다.
- **Python**: 3.10+
- **패키지**:
  - 필수: `numpy`
  - 선택: `opencv-python` (이미지 저장/내보내기 시), `cupy-cuda12x` 등 (GPU 판정 시)

## 설치

**요구 사항·패키지 설명**: [Docs/REQUIREMENTS.md](Docs/REQUIREMENTS.md)

```bash
# 클론
git clone <repository-url>
cd tri-split-finder

# 가상환경 권장
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS

# 필수 패키지만 설치 후 바로 사용 (권장)
pip install -r requirements_tripartite.txt

# 선택: 검증 시 이미지 내보내기
pip install opencv-python

# 선택: GPU 판정 (NVIDIA CUDA 12.x)
pip install cupy-cuda12x
```

## 사용법

### 기본 실행

```bash
python tripartite_section_check.py input.ts
```

### 자주 쓰는 옵션

| 옵션 | 설명 |
|------|------|
| `--workers N` | 코스 스캔 병렬 프로세스 수 (기본: CPU 코어 수) |
| `--cuda` | GPU 사용 (ffmpeg CUDA 디코딩, CuPy 있으면 판정도 GPU) |
| `--no-min-duration` | 최소 구간 길이(20초) 검사 없이 모두 출력 |
| `--strict` | 좌=중앙=우 모두 비교 (기본은 좌·우만 비교) |
| `--coarse-interval SEC` | 코스 스캔 간격(초). 기본 30 |
| `--coarse-scale W` | 코스 스캔만 저해상도 (640/960/1280) |
| `--boundary-tolerance SEC` | 경계 이진 탐색 정밀도(초) |
| `--align-tolerance PX` | 경계 정렬 탐색 허용(픽셀). 0=고정 1/3·2/3 |
| `--merge [OUTPUT]` | 검출 구간만 잘라 한 파일로 저장 |
| `--segments-out [FILE]` | 구간 목록을 텍스트 파일로 저장 |
| `--segments-in [FILE]` | 파일에서 구간 목록 읽어 병합만 수행 |
| `--verify START END` | 지정 구간만 검증 (삼분할 여부·MSE 리포팅) |

### 예시

```bash
# 멀티코어 + GPU
python tripartite_section_check.py live.ts --workers 8 --cuda

# 구간 검출 후 목록 저장 + 병합
python tripartite_section_check.py live.ts --segments-out --merge

# 구간 목록만 읽어서 병합 (검출 생략)
python tripartite_section_check.py live.ts --segments-in --merge result.mp4

# 특정 구간만 검증 (파인튜닝·미검출 원인 분석)
python tripartite_section_check.py live.ts --verify 02:21:48.357 02:30:00.679 --verify-interval 2

# 검증 시 보는 영역 이미지 저장 (opencv-python 필요)
python tripartite_section_check.py live.ts --verify 00:10:00 00:15:00 --verify-export ./debug_frames
```

## 출력 예

```
입력: C:\Videos\live.ts
길이: 06:23:45.0 (23025.0초), 해상도: 1920x1080
비교 영역: 높이 35% ~ 65% (상단 팝업 제외)
모드: 좌·우만 비교 (중앙 무시)
코스 스캔: 간격 30s, 병렬 프로세스 8개, 자원 CPU
--------------------------------------------------
...
삼분할 구간:
  [1] 00:12:34.5 ~ 00:15:22.1  (길이 167.6초)
  [2] 01:45:00.0 ~ 01:48:30.2  (길이 210.2초)
--------------------------------------------------
총 2개 구간  (전체 소요: 12분 34초)
길이 총합: 377.8초  (6분 17초)
```

## 파이프라인 개요

1. **메타**: ffprobe로 영상 길이·해상도 조회  
2. **시점 목록**: 코스 스캔용 시점 생성  
3. **코스 스캔**: 일정 간격으로 프레임 추출 → 좌/중/우(또는 좌·우만) 비교, MSE 임계치로 삼분할 여부 판정  
4. **경계 정밀화**: 삼분할이 시작/끝나는 시점을 이진 탐색으로 정밀화  
5. **병합**: 인접 구간을 `MERGE_GAP_SECONDS` 이내면 하나로 병합  
6. **최소 길이 필터**: `MIN_SEGMENT_DURATION`(기본 20초) 미만 구간 제외  

자세한 옵션과 파이프라인 설명은 `Docs/CLI_OPTIONS.md`, `Docs/PIPELINE_AND_OPTIMIZATION.md`를 참고하세요.

## 라이선스

프로젝트 루트의 라이선스 파일을 참고하세요.
