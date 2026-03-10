# Tri-Split Finder 요구 사항 및 설치

`tripartite_section_check.py`만 사용할 때 필요한 환경과 패키지, 설치 방법을 정리했습니다.

---

## 1. 시스템 요구 사항

| 항목 | 설명 |
|------|------|
| **ffmpeg / ffprobe** | PATH에 설치되어 있어야 합니다. 영상 메타·프레임 추출에 사용. [ffmpeg 공식](https://ffmpeg.org/) |
| **Python** | 3.10 이상 |

```bash
# ffmpeg 설치 확인
ffmpeg -version
ffprobe -version
```

---

## 2. Python 패키지

### 필수 (기본 실행)

| 패키지 | 용도 |
|--------|------|
| **numpy** | 프레임 배열·MSE 계산 등 핵심 연산 |

이 두 가지만 있으면 **구간 검출**은 바로 사용 가능합니다.

### 선택 (기능별)

| 패키지 | 용도 | 필요한 경우 |
|--------|------|-------------|
| **opencv-python** | 이미지 저장 | `--verify-export` 로 검증 구간을 이미지로 저장할 때 |
| **cupy** (cupy-cuda12x 등) | GPU 판정 | `--cuda` 사용 시 판정 연산을 GPU로 돌리고 싶을 때 |

- CuPy 없이 `--cuda`를 켜도 동작합니다. ffmpeg CUDA 디코딩만 쓰고, 판정은 CPU로 합니다.
- CuPy는 CUDA 버전에 맞춰 설치: CUDA 12.x → `cupy-cuda12x`, CUDA 11.x → `cupy-cuda11x`

---

## 3. 설치 방법

### 3.1 최소 설치 (바로 사용)

기본 구간 검출만 쓸 경우:

```bash
pip install -r requirements_tripartite.txt
```

또는:

```bash
pip install numpy
```

이후:

```bash
python tripartite_section_check.py input.ts
```

### 3.2 가상환경 권장

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS

pip install -r requirements_tripartite.txt
```

### 3.3 선택 기능까지 설치

| 목적 | 명령 |
|------|------|
| 검증 구간 이미지 저장 (`--verify-export`) | `pip install opencv-python` |
| GPU 판정 (NVIDIA CUDA 12.x) | `pip install cupy-cuda12x` |

한 번에:

```bash
pip install -r requirements_tripartite.txt opencv-python
# GPU 사용 시 추가:
# pip install cupy-cuda12x
```

---

## 4. 동작 확인

```bash
# 도움말
python tripartite_section_check.py --help

# 짧은 영상으로 테스트 (ffmpeg/ffprobe 경로도 함께 확인됨)
python tripartite_section_check.py your_video.ts
```

- `opencv-python`이 없을 때 `--verify-export`를 쓰면, 스크립트에서 해당 옵션 사용 시 안내 메시지를 출력합니다.
- CuPy가 없을 때 `--cuda`를 켜면, 디코딩만 GPU이고 판정은 CPU로 동작합니다.

---

## 5. 요약

- **시스템**: ffmpeg, ffprobe (PATH), Python 3.10+
- **필수 패키지**: `numpy` → `pip install -r requirements_tripartite.txt` 로 설치 후 바로 사용 가능
- **선택**: `opencv-python` (이미지 저장), `cupy-cuda12x` 등 (GPU 판정)
