# Project Specification: Fast Tri-Split Video Segment Detector

## 1. Project Goal
3시간 분량의 FHD(1920x1080) 비디오에서 특정 형태의 '삼분할 영상'이 삽입된 구간의 정확한 시작(Start)과 끝(End) 타임스탬프를 초고속으로 찾아내는 파이썬 스크립트를 작성한다.

## 2. Hardware Environment
* **CPU:** AMD Ryzen 7 9800X3D (대용량 L3 캐시 활용, Numpy 연산에 극대화)
* **GPU:** NVIDIA RTX 5070 Ti (NVDEC를 활용한 초고속 하드웨어 디코딩용)
* **CUDA:** Ver 12.8
* **Language:** Python 3.10+
* **Key Libraries:** `decord` (초고속 프레임 랜덤 액세스), `opencv-python`, `numpy`

## 3. Core Constraints & Rules
1. **Resolution:** 원본 영상은 항상 FHD(1920x1080)이다.
2. **Tri-Split Characteristic:** 좌측, 중앙, 우측이 각각 약 640 폭으로 분할된 영상이다. 중앙 영상은 무시하며, **좌측 영상과 우측 영상이 동일한지**만 비교한다.
3. **Offset Variation:** 작업자 실수로 분할선이 최대 $\pm 5$ 픽셀 틀어질 수 있다. (예: 좌측 637, 우측 647). 따라서 단순 픽셀 비교 전, 최초 1회 `Template Matching`을 통해 정확한 X 오프셋을 찾아야 한다.
4. **Offset Lock:** 최초에 캘리브레이션 된 X 좌표 오프셋은 **해당 영상 내에서 절대 바뀌지 않는다.** 한 번 찾으면 이후 구간은 고정된 좌표로 뺄셈 연산만 수행한다.
5. **Minimum Duration:** 삼분할 영상은 한 번 등장하면 **최소 30초 이상** 지속된다.
6. **Y-Axis Safe Zone:** 상하단 20% 구간에는 노이즈나 자막이 있을 수 있으므로, 검사 영역은 Y 좌표 `216 ~ 864` 사이의 중앙부(높이 648)만 사용한다.
7. **Input Format:** 모든 원본 영상은 반드시 `*.ts` (Transport Stream) 포맷이다.
8. **Output Format:** 최종 추출되거나 병합(Merge)된 결과물은 반드시 `*.mp4` 컨테이너를 사용해야 한다.
9. **Lossless Processing:** 키프레임 단위로 구간을 구했으므로, 비디오와 오디오 스트림은 절대 재인코딩(Re-encoding)하지 않고 `Stream Copy(-c copy)` 방식으로 초고속으로 처리해야 한다. (화질 손실 금지)

## 4. Search Algorithm Strategy (Two-Track)
* **Phase 1 (10-Second Skip & Calibrate):** 최소 30초 이상 지속되므로 모든 프레임을 볼 필요가 없다. `decord`를 사용해 10초(10 * fps) 단위로 건너뛰며 프레임을 가져온다. 아직 오프셋을 모른다면 `cv2.matchTemplate`을 사용해 좌측 템플릿(폭 610)이 우측 영역에 존재하는지 확인하고, 일치율이 0.95 이상이면 해당 X 좌표를 전역 오프셋으로 `Lock` 한다.
* **Phase 2 (Fast Array Subtraction):** 오프셋이 고정된 후에는 무거운 매칭 함수를 버린다. 10초 점프를 유지하며, 고정된 좌표의 `left_crop`과 `right_crop` 간의 Numpy `absdiff` 평균값이 특정 임계치(예: 5.0) 이하인지 확인하여 구간 진입/이탈을 판별한다.
* **Phase 3 (Boundary Binary Search):** 10초 점프 중 상태가 변한 것(일반 영상 <-> 삼분할 영상)을 감지하면, 해당 10초 구간 내에서 이진 탐색(Binary Search) 또는 1초 단위 탐색을 통해 정확한 경계 프레임을 찾아낸다.

## 5. Keyframe-Aware Boundary Expansion (Snap to I-Frame)
* **Objective:** 삼분할 구간을 손실 없이 빠르고 깔끔하게 잘라내기(Lossless Cut) 위해, 검출된 정확한 Start/End 타임스탬프를 바깥쪽 방향의 키프레임(I-Frame)으로 확장하여 보정한다.
* **Logic:**
  1. Phase 3(Binary Search)를 통해 삼분할 영상이 시작되는 `exact_start_frame`과 끝나는 `exact_end_frame`을 찾는다.
  2. `decord` 라이브러리의 `vr.get_key_indices()` 함수를 호출하여 영상 전체의 키프레임 인덱스 리스트를 1회 확보한다.
  3. **Start 보정:** `exact_start_frame`보다 **작거나 같은(과거 방향)** 가장 가까운 키프레임을 찾아 `final_start_frame`으로 확정한다. (최대 3초 앞까지 허용)
  4. **End 보정:** `exact_end_frame`보다 **크거나 같은(미래 방향)** 가장 가까운 키프레임을 찾아 `final_end_frame`으로 확정한다. (최대 3초 뒤까지 허용)
  5. 최종적으로 반환하는 타임스탬프 구간은 이 키프레임 보정이 완료된 `final_start`와 `final_end`여야 하며, 이로써 삼분할 구간이 1프레임도 잘려나가지 않음을 100% 보장해야 한다.

## 6. Extraction & Merge Pipeline (FFmpeg 활용)
* **Objective:** 파이썬에서 찾은 타임스탬프 배열을 바탕으로, FFmpeg를 서브프로세스(subprocess)로 호출하여 실제 영상을 잘라내고 병합한다.
* **Extraction (자르기):** - `ffmpeg -ss [시작시간] -to [종료시간] -i input.ts -c copy output_segment_1.ts`
  - 각 삼분할 구간을 일단 개별 `.ts` 파일로 잘라낸다. (TS는 자르기 과정에서 타임스탬프 오류가 적음)
* **Merge to MP4 (병합 및 변환):** - 잘라낸 여러 개의 `.ts` 구간들을 하나로 합쳐서 최종 `.mp4`로 만든다.
  - FFmpeg의 `concat` demuxer를 사용한다. (임시 텍스트 파일 `list.txt`에 `file 'output_segment_1.ts'` 형태로 작성 후 입력)
  - 명령어 예시: `ffmpeg -f concat -safe 0 -i list.txt -c copy final_output.mp4`
  - 작업이 완료되면 중간 산출물인 임시 `.ts` 조각들과 `list.txt`는 파이썬에서 자동으로 삭제(Clean-up)한다.

## 7. Skeleton Code (Initial Setup)

```python
import numpy as np
import cv2
from decord import VideoReader, cpu, gpu

class SplitVideoDetector:
    def __init__(self, video_path):
        # decord를 사용하여 GPU 디코딩을 시도하거나, CPU로 빠르게 프레임 추출
        self.vr = VideoReader(video_path, ctx=cpu(0)) 
        self.fps = self.vr.get_avg_fps()
        self.total_frames = len(self.vr)
        
        # State variables
        self.is_calibrated = False
        self.left_x_end = 0
        self.right_x_start = 0
        self.crop_width = 0
        
        # Y-Axis Safe Zone
        self.y_start = 216
        self.y_end = 864

    def calibrate(self, frame):
        """
        초기 오프셋을 찾기 위한 Template Matching.
        성공 시 self.is_calibrated를 True로 변경하고 오프셋을 저장.
        """
        template = frame[self.y_start:self.y_end, 10:620]
        search_area = frame[self.y_start:self.y_end, 1260:1920]
        
        # TODO: cv2.matchTemplate 구현 및 임계값 검사
        # 찾았을 경우 self.left_x_end, self.right_x_start, self.crop_width 설정
        pass

    def check_fast_match(self, frame):
        """
        캘리브레이션 완료 후 Numpy 연산으로 빠르게 비교
        """
        if not self.is_calibrated:
            return False
            
        left_crop = frame[self.y_start:self.y_end, 0:self.left_x_end]
        # TODO: right_crop 추출 및 cv2.absdiff(left, right)로 평균 오차 계산
        # return True if match else False
        pass

    def scan_video(self):
        """
        10초 단위로 프레임을 점프하며 스캔하는 메인 루프.
        상태 변화 감지 시 경계선 탐색 로직 호출.
        """
        jump_frames = int(10 * self.fps)
        
        # TODO: 메인 탐색 루프 및 Binary Search 로직 구현
        pass

# Usage
# detector = SplitVideoDetector("input.mp4")
# segments = detector.scan_video()
# print(segments)