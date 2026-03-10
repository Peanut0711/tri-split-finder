📋 [시스템 프롬프트 / 지시사항]
당신은 동영상 파일(.ts)에서 특정 화면 분할 패턴(삼분할)이 나타나는 구간의 시작 시간과 종료 시간을 추출하는 파이썬 스크립트(tripartite_section_check.py)를 작성해야 합니다.
성능 최적화를 위해 모든 프레임을 검사하지 않고, 하이브리드 탐색(Grid Search + Binary Search) 알고리즘을 사용합니다.

1. 핵심 전제 조건 (Constraints)
목표 화면: 화면 정중앙 영역(가로 1/3)이 좌/우 영역(각 가로 1/3)에 복제되어 있는 '삼분할' 화면.

최소 유지 시간: 삼분할 화면이 한 번 시작되면 최소 30초 이상 유지됨. (이 조건이 Grid Search의 Maximum Step Size가 됨)

목표 정밀도 (Precision): 시작점과 종료점의 타임스탬프 오차는 0.5초 이내여야 함.

사용 라이브러리: OpenCV (cv2), 영상 픽셀 배열 연산을 위한 numpy. (필요시 프레임 추출을 위해 ffmpeg-python 등 활용 가능)

2. 핵심 함수 정의 (Core Functions)
check_tripartite(timestamp: float) -> bool

주어진 timestamp의 프레임을 하나 추출.

화면을 가로 기준 3등분(Left, Center, Right)하여 Numpy 배열로 나눔.

Center와 Left, Center와 Right의 픽셀 차이(MSE 또는 MAE 등)를 계산.

오차가 임계값(Threshold) 이하면 True(삼분할 맞음), 아니면 False(일반 화면) 반환.

find_edge(start_t, end_t, edge_type)

이진 탐색(Binary Search)을 사용하여 정확한 경계(0.5초 정밀도)를 찾는 함수.

edge_type == 'start': False -> True로 변하는 Rising Edge 탐색.

edge_type == 'end': True -> False로 변하는 Falling Edge 탐색.

3. 하이브리드 탐색 알고리즘 수도코드 (Pseudo-code)
Python
# Constants
STEP_SIZE = 30.0      # 최소 유지 시간이 30초이므로, 누락을 막기 위한 최대 스킵 간격
PRECISION = 0.5       # 이진 탐색 종료 조건 (0.5초 단위 정밀도)

def find_tripartite_sections(video_path):
    total_duration = get_video_duration(video_path)
    current_t = 0.0
    is_in_section = False
    
    sections = [] # (start_time, end_time) 튜플을 저장할 리스트
    current_start = 0.0

    while current_t <= total_duration:
        state = check_tripartite(current_t)
        
        # Case 1: 진입점(Rising Edge) 발견 (False -> True)
        if not is_in_section and state is True:
            # current_t - STEP_SIZE 와 current_t 사이에 진짜 시작점이 있음
            search_left = max(0.0, current_t - STEP_SIZE)
            search_right = current_t
            
            # Binary Search for Start Edge
            while (search_right - search_left) > PRECISION:
                mid_t = (search_left + search_right) / 2.0
                if check_tripartite(mid_t) is True:
                    search_right = mid_t # 시작점은 더 앞쪽에 있음
                else:
                    search_left = mid_t  # 시작점은 더 뒤쪽에 있음
                    
            current_start = search_right
            is_in_section = True
            
            # 최소 30초는 유지되므로, 다음 탐색 위치를 확 건너뜀
            current_t = current_start + STEP_SIZE
            continue

        # Case 2: 이탈점(Falling Edge) 발견 (True -> False)
        elif is_in_section and state is False:
            # current_t - STEP_SIZE 와 current_t 사이에 진짜 종료점이 있음
            search_left = current_t - STEP_SIZE
            search_right = current_t
            
            # Binary Search for End Edge
            while (search_right - search_left) > PRECISION:
                mid_t = (search_left + search_right) / 2.0
                if check_tripartite(mid_t) is True:
                    search_left = mid_t  # 종료점은 더 뒤쪽에 있음
                else:
                    search_right = mid_t # 종료점은 더 앞쪽에 있음
                    
            current_end = search_left
            sections.append((current_start, current_end))
            
            # 로그 출력 (원하는 포맷으로: HH:MM:SS.mmm)
            print_log(current_start, current_end)
            
            is_in_section = False
            # 이탈했으므로 다시 STEP_SIZE 간격으로 탐색 재개
            current_t += STEP_SIZE
            continue

        # 상태 변화가 없으면 다음 Step으로 전진
        current_t += STEP_SIZE
        
    return sections
4. 구현 시 주의사항 및 요구사항
긴 러닝타임의 .ts 파일에서 cv2.VideoCapture의 .set(cv2.CAP_PROP_POS_MSEC, ms)를 이용한 타임스탬프 점프가 정확하지 않거나 느릴 수 있습니다. 이 점을 고려하여 타임스탬프 탐색의 신뢰성을 확보할 수 있는 코드(예: 필요시 ffmpeg 서브프로세스 호출 등)로 작성해 주세요.

터미널에 실시간으로 탐색 진행 상황과 발견된 구간(00:32:30.000 ~ 00:34:30.000 형태)을 Print 해주세요.