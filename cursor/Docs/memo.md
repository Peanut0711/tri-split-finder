# 지정 구간 검증 + verify_export 폴더에 x축만, max 0으로 내보내기
python .\tripartite_section_check.py "input.ts" --cuda --verify 02:21:48.357 02:24:00.679 --verify-export verify_export --verify-export-only-x --verify-export-max 0

# CUDA로 구간 검출, 워커 16개
python .\tripartite_section_check.py "input.ts" --cuda --workers 16

# 구간 검출 후 결과를 segments.txt에 저장
python tripartite_section_check.py "input.ts" --cuda --segments-out "segments.txt"

# 10번 구간만 빼고 병합하려면: segments.txt에서 10번째 구간 줄을 삭제하거나 # 처리
python tripartite_section_check.py "input.ts" --segments-in "segments.txt" --merge

# 구간 검출 후 같은 폴더에 video_seg.txt 저장
python tripartite_section_check.py "input.ts" --cuda --segments-out

# video_seg.txt 편집 후, 인자 없이 같은 폴더의 video_seg.txt 로 병합
python tripartite_section_check.py "input.ts" --segments-in --merge

# coarse-scale 640으로 구간 검출 후 같은 폴더에 segments 저장
python tripartite_section_check.py "input.ts" --cuda --workers 16 --coarse-scale 640 --segments-out

# coarse-scale 640 + 저장된 segments로 병합
python tripartite_section_check.py "input.ts" --cuda --workers 16 --coarse-scale 640 --segments-in --merge

# 기본(0): 고정 1/3·2/3만 사용
python tripartite_section_check.py input.ts

# 경계를 ±30픽셀 탐색
python tripartite_section_check.py input.ts --align-tolerance 30
