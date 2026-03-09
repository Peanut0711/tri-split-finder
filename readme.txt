python .\tripartite_section_check.py "F:\세경\wannabe33_20251228_210529_란제리.ts" --cuda --verify 02:21:48.357 02:24:00.679 --verify-export F:\세경\verify_export --verify-export-only-x --verify-export-max 0  

python .\tripartite_section_check.py "F:\세경\wannabe33_20251228_210529_란제리.ts" --cuda --workers 16

python tripartite_section_check.py "F:\세경\video.ts" --cuda --segments-out "F:\세경\segments.txt"

# 10번 구간만 빼고 병합하려면: segments.txt에서 10번째 구간 줄을 삭제하거나 # 처리
python tripartite_section_check.py "F:\세경\video.ts" --segments-in "F:\세경\segments.txt" --merge

# 1) 구간 검출 후 같은 폴더에 video_seg.txt 저장
python tripartite_section_check.py "F:\세경\video.ts" --cuda --segments-out

# 2) video_seg.txt 편집 후, 인자 없이 같은 폴더의 video_seg.txt 로 병합
python tripartite_section_check.py "F:\세경\video.ts" --segments-in --merge

python tripartite_section_check.py "\\192.168.8.113\ffmpeg\m0m099_20260305_190116.ts" --cuda --workers 16 --coarse-scale 640 --segments-out   

python tripartite_section_check.py "\\192.168.8.113\ffmpeg\m0m099_20260305_190116.ts" --cuda --workers 16 --coarse-scale 640 --segments-in --merge

# 기본(0): 고정 1/3·2/3만 사용
python tripartite_section_check.py input.ts

# 경계를 ±30픽셀 탐색
python tripartite_section_check.py input.ts --align-tolerance 30