python .\tripartite_section_check.py "F:\세경\wannabe33_20251228_210529_란제리.ts" --cuda --verify 02:21:48.357 02:24:00.679 --verify-export F:\세경\verify_export --verify-export-only-x --verify-export-max 0  

python .\tripartite_section_check.py "F:\세경\wannabe33_20251228_210529_란제리.ts" --cuda --workers 16

python tripartite_section_check.py "F:\세경\video.ts" --cuda --segments-out "F:\세경\segments.txt"

# 10번 구간만 빼고 병합하려면: segments.txt에서 10번째 구간 줄을 삭제하거나 # 처리
python tripartite_section_check.py "F:\세경\video.ts" --segments-in "F:\세경\segments.txt" --merge