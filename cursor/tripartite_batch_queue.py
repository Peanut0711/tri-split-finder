"""
tripartite_section_check.py 미처리 .ts 큐를 순차 실행합니다.
Ctrl+C 시 현재 파일만 중단하며, 완료 목록(done)에는 반영하지 않습니다.
이번 실행에서 이미 기록된 완료 파일 목록·누적 개수·상태 파일 경로를 출력합니다.
exit code 0일 때만 해당 파일을 기록합니다.
같은 폴더에 <입력파일stem>_merged.mp4 가 있으면 완료로 보고 해당 .ts 는 스킵합니다.
파일명에 merged 가 포함된 .ts(예: …-cut-merged-….ts)는 산출물로 보고 스킵합니다(대소문자 무시).

사용 예 (스캔 폴더는 --scan-dir 로 지정; positional + REMAINDER 를 같이 쓰면 --state 등이
  하위 스크립트로 넘어가 버리므로 이렇게 호출해야 함):
  python tripartite_batch_queue.py --scan-dir "E:\\ffmpeg" ^
    --state "E:\\ffmpeg\\tripartite_done.txt" ^
    --glob "m0m099_*.ts" ^
    --since-date 20240702 ^
    -- --workers 4 --coarse-interval 30 --coarse-scale 640 ^
       --merge-workers 4 --merge-notify all --segments-out --merge --edge-fallback

  (-- 이후 인자는 tripartite_section_check.py에 그대로 전달, 입력 파일 경로는 이 스크립트가 붙임)
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

# 파일명 안의 m0m099_20240702_220400… 처럼 _YYYYMMDD_ 토큰에서 날짜 추출
_DATE_IN_NAME = re.compile(r"_(\d{8})_")


def parse_since_date_arg(value: str) -> date:
    """YYYYMMDD 또는 ISO(2024-07-02) 형식."""
    s = value.strip()
    if re.fullmatch(r"\d{8}", s):
        return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    return date.fromisoformat(s)


def embedded_date_in_ts_name(path: Path) -> date | None:
    m = _DATE_IN_NAME.search(path.name)
    if not m:
        return None
    d = m.group(1)
    try:
        return date(int(d[:4]), int(d[4:6]), int(d[6:8]))
    except ValueError:
        return None


def _norm_key(p: Path) -> str:
    try:
        return str(p.resolve())
    except OSError:
        return str(p)


def load_done(state_path: Path) -> set[str]:
    if not state_path.is_file():
        return set()
    out: set[str] = set()
    for line in state_path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.add(s)
    return out


def append_done(state_path: Path, path_key: str) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("a", encoding="utf-8") as f:
        f.write(path_key + "\n")


def list_candidates(scan_dir: Path, glob_pat: str) -> list[Path]:
    if not scan_dir.is_dir():
        raise SystemExit(f"스캔 폴더가 없습니다: {scan_dir}")
    files = sorted(scan_dir.glob(glob_pat), key=lambda p: p.name.lower())
    return [p for p in files if p.is_file()]


def merged_output_path(ts: Path) -> Path:
    """tripartite --merge 등으로 나오는 관례적 산출물: <stem>_merged.mp4"""
    return ts.parent / f"{ts.stem}_merged.mp4"


def has_merged_output(ts: Path) -> bool:
    return merged_output_path(ts).is_file()


def ts_name_contains_merged(path: Path) -> bool:
    """파일명에 merged 포함 시 True (이미 병합·후처리 산출물로 간주, 대소문자 무시)."""
    return "merged" in path.name.lower()


def run_one(
    python_exe: Path,
    script: Path,
    ts_file: Path,
    tripartite_argv: list[str],
) -> int:
    cmd = [str(python_exe), str(script), str(ts_file), *tripartite_argv]
    print(f"\n>>> 실행 ({ts_file.name})\n    {' '.join(cmd[:3])} ... (+{len(tripartite_argv)} opts)\n", flush=True)
    proc = subprocess.Popen(cmd)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        raise


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="tripartite_section_check 큐 배치 (Ctrl+C 안전 중단, 성공 시만 done 기록)",
    )
    parser.add_argument(
        "-d",
        "--scan-dir",
        type=Path,
        required=True,
        help="ts 파일이 있는 폴더 (예: E:\\ffmpeg). positional 이 아님 — REMAINDER 와 충돌 방지",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=None,
        help="완료 기록 파일 (기본: <scan_dir>/tripartite_done.txt)",
    )
    parser.add_argument(
        "--glob",
        dest="glob_pat",
        default="m0m099_*.ts",
        help='glob 패턴 (기본: "%(default)s")',
    )
    parser.add_argument(
        "--script",
        type=Path,
        default=here / "tripartite_section_check.py",
        help="tripartite_section_check.py 경로",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="사용할 python.exe (기본: 현재 인터프리터)",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="한 파일이라도 실패(exit!=0)하면 큐 중단",
    )
    parser.add_argument(
        "--since-date",
        type=parse_since_date_arg,
        default=None,
        metavar="DATE",
        help="파일명 속 _YYYYMMDD_ 기준으로, 해당 날짜 이후(당일 포함)만 큐에 넣음. "
        "형식: 20240702 또는 2024-07-02. 이름에서 날짜를 못 읽으면 필터에서 제외하지 않음",
    )
    parser.add_argument(
        "tripartite_args",
        nargs=argparse.REMAINDER,
        help="tripartite_section_check.py 인자 (-- 로 구분)",
    )

    args = parser.parse_args()
    scan_dir = args.scan_dir
    state_path = args.state or (scan_dir / "tripartite_done.txt")
    tripartite_argv = args.tripartite_args
    if tripartite_argv and tripartite_argv[0] == "--":
        tripartite_argv = tripartite_argv[1:]

    if not args.script.is_file():
        raise SystemExit(f"스크립트 없음: {args.script}")

    done = load_done(state_path)
    raw_list = list_candidates(scan_dir, args.glob_pat)
    since = args.since_date
    if since is not None:
        after_date = []
        for p in raw_list:
            emb = embedded_date_in_ts_name(p)
            if emb is None or emb >= since:
                after_date.append(p)
        date_excluded = len(raw_list) - len(after_date)
    else:
        after_date = raw_list
        date_excluded = 0

    candidates = [p for p in after_date if not ts_name_contains_merged(p)]
    name_merged_excluded = len(after_date) - len(candidates)

    in_done = {p for p in candidates if _norm_key(p) in done}
    merged_skip = {p for p in candidates if p not in in_done and has_merged_output(p)}
    pending = [p for p in candidates if p not in in_done and p not in merged_skip]

    date_line = ""
    if since is not None:
        date_line = (
            f"날짜 필터: --since-date {since.isoformat()} "
            f"(glob 일치 {len(raw_list)}개 → 이후·파싱불가 포함 {len(after_date)}개, "
            f"이전만 제외 {date_excluded}개)\n"
        )
    print(
        f"스캔: {scan_dir}  패턴: {args.glob_pat!r}\n"
        f"{date_line}"
        f"파일명에 merged 포함 제외: {name_merged_excluded}개\n"
        f"후보 {len(candidates)}개, 상태파일 완료 {len(in_done)}개, "
        f"_merged.mp4 있어 스킵 {len(merged_skip)}개, 남음 {len(pending)}개\n"
        f"상태 파일: {state_path}\n",
        flush=True,
    )
    if not pending:
        print("처리할 파일이 없습니다.", flush=True)
        return

    session_completed: list[str] = []

    def _print_interrupt_summary() -> None:
        print("\n--- Ctrl+C 중단 ---", flush=True)
        print(
            "현재 돌리던 파일은 완료로 기록하지 않습니다. 다음 실행 시 같은 파일부터 이어집니다.\n",
            flush=True,
        )
        if session_completed:
            print(f"이번 실행에서 이미 완료 기록된 파일 ({len(session_completed)}개):", flush=True)
            for p in session_completed:
                print(f"  {p}", flush=True)
        else:
            print("이번 실행에서 새로 완료 기록된 파일은 없습니다.", flush=True)
        print(
            f"\n누적 완료 항목 수(메모리 기준, 상태 파일과 동일): {len(done)}개\n"
            f"상태 파일: {state_path}",
            flush=True,
        )

    interrupted = False
    for i, ts in enumerate(pending, start=1):
        key = _norm_key(ts)
        print(f"[{i}/{len(pending)}] {ts.name}", flush=True)
        try:
            code = run_one(args.python, args.script, ts, tripartite_argv)
        except KeyboardInterrupt:
            interrupted = True
            _print_interrupt_summary()
            sys.exit(130)
        if code != 0:
            print(f"실패 (exit {code}): {ts} — done에 추가하지 않음\n", flush=True, file=sys.stderr)
            if args.stop_on_error:
                sys.exit(code)
            continue
        if key not in done:
            append_done(state_path, key)
            done.add(key)
            session_completed.append(key)
        print(f"완료 기록: {key}\n", flush=True)

    if not interrupted:
        print("큐 전부 처리했습니다.", flush=True)


if __name__ == "__main__":
    main()
