"""
tripartite_section_check.py 미처리 .ts 큐를 순차 실행합니다.
Ctrl+C 시 현재 파일만 중단하며, 완료 목록(done)에는 반영하지 않습니다.
이번 실행에서 이미 기록된 완료 파일 목록·누적 개수·상태 파일 경로를 출력합니다.
exit code 0일 때만 해당 파일을 기록합니다.
같은 폴더에 <입력파일stem>_merged.mp4 가 있으면 완료로 보고 해당 .ts 는 스킵합니다.

사용 예 (스캔 폴더는 --scan-dir 로 지정; positional + REMAINDER 를 같이 쓰면 --state 등이
  하위 스크립트로 넘어가 버리므로 이렇게 호출해야 함):
  python tripartite_batch_queue.py --scan-dir "E:\\ffmpeg" ^
    --state "E:\\ffmpeg\\tripartite_done.txt" ^
    --glob "m0m099_*.ts" ^
    -- --workers 4 --coarse-interval 30 --coarse-scale 640 ^
       --merge-workers 4 --merge-notify all --segments-out --merge --edge-fallback

  (-- 이후 인자는 tripartite_section_check.py에 그대로 전달, 입력 파일 경로는 이 스크립트가 붙임)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


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
    candidates = list_candidates(scan_dir, args.glob_pat)
    in_done = {p for p in candidates if _norm_key(p) in done}
    merged_skip = {p for p in candidates if p not in in_done and has_merged_output(p)}
    pending = [p for p in candidates if p not in in_done and p not in merged_skip]

    print(
        f"스캔: {scan_dir}  패턴: {args.glob_pat!r}\n"
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
