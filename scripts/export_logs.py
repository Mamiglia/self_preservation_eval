"""Copy every .eval log under logs/ to export/logs/, stamping the canary into each log's
eval metadata, and write export/CANARY.txt. The originals are left untouched.

    python scripts/export_logs.py [--src logs] [--dst export/logs]
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tbsp.canary import CANARY  # noqa: E402

from inspect_ai.log import read_eval_log, write_eval_log  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="logs")
    p.add_argument("--dst", default="export/logs")
    args = p.parse_args()
    src, dst = Path(args.src), Path(args.dst)

    files = sorted(src.rglob("*.eval"))
    dst.mkdir(parents=True, exist_ok=True)
    (dst.parent / "CANARY.txt").write_text(CANARY + "\n")

    for i, f in enumerate(files, 1):
        out = dst / f.relative_to(src)
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        log = read_eval_log(str(f))
        log.eval.metadata = {**(log.eval.metadata or {}), "canary": CANARY}
        write_eval_log(log, str(out))
        print(f"[{i}/{len(files)}] {out}", flush=True)


if __name__ == "__main__":
    main()
