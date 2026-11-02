import argparse
import pickletools
import re
import sys
import zipfile
from pathlib import Path


def checkpoint_progress(payload: bytes) -> dict[str, int]:
    progress = {}
    pending = None
    state_keys = set()
    for opcode, value, _ in pickletools.genops(payload):
        if pending is not None:
            if opcode.name in {"BINPUT", "LONG_BINPUT", "PUT", "MEMOIZE"}:
                continue
            if opcode.name not in {"INT", "BININT", "BININT1", "BININT2", "LONG", "LONG1", "LONG4"}:
                raise ValueError(f"Non-integer checkpoint field: {pending}")
            progress[pending] = int(value)
            pending = None
        elif opcode.name in {"UNICODE", "BINUNICODE", "SHORT_BINUNICODE", "BINUNICODE8"}:
            if value in {"model", "optimizer"}:
                state_keys.add(value)
            if value in {"epoch", "step_in_epoch"}:
                pending = value
    if "epoch" not in progress or state_keys != {"model", "optimizer"}:
        raise ValueError("Checkpoint lacks epoch, model or optimizer state")
    return progress


def select_completed_epoch(output_dir: Path, min_epoch: int = 0) -> tuple[Path, int]:
    candidates = []
    for path in output_dir.glob("checkpoint-*.pth"):
        match = re.fullmatch(r"checkpoint-([0-9]+)\.pth", path.name)
        if match and int(match[1]) >= min_epoch:
            candidates.append((int(match[1]), path))
    for epoch, path in sorted(candidates, reverse=True):
        try:
            with zipfile.ZipFile(path) as archive:
                members = [name for name in archive.namelist()
                           if name == "data.pkl" or name.endswith("/data.pkl")]
                if len(members) != 1:
                    raise ValueError("Expected one checkpoint metadata member")
                # Inspect pickle opcodes without executing globals or reading tensor storage.
                progress = checkpoint_progress(archive.read(members[0]))
            if progress["epoch"] != epoch or progress.get("step_in_epoch", 0) != 0:
                raise ValueError("Not a matching completed-epoch checkpoint")
            return path, epoch
        except (OSError, ValueError, EOFError, RuntimeError, zipfile.BadZipFile) as error:
            print(f"Skipping {path.name}: {error}", file=sys.stderr)
    raise FileNotFoundError(
        f"No completed epoch >= {min_epoch} in {output_dir}; refusing to start from scratch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--min-epoch", type=int, default=0)
    args = parser.parse_args()
    path, epoch = select_completed_epoch(args.output_dir, args.min_epoch)
    print(f"Epoch-boundary resume: saved epoch {epoch}, next epoch {epoch + 1}",
          file=sys.stderr)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())