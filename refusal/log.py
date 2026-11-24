"""Run directories, the journal each script keeps, elapsed-time messages and resumable phases.

Every experiment script writes its own journal (``<out>/journal.log``): stdout and stderr are duplicated into it,
so no shell redirection is ever needed and successive launches stack up with the date and the command as header.
Messages carry the time elapsed since launch (``[0h12m03s]``), not the wall-clock time.
"""
import datetime
import json
import os
import sys
import time
import warnings
from pathlib import Path

import torch


class Tee:
    """Writes both to a stream (the terminal) and to a file (the journal)."""

    def __init__(self, stream, file):
        self.stream, self.file = stream, file

    def write(self, text):
        self.stream.write(text)
        self.stream.flush()
        self.file.write(text)
        self.file.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def isatty(self):
        return getattr(self.stream, "isatty", lambda: False)()


class Clock:
    """``clock("message")`` prints the message prefixed by the time elapsed since the clock was created."""

    def __init__(self):
        self.t0 = time.time()

    def __call__(self, msg):
        dt = time.time() - self.t0
        print(f"[{int(dt // 3600)}h{int(dt % 3600 // 60):02d}m{int(dt % 60):02d}s] {msg}", flush=True)


def quiet():
    """Cut repeated library messages at the source (progress bars, generation warnings, ROCm notice)."""
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    warnings.filterwarnings("ignore", message=".*AMD GPU is still experimental.*")
    warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*")
    warnings.filterwarnings("ignore", message=".*past_key_values.*is deprecated.*")
    try:
        import transformers
        transformers.logging.set_verbosity_error()
        transformers.utils.logging.disable_progress_bar()
    except ImportError:
        pass


def start_run(out_dir, args=None):
    """Create ``out_dir`` (and ``out_dir/figures``), open the journal, duplicate stdout/stderr into it, save the
    arguments. Returns a Clock."""
    out = Path(out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    f = open(out / "journal.log", "a", buffering=1, encoding="utf-8")
    f.write(f"\n{'=' * 78}\n{datetime.datetime.now():%Y-%m-%d %H:%M:%S}  {' '.join(sys.argv)}\n{'=' * 78}\n")
    base_out = sys.stdout.stream if isinstance(sys.stdout, Tee) else sys.stdout
    base_err = sys.stderr.stream if isinstance(sys.stderr, Tee) else sys.stderr
    sys.stdout, sys.stderr = Tee(base_out, f), Tee(base_err, f)
    if args is not None:
        json.dump({**vars(args), "code_commit": code_commit()}, open(out / "arguments.json", "w"), indent=1,
                  default=str)
    quiet()
    return Clock()


def code_commit():
    """Git commit of this repository (with "+modified" if the working tree differs), or None outside git."""
    import subprocess
    root = Path(__file__).resolve().parent.parent
    try:
        c = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                               capture_output=True, text=True).stdout.strip()
        return c.stdout.strip() + ("+modified" if dirty else "")
    except Exception:  # noqa: BLE001
        return None


def redo_set(spec):
    return {x.strip() for x in (spec or "").split(",") if x.strip()}


def phase(name, path, redo, clock, fn):
    """Run ``fn`` and save its result to ``path``, unless the file exists (then reload it). One phase = one file:
    after an interruption, the same command resumes at the phase reached. ``redo`` forces named phases."""
    path = Path(path)
    if path.exists() and name not in redo:
        clock(f"{name}: already done ({path.name}), reloaded")
        return torch.load(path, map_location="cpu", weights_only=False)
    res = fn()
    torch.save(res, path)
    clock(f"{name}: written to {path.name}")
    return res


def figures_written(paths):
    """Final message of every script: where the figures are."""
    paths = [str(p) for p in paths if Path(p).exists()]
    if paths:
        print("figures written:\n  " + "\n  ".join(paths))
