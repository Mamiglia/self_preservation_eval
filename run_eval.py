# ---------- Init ------------------------

import os
import random
import re
import sys
from functools import partial, wraps
from pathlib import Path
from pprint import pprint
from typing import Any, Literal
import time
import types

print("[run_eval] starting with interpreter:", sys.executable, flush=True)

def retry_with_exponential_backoff(
    func=None,
    *,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    max_retries: int = 6,
    multiplier: float = 2.0,
    jitter: float = 0.2,
    retry_exceptions: tuple = (Exception,),
):
    """Decorator to retry a function with exponential backoff and jitter."""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            attempt = 0
            while True:
                try:
                    return f(*args, **kwargs)
                except retry_exceptions:
                    if attempt >= max_retries:
                        raise
                    jitter_amount = delay * jitter * (2 * random.random() - 1)
                    time.sleep(max(0.0, delay + jitter_amount))
                    delay = min(max_delay, delay * multiplier)
                    attempt += 1
        return wrapper
    if func is not None:
        return decorator(func)
    return decorator

# Shim out problematic ARENA import that relies on Path.cwd() discovery at import time
sys.modules.setdefault(
    'part1_intro_to_evals.solutions',
    types.SimpleNamespace(retry_with_exponential_backoff=retry_with_exponential_backoff)
)

from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI

chapter = "chapter3_llm_evals"
section = "part2_dataset_generation"

# Try resolving ARENA repo root in multiple ways
root_dir = None

# 1) Explicit env var
arena_root_env = os.getenv("ARENA_ROOT")
if arena_root_env:
    p = Path(arena_root_env).expanduser().resolve()
    if (p / chapter).exists():
        root_dir = p

# 2) Search upwards from CWD
if root_dir is None:
    candidates = (Path.cwd(), *Path.cwd().parents)
    root_dir = next((p for p in candidates if (p / chapter).exists()), None)

# 3) Search upwards from script dir
here = Path(__file__).resolve().parent
if root_dir is None:
    candidates = (here, *here.parents)
    root_dir = next((p for p in candidates if (p / chapter).exists()), None)

# 4) Common default location for this machine
if root_dir is None:
    common = Path("/Users/joaquinpereirapizzini/ARENA_3.0")
    if (common / chapter).exists():
        root_dir = common

if root_dir is None:
    raise FileNotFoundError(
        "Could not locate the ARENA repo root.\n"
        f"- Looked for a folder named '{chapter}' above:\n"
        f"  • CWD: {Path.cwd()}\n"
        f"  • Script dir: {here}\n"
        f"- Also checked ARENA_ROOT env: {arena_root_env!r}\n\n"
        "Fix by either:\n"
        "  1) Run inside the ARENA_3.0 repo, or\n"
        "  2) Export ARENA_ROOT=/path/to/ARENA_3.0, or\n"
        "  3) Place the repo at /Users/joaquinpereirapizzini/ARENA_3.0\n"
    )

# Ensure CWD is inside the chapter directory so modules that use Path.cwd() work
chapter_dir = root_dir / chapter
try:
    os.chdir(chapter_dir)
    print(f"[run_eval] chdir -> {chapter_dir}")
except Exception as e:
    raise RuntimeError(f"Failed to change working directory to {chapter_dir}: {e}")

exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))
import part2_dataset_generation.tests as tests
from part2_dataset_generation.utils import pretty_print_questions

MAIN = __name__ == "__main__"

# --------- Init --------------
# --------- keys -----------------

if MAIN:
    # Load .env from repo root if present, else fall back to CWD
    repo_env = (root_dir / '.env') if root_dir else None
    if repo_env and repo_env.exists():
        load_dotenv(dotenv_path=repo_env)
    else:
        load_dotenv()
    # Debug banner to confirm paths
    print(f"[run_eval] CWD={Path.cwd()}")
    print(f"[run_eval] root_dir={root_dir}")
    print(f"[run_eval] loaded .env from: {repo_env if repo_env and repo_env.exists() else 'CWD search'}")

    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    print(f"[run_eval] OPENAI_API_KEY present: {bool(openai_key)}", flush=True)
    print(f"[run_eval] ANTHROPIC_API_KEY present: {bool(anthropic_key)}", flush=True)

    # Initialize clients with visible diagnostics
    try:
        if openai_key:
            openai_client = OpenAI()
            print("[run_eval] OpenAI client OK", flush=True)
        else:
            openai_client = None
            print("[run_eval] Skipping OpenAI client init (no OPENAI_API_KEY)", flush=True)
    except Exception as e:
        print(f"[run_eval] OpenAI client init failed: {e!r}", flush=True)
        openai_client = None

    try:
        if anthropic_key:
            anthropic_client = Anthropic()
            print("[run_eval] Anthropic client OK", flush=True)
        else:
            anthropic_client = None
            print("[run_eval] Skipping Anthropic client init (no ANTHROPIC_API_KEY)", flush=True)
    except Exception as e:
        print(f"[run_eval] Anthropic client init failed: {e!r}", flush=True)
        anthropic_client = None

    print(f"[run_eval] __file__={__file__}", flush=True)
    print(f"[run_eval] interpreter={sys.executable}", flush=True)
    print("[run_eval] proceeding to inspect block...", flush=True)

# -------------------------

# ------- inspect----------
print("[run_eval] entering inspect section", flush=True)

try:
    print("[run_eval] importing inspect_ai...", flush=True)
    # Import inspect-ai with a distinct alias to avoid shadowing builtins.eval
    from inspect_ai import Task, eval as inspect_eval, task
    from inspect_ai.dataset import example_dataset
    from inspect_ai.scorer import model_graded_fact
    from inspect_ai.solver import chain_of_thought, generate, self_critique

    print("[run_eval] inspect_ai imported", flush=True)

    @task
    def custom_dataset_task() -> Task:
        return Task(
            dataset=example_dataset(str(section_dir / "dataset.json")),
            solver=[chain_of_thought(), generate(), self_critique(model="openai/gpt-4o-mini")],
            scorer=model_graded_fact(model="openai/gpt-4o-mini"),
        )

    print("[run_eval] starting inspect-ai eval…")
    log = inspect_eval(
        custom_dataset_task(),
        model="openai/gpt-4o-mini",
        limit=10,
        log_dir=str(section_dir / "logs"),
    )
    print("[run_eval] eval finished. Logs at:", section_dir / "logs")

    # Try to show something useful from log
    try:
        summary = getattr(log, "summary", None)
        if callable(summary):
            print("[run_eval] summary:\n", summary())
        else:
            print("[run_eval] log object:", log)
    except Exception as e:
        print("[run_eval] could not pretty-print log:", e)

except ModuleNotFoundError as e:
    print("[run_eval] Missing package:", e)
    print("[run_eval] Install into THIS interpreter and retry:")
    print("    python -m pip install inspect-ai  "
          "||  python -m pip install git+https://github.com/openai/inspect_ai.git")
except Exception as e:
    import traceback
    print("[run_eval] Unexpected error while running inspect task:")
    traceback.print_exc()
# ------------------------

print("[run_eval] end of script", flush=True)
