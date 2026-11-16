"""The tests must not touch the repository that runs them, even when a hook or a
rebase has exported GIT_DIR (it points at the linked worktree's own directory)."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLEAN = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
IDENTITY = ["-c", "user.name=t", "-c", "user.email=t@example.com"]


def git(*args, cwd):
    done = subprocess.run(
        ["git", *args], cwd=cwd, env=CLEAN, check=True, capture_output=True, text=True
    )
    return done.stdout.strip()


def test_formatter_tests_leave_the_enclosing_repository_alone(tmp_path):
    main = tmp_path / "main"
    main.mkdir()
    git("init", cwd=main)
    git(*IDENTITY, "commit", "--allow-empty", "-m", "first", cwd=main)
    linked = tmp_path / "linked"
    git("worktree", "add", "-b", "other", str(linked), cwd=main)
    gitdir = main / ".git" / "worktrees" / "linked"
    inherited = {
        **CLEAN,
        "GIT_DIR": str(gitdir),
        "GIT_INDEX_FILE": str(gitdir / "index"),
        "PYTHONPATH": str(ROOT / "src"),
    }
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
        + [str(ROOT / "tests/core/tools/test_cxx_format.py")]
        + ["-k", "listing or ignore_file"],
        cwd=linked,
        env=inherited,
        check=True,
        capture_output=True,
    )
    assert git("config", "--get", "core.bare", cwd=main) == "false"
    assert git("status", "--porcelain", cwd=linked) == ""
