#!/usr/bin/env python3
"""
Creates a private GitHub repo that mirrors the current repo plus all
gitignored data files and the /share/garg/arxiv_kaggle/ parquet/csv data.

Files > MAX_FILE_SIZE (100 MB) are skipped; this avoids GitHub's hard per-file
limit.  Data files are pushed in batches of PUSH_BATCH_SIZE to stay under
GitHub's per-push size limit (avoids HTTP 500 on large single pushes).

The original repo / main branch is never touched.

Auth: set GITHUB_TOKEN env var, or the token is read from the existing
      git remote URL if it is in the form https://<token>@github.com/...

Usage:
    python create_private_data_fork.py [--repo-name REPO_NAME] [--skip-secrets]
                                       [--tmp-dir DIR] [--skip-repo-create]
                                       [--batch-size N]
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *, desc="", total=None, **kwargs):
        n = total if total is not None else (len(iterable) if hasattr(iterable, "__len__") else "?")
        print(f"{desc}: {n} items…")
        return iterable

# ── config ──────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent.resolve()
DATA_EXTENSIONS = {".csv", ".parquet", ".json", ".pdf", ".pickle", ".pkl",
                   ".md", ".tex", ".npz"}
DATA_PATTERNS_TO_REMOVE = {"*.csv", "*.json", "*.parquet", "*.pdf",
                            "*.pickle", "*.pkl", "*.md", "*.tex", "*.npz"}
ARXIV_KAGGLE_DIR = Path("/share/garg/arxiv_kaggle")
MAX_FILE_SIZE = 100 * 1024 * 1024   # 100 MB — GitHub's hard per-file limit
DEFAULT_FORK_NAME = "llm_detection_data"
DEFAULT_PUSH_BATCH = 500            # files per commit/push to avoid HTTP 500

# ── helpers ──────────────────────────────────────────────────────────────────

def run(cmd, cwd=None, check=True):
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"\nERROR running: {' '.join(str(c) for c in cmd)}", file=sys.stderr)
        print(result.stderr.strip(), file=sys.stderr)
        sys.exit(1)
    return result


def get_token():
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        return token
    r = run(["git", "remote", "get-url", "origin"], cwd=REPO_ROOT, check=False)
    m = re.match(r"https://([^@]+)@github\.com/", r.stdout.strip())
    if m:
        return m.group(1)
    print(
        "ERROR: No GitHub token found.\n"
        "  Set GITHUB_TOKEN or ensure remote URL is https://<token>@github.com/...",
        file=sys.stderr,
    )
    sys.exit(1)


def github_get(path, token):
    req = urllib.request.Request(
        f"https://api.github.com{path}",
        method="GET",
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, {}


def github_post(path, token, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"https://api.github.com{path}",
        data=data,
        method="POST",
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"GitHub API error {e.code} on POST {path}:", file=sys.stderr)
        print(body, file=sys.stderr)
        sys.exit(1)


def find_gitignored_data_files():
    r = run(
        ["git", "ls-files", "--others", "--ignored", "--exclude-standard", "-z"],
        cwd=REPO_ROOT,
    )
    files = [p for p in r.stdout.split("\0") if p]
    return [Path(p) for p in files if Path(p).suffix in DATA_EXTENSIONS]


def find_arxiv_kaggle_files():
    if not ARXIV_KAGGLE_DIR.exists():
        print(f"  WARNING: {ARXIV_KAGGLE_DIR} not found — skipping.")
        return []
    pairs = []
    for ext in (".parquet", ".csv"):
        for f in sorted(ARXIV_KAGGLE_DIR.rglob(f"*{ext}")):
            pairs.append((f, f.relative_to(ARXIV_KAGGLE_DIR)))
    return pairs


def strip_data_patterns_from_gitignore(gitignore_path: Path):
    lines = gitignore_path.read_text().splitlines(keepends=True)
    filtered = [l for l in lines if l.strip() not in DATA_PATTERNS_TO_REMOVE]
    gitignore_path.write_text("".join(filtered))


def human_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def configure_pack(clone_dir):
    run(["git", "config", "pack.windowMemory", "128m"], cwd=clone_dir)
    run(["git", "config", "pack.threads", "1"], cwd=clone_dir)
    run(["git", "config", "http.postBuffer", "524288000"], cwd=clone_dir)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-name", default=DEFAULT_FORK_NAME)
    parser.add_argument("--skip-secrets", action="store_true",
                        help="Auto-skip files whose path contains 'secret'")
    parser.add_argument("--tmp-dir", default=None,
                        help="Parent dir for the temp clone (use /share/garg/... for space)")
    parser.add_argument("--skip-repo-create", action="store_true",
                        help="Skip GitHub repo creation (repo already exists)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_PUSH_BATCH,
                        help=f"Files per commit/push (default {DEFAULT_PUSH_BATCH})")
    args = parser.parse_args()

    token = get_token()

    status, user_info = github_get("/user", token)
    if status != 200:
        print("ERROR: GitHub auth failed. Check your token.", file=sys.stderr)
        sys.exit(1)
    username = user_info["login"]
    print(f"GitHub user: {username}")

    status, _ = github_get(f"/repos/{username}/{args.repo_name}", token)
    if status == 200 and not args.skip_repo_create:
        print(
            f"ERROR: {username}/{args.repo_name} already exists.\n"
            "Pass --skip-repo-create to push to it, or pick a different name.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── scan files ───────────────────────────────────────────────────────────
    print("Scanning for gitignored data files in the repo…")
    data_files = find_gitignored_data_files()

    secret_files = [f for f in data_files if "secret" in str(f).lower()]
    if secret_files:
        print("\nWARNING — possible secret files:")
        for f in secret_files:
            print(f"  {f}")
        if args.skip_secrets:
            data_files = [f for f in data_files if f not in secret_files]
            print("  Auto-skipped (--skip-secrets).")
        else:
            if input("Include them anyway? [y/N] ").strip().lower() != "y":
                data_files = [f for f in data_files if f not in secret_files]
                print("  Excluded.")

    print(f"  {len(data_files)} candidate repo data files.")

    print(f"\nScanning {ARXIV_KAGGLE_DIR} for .parquet / .csv …")
    kaggle_pairs = find_arxiv_kaggle_files()
    print(f"  {len(kaggle_pairs)} candidate arxiv_kaggle files.")

    # ── create GitHub repo ───────────────────────────────────────────────────
    if args.skip_repo_create:
        print(f"\nUsing existing repo: {username}/{args.repo_name}")
    else:
        print(f"\nCreating private repo: {username}/{args.repo_name}")
        github_post("/user/repos", token, {
            "name": args.repo_name,
            "private": True,
            "description": f"Private data fork of {username}/llm_detection",
            "auto_init": False,
        })

    private_url = f"https://{token}@github.com/{username}/{args.repo_name}.git"

    # ── clone → populate → push in batches ──────────────────────────────────
    with tempfile.TemporaryDirectory(prefix="llm_detection_fork_",
                                     dir=args.tmp_dir) as tmpdir:
        clone_dir = Path(tmpdir) / "clone"

        print(f"\nCloning {REPO_ROOT.name} …")
        run(["git", "clone", "--no-local", str(REPO_ROOT), str(clone_dir)])
        run(["git", "remote", "set-url", "origin", private_url], cwd=clone_dir)
        configure_pack(clone_dir)

        # Modify .gitignore and commit it as the very first (tiny) push
        gitignore = clone_dir / ".gitignore"
        if gitignore.exists():
            strip_data_patterns_from_gitignore(gitignore)
        run(["git", "add", ".gitignore"], cwd=clone_dir)
        run(["git", "commit", "-m", "update .gitignore: allow data file extensions"],
            cwd=clone_dir)
        print("Pushing .gitignore commit (sets upstream)…")
        push_flags = ["--set-upstream", "origin", "main"]
        if args.skip_repo_create:
            push_flags = ["--force"] + push_flags  # remote may have partial history from prior run
        run(["git", "push"] + push_flags, cwd=clone_dir)

        # ── copy repo data files ──────────────────────────────────────────
        skipped_size_repo, missing_repo, copied_repo = [], [], 0
        for rel_path in tqdm(data_files, desc="Copying repo data files"):
            src = REPO_ROOT / rel_path
            if not src.exists():
                missing_repo.append(rel_path)
                continue
            size = src.stat().st_size
            if size > MAX_FILE_SIZE:
                skipped_size_repo.append((src, size))
                continue
            dst = clone_dir / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied_repo += 1

        if missing_repo:
            print(f"  WARNING: {len(missing_repo)} file(s) not found on disk (skipped)")
        if skipped_size_repo:
            print(f"  Skipped {len(skipped_size_repo)} file(s) > {human_bytes(MAX_FILE_SIZE)}:")
            for p, sz in skipped_size_repo:
                print(f"    {p.relative_to(REPO_ROOT)}  ({human_bytes(sz)})")
        print(f"  {copied_repo} repo data file(s) copied.")

        # ── copy arxiv_kaggle data files → data/ ──────────────────────────
        skipped_size_kaggle, copied_kaggle = [], 0
        for src, rel in tqdm(kaggle_pairs, desc="Copying arxiv_kaggle files"):
            try:
                size = src.stat().st_size
            except OSError:
                continue
            if size > MAX_FILE_SIZE:
                skipped_size_kaggle.append((src, size))
                continue
            dst = clone_dir / "data" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied_kaggle += 1

        if skipped_size_kaggle:
            print(f"  Skipped {len(skipped_size_kaggle)} arxiv_kaggle file(s) > {human_bytes(MAX_FILE_SIZE)}:")
            for p, sz in skipped_size_kaggle:
                print(f"    {p.relative_to(ARXIV_KAGGLE_DIR)}  ({human_bytes(sz)})")
        print(f"  {copied_kaggle} arxiv_kaggle file(s) copied to data/.")

        # ── get list of all new (untracked) files ─────────────────────────
        result = run(["git", "ls-files", "--others", "-z"], cwd=clone_dir)
        new_files = [p for p in result.stdout.split("\0") if p]
        total_new = len(new_files)

        if total_new == 0:
            print("No new files to commit.")
            sys.exit(0)

        # ── batch-commit and push ─────────────────────────────────────────
        batch_size = args.batch_size
        n_batches = (total_new + batch_size - 1) // batch_size
        print(f"\n{total_new} new file(s) → {n_batches} batch(es) of ≤{batch_size} files each")

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            batch = new_files[start:start + batch_size]
            label = f"Staging batch {batch_idx+1}/{n_batches}"
            for f in tqdm(batch, desc=label):
                run(["git", "add", "-f", "--", f], cwd=clone_dir)
            run(["git", "commit", "-m",
                 f"add data files batch {batch_idx+1}/{n_batches} ({len(batch)} files)"],
                cwd=clone_dir)
            print(f"  Pushing batch {batch_idx+1}/{n_batches} ({len(batch)} files)…")
            run(["git", "push"], cwd=clone_dir)
            print(f"  ✓ batch {batch_idx+1}/{n_batches} done")

    skipped_total = len(skipped_size_repo) + len(skipped_size_kaggle)
    total_copied = copied_repo + copied_kaggle
    print(f"\nDone!  https://github.com/{username}/{args.repo_name}")
    print(f"  {total_copied} file(s) pushed ({copied_repo} repo, {copied_kaggle} arxiv_kaggle)")
    if skipped_total:
        print(f"  {skipped_total} file(s) skipped (>{human_bytes(MAX_FILE_SIZE)})")


if __name__ == "__main__":
    main()
