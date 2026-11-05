"""Check explicitly selected publication paths without changing repository state."""

import argparse
import json
import posixpath
import re
import subprocess
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

PRIVATE_ROOTS = {
    ".local",
    "data",
    "results",
    "Colab Notebooks",
    ".hallmark",
    ".git",
    ".venv",
    "build",
    "dist",
    "__pycache__",
}
ARTIFACT_SUFFIXES = {
    ".ubj",
    ".pt",
    ".pth",
    ".safetensors",
    ".faiss",
    ".parquet",
    ".npy",
    ".npz",
    ".zip",
    ".tar",
    ".gz",
    ".whl",
    ".bz2",
    ".xz",
    ".7z",
    ".pkl",
    ".pickle",
    ".onnx",
    ".bin",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".pyc",
    ".pem",
    ".key",
}
SECRET = re.compile(
    r"\bgh[pousr]_[A-Za-z0-9]{36,}\b|\bgithub_pat_[A-Za-z0-9_]{60,}\b"
    r"|\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{40,}\b"
    r"|-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"
)
MARKDOWN_LINK = re.compile(r"\]\((<[^>]+>|[^)]+)\)")


def check_repository(root: Path, files: list[str]) -> list[str]:
    """Return path-scoped issues, reading only eligible regular source files."""
    issues: list[str] = []
    candidates = set(files)
    root = root.resolve()
    for name in sorted(candidates):
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name or not name:
            issues.append(f"{name}: unsafe publication path")
            continue
        if (
            relative.parts[0] in PRIVATE_ROOTS
            or name.startswith("docs/superpowers/")
            or (name.startswith("docs/reviews/") and relative.suffix == ".json")
            or relative.suffix.lower() in ARTIFACT_SUFFIXES
            or (relative.name.startswith(".env") and relative.name != ".env.example")
        ):
            issues.append(f"{name}: private, generated, artifact or credential path")
            continue
        path = root / name
        if any(parent.is_symlink() for parent in (path, *path.parents) if parent != root):
            issues.append(f"{name}: symlinked publication path")
            continue
        if not path.is_file():
            issues.append(f"{name}: candidate file is missing")
            continue
        if path.stat().st_size > 2 * 1024**2:
            issues.append(f"{name}: file exceeds 2 MiB")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeError, OSError):
            issues.append(f"{name}: unreadable source text")
            continue
        if SECRET.search(text):
            issues.append(f"{name}: potential secret token or private key")
        if relative.suffix == ".ipynb":
            try:
                notebook = json.loads(text)
                cells = notebook["cells"]
                if not isinstance(cells, list):
                    raise ValueError("Expected notebook cells")
                for cell in cells:
                    if cell.get("outputs") or cell.get("execution_count") is not None:
                        issues.append(f"{name}: notebook outputs or execution count retained")
                        break
            except (ValueError, KeyError, TypeError, AttributeError):
                issues.append(f"{name}: invalid notebook JSON")
        if relative.suffix.lower() == ".md":
            for match in MARKDOWN_LINK.finditer(text):
                target = match[1].strip().split(' "', 1)[0].strip("<>")
                parsed = urlsplit(target)
                if parsed.scheme or parsed.netloc or not parsed.path:
                    continue
                target_path = unquote(parsed.path)
                destination = posixpath.normpath(str(relative.parent / target_path))
                if destination in candidates or any(
                    item.startswith(destination.rstrip("/") + "/") for item in candidates
                ):
                    continue
                issues.append(f"{name}: local link target absent from publication: {target}")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--files-from", type=Path)
    args = parser.parse_args()
    if args.files_from:
        files = args.files_from.read_text(encoding="utf-8").splitlines()
    else:
        result = subprocess.run(
            ["git", "ls-files", "-z"], cwd=args.root, check=True, capture_output=True, text=True
        )
        files = result.stdout.rstrip("\0").split("\0") if result.stdout else []
    issues = check_repository(args.root, files)
    for issue in issues:
        print(issue)
    print(f"Checked {len(files)} candidate files, {len(issues)} issues")
    return int(bool(issues))


if __name__ == "__main__":
    raise SystemExit(main())
