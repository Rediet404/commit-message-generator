"""Collect and quality-filter real Git commit/diff examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class RepositorySpec:
    name: str
    url: str
    license: str
    license_file: str
    license_marker: str


REPOSITORIES = (
    RepositorySpec("psf/requests", "https://github.com/psf/requests.git", "Apache-2.0", "LICENSE", "Apache License"),
    RepositorySpec("pallets/flask", "https://github.com/pallets/flask.git", "BSD-3-Clause", "LICENSE.txt", "Redistribution and use"),
    RepositorySpec("pallets/click", "https://github.com/pallets/click.git", "BSD-3-Clause", "LICENSE.txt", "Redistribution and use"),
    RepositorySpec("pydantic/pydantic", "https://github.com/pydantic/pydantic.git", "MIT", "LICENSE", "MIT License"),
    RepositorySpec("encode/httpx", "https://github.com/encode/httpx.git", "BSD-3-Clause", "LICENSE.md", "Redistribution and use"),
)

DEFAULT_MAX_COMMITS_PER_REPOSITORY = 800
DEFAULT_TARGET_EXAMPLES = 2000
DEFAULT_MAX_DIFF_BYTES = 100_000
UNINFORMATIVE_SUBJECTS = {"update", "updates", "changes", "change", "wip", "fix", "fixed"}
EXCLUDED_PATH = re.compile(
    r"(^|/)(vendor|vendors|node_modules|dist|build|generated|coverage)(/|$)"
    r"|(\.min\.(js|css)$)"
    r"|(^|/)(package-lock\.json|yarn\.lock|pnpm-lock\.yaml|poetry\.lock|Cargo\.lock)$",
    re.IGNORECASE,
)


@dataclass
class CollectionStats:
    commits_inspected: int = 0
    accepted: int = 0
    skipped_merge: int = 0
    skipped_no_parent: int = 0
    skipped_empty_diff: int = 0
    skipped_empty_subject: int = 0
    skipped_uninformative_subject: int = 0
    skipped_binary: int = 0
    skipped_generated_or_vendor: int = 0
    skipped_large_diff: int = 0
    skipped_duplicate: int = 0
    skipped_invalid: int = 0


def _run_git(args: Sequence[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout


def _parse_log(output: str) -> Iterable[dict[str, str]]:
    for record in output.split("\x1e"):
        fields = record.strip("\r\n").split("\x1f", 4)
        if len(fields) != 5 or not fields[0]:
            continue
        commit_sha, parents, commit_date, subject, body = fields
        yield {
            "commit_sha": commit_sha.strip(),
            "parents": parents.strip(),
            "commit_date": commit_date.strip(),
            "commit_subject": subject.strip(),
            "commit_body": body.strip(),
        }


def _is_informative(subject: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", " ", subject.lower()).strip()
    return bool(normalized) and normalized not in UNINFORMATIVE_SUBJECTS


def _changed_paths(repo_path: Path, parent_sha: str, commit_sha: str) -> list[str]:
    output = _run_git(["diff", "--name-only", "--no-renames", parent_sha, commit_sha], repo_path)
    return [line.strip() for line in output.splitlines() if line.strip()]


def _has_binary_change(repo_path: Path, parent_sha: str, commit_sha: str) -> bool:
    output = _run_git(["diff", "--numstat", "--no-renames", parent_sha, commit_sha], repo_path)
    return any(line.startswith("-\t-\t") for line in output.splitlines())


def _complete_diff(repo_path: Path, parent_sha: str, commit_sha: str) -> str:
    return _run_git(
        ["diff", "--binary", "--no-ext-diff", "--no-renames", "--unified=3", parent_sha, commit_sha],
        repo_path,
    )


def _record(spec: RepositorySpec, commit: dict[str, str], diff: str) -> dict[str, str]:
    return {
        "repository": spec.name,
        "repository_url": spec.url.removesuffix(".git"),
        "commit_sha": commit["commit_sha"],
        "parent_sha": commit["parents"].split()[0],
        "commit_date": commit["commit_date"],
        "license": spec.license,
        "diff": diff,
        "commit_subject": commit["commit_subject"],
        "commit_body": commit["commit_body"],
    }


def collect_repository(
    spec: RepositorySpec,
    repo_path: Path,
    *,
    max_commits: int,
    max_diff_bytes: int,
    stats: CollectionStats,
) -> list[dict[str, str]]:
    commits = _parse_log(
        _run_git(
            ["log", "--no-merges", f"-n{max_commits}", "--format=%H%x1f%P%x1f%cI%x1f%s%x1f%B%x1e"],
            repo_path,
        )
    )
    accepted: list[dict[str, str]] = []
    seen_commits: set[tuple[str, str]] = set()
    seen_content: set[str] = set()
    for commit in commits:
        stats.commits_inspected += 1
        parents = commit["parents"].split()
        if len(parents) != 1:
            stats.skipped_merge += 1
            continue
        parent_sha = parents[0]
        identity = (spec.name, commit["commit_sha"])
        if identity in seen_commits:
            stats.skipped_duplicate += 1
            continue
        try:
            paths = _changed_paths(repo_path, parent_sha, commit["commit_sha"])
            if not paths:
                stats.skipped_empty_diff += 1
                continue
            if any(EXCLUDED_PATH.search(path) for path in paths):
                stats.skipped_generated_or_vendor += 1
                continue
            if _has_binary_change(repo_path, parent_sha, commit["commit_sha"]):
                stats.skipped_binary += 1
                continue
            diff = _complete_diff(repo_path, parent_sha, commit["commit_sha"])
        except subprocess.CalledProcessError:
            stats.skipped_no_parent += 1
            continue
        if not diff.strip():
            stats.skipped_empty_diff += 1
            continue
        if len(diff.encode("utf-8")) > max_diff_bytes:
            stats.skipped_large_diff += 1
            continue
        if not commit["commit_subject"]:
            stats.skipped_empty_subject += 1
            continue
        if not _is_informative(commit["commit_subject"]):
            stats.skipped_uninformative_subject += 1
            continue
        record = _record(spec, commit, diff)
        content_key = hashlib.sha256(
            f"{record['diff']}\n{record['commit_subject']}".encode("utf-8")
        ).hexdigest()
        if content_key in seen_content:
            stats.skipped_duplicate += 1
            continue
        seen_commits.add(identity)
        seen_content.add(content_key)
        accepted.append(record)
        stats.accepted += 1
    return accepted


def _write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _length_stats(values: Iterable[int]) -> dict[str, float | int]:
    numbers = sorted(values)
    if not numbers:
        return {"count": 0, "min": 0, "median": 0, "max": 0, "mean": 0}
    middle = len(numbers) // 2
    median = numbers[middle] if len(numbers) % 2 else (numbers[middle - 1] + numbers[middle]) / 2
    return {
        "count": len(numbers),
        "min": numbers[0],
        "median": median,
        "max": numbers[-1],
        "mean": round(sum(numbers) / len(numbers), 2),
    }


def split_by_repository(
    records: Sequence[dict[str, str]], seed: int = 42
) -> tuple[dict[str, list[dict[str, str]]], dict[str, object]]:
    """Split by repository when at least five repositories are available."""
    repositories = sorted({record["repository"] for record in records})
    if len(repositories) < 5:
        return {}, {
            "valid": False,
            "reason": "Repository-independent 70/15/15 splitting requires at least five repositories.",
            "repository_assignments": {},
        }
    import random

    shuffled = repositories[:]
    random.Random(seed).shuffle(shuffled)
    assignments = {
        repository: ("train" if index < len(shuffled) - 2 else "validation" if index == len(shuffled) - 2 else "test")
        for index, repository in enumerate(shuffled)
    }
    splits = {name: [record for record in records if assignments[record["repository"]] == name] for name in ("train", "validation", "test")}
    return splits, {
        "valid": True,
        "reason": "Repository-aware split; no repository occurs in more than one split.",
        "repository_assignments": assignments,
        "record_counts": {name: len(split) for name, split in splits.items()},
    }


def build_report(
    records: Sequence[dict[str, str]],
    stats: CollectionStats,
    split_manifest: dict[str, object],
    repositories: Sequence[RepositorySpec],
    *,
    max_diff_bytes: int,
) -> dict[str, object]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selected_repositories": [asdict(repository) for repository in repositories],
        "collection_stats": asdict(stats),
        "accepted_examples": len(records),
        "repository_distribution": dict(Counter(record["repository"] for record in records)),
        "diff_size_bytes": _length_stats(len(record["diff"].encode("utf-8")) for record in records),
        "subject_length_characters": _length_stats(len(record["commit_subject"]) for record in records),
        "max_diff_bytes": max_diff_bytes,
        "split": split_manifest,
        "known_limitations": [
            "The sample represents only the selected repositories.",
            "Repository-aware splitting uses one or more whole repositories per split and may be imbalanced.",
            "Upstream repository and commit licenses remain applicable to collected content.",
        ],
    }


def collect_dataset(
    output_dir: Path,
    *,
    repositories: Sequence[RepositorySpec] = REPOSITORIES,
    max_commits_per_repository: int = DEFAULT_MAX_COMMITS_PER_REPOSITORY,
    target_examples: int = DEFAULT_TARGET_EXAMPLES,
    max_diff_bytes: int = DEFAULT_MAX_DIFF_BYTES,
) -> dict[str, object]:
    """Clone bounded histories and write raw, processed, split, and report files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    processed_dir = output_dir / "processed"
    split_dir = output_dir / "splits"
    stats = CollectionStats()
    all_records: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory(prefix="commitgen-repositories-") as clone_root:
        clone_root_path = Path(clone_root)
        for spec in repositories:
            repo_path = clone_root_path / spec.name.replace("/", "__")
            subprocess.run(
                ["git", "clone", f"--depth={max_commits_per_repository}", spec.url, str(repo_path)],
                check=True,
                capture_output=True,
                text=True,
            )
            license_text = (repo_path / spec.license_file).read_text(encoding="utf-8", errors="replace")
            if spec.license_marker.lower() not in license_text.lower():
                raise ValueError(f"License verification failed for {spec.name}: {spec.license_file}")
            remaining = max(target_examples - len(all_records), 0)
            if not remaining:
                break
            all_records.extend(
                collect_repository(
                    spec,
                    repo_path,
                    max_commits=min(max_commits_per_repository, remaining * 2),
                    max_diff_bytes=max_diff_bytes,
                    stats=stats,
                )
            )
    records = all_records[:target_examples]
    _write_jsonl(raw_dir / "commits.jsonl", all_records)
    _write_jsonl(processed_dir / "commits.jsonl", records)
    splits, split_manifest = split_by_repository(records)
    for name, split_records in splits.items():
        _write_jsonl(split_dir / f"{name}.jsonl", split_records)
    report = build_report(records, stats, split_manifest, repositories, max_diff_bytes=max_diff_bytes)
    (processed_dir / "quality_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (processed_dir / "quality_report.md").write_text(_report_markdown(report), encoding="utf-8")
    (split_dir / "manifest.json").write_text(json.dumps(split_manifest, indent=2) + "\n", encoding="utf-8")
    return report


def _report_markdown(report: dict[str, object]) -> str:
    stats = report["collection_stats"]
    lines = [
        "# Commit dataset quality report",
        "",
        f"- Accepted examples: {report['accepted_examples']}",
        f"- Maximum diff size: {report['max_diff_bytes']} bytes",
        f"- Repository distribution: `{json.dumps(report['repository_distribution'], sort_keys=True)}`",
        f"- Diff size statistics: `{json.dumps(report['diff_size_bytes'])}`",
        f"- Subject length statistics: `{json.dumps(report['subject_length_characters'])}`",
        "",
        "## Filter counts",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in stats.items())
    lines.extend(["", "## Split", "", f"```json\n{json.dumps(report['split'], indent=2)}\n```", ""])
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--max-commits-per-repository", type=int, default=DEFAULT_MAX_COMMITS_PER_REPOSITORY)
    parser.add_argument("--target-examples", type=int, default=DEFAULT_TARGET_EXAMPLES)
    parser.add_argument("--max-diff-bytes", type=int, default=DEFAULT_MAX_DIFF_BYTES)
    args = parser.parse_args(argv)
    report = collect_dataset(
        args.output_dir,
        max_commits_per_repository=args.max_commits_per_repository,
        target_examples=args.target_examples,
        max_diff_bytes=args.max_diff_bytes,
    )
    print(json.dumps({"accepted_examples": report["accepted_examples"], "repository_distribution": report["repository_distribution"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
