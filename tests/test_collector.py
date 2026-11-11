import json
from pathlib import Path

from commitgen.data.collector import (
    CollectionStats,
    RepositorySpec,
    _parse_log,
    build_report,
    collect_repository,
    split_by_repository,
)


def test_parse_log_preserves_subject_and_body() -> None:
    output = "sha\x1fparent\x1f2026-01-01T00:00:00+00:00\x1fAdd feature\x1fAdd feature\n\nDetails\x1e\nnext\x1fparent2\x1f2026-01-02T00:00:00+00:00\x1fNext\x1fNext\x1e"
    records = list(_parse_log(output))
    assert records[0]["commit_sha"] == "sha"
    assert records[0]["commit_body"] == "Add feature\n\nDetails"
    assert records[1]["commit_sha"] == "next"


def test_collect_repository_filters_and_deduplicates(monkeypatch, tmp_path: Path) -> None:
    spec = RepositorySpec("example/repo", "https://example.com/repo.git", "MIT", "LICENSE", "MIT License")
    log = (
        "one\x1fparent\x1f2026-01-01T00:00:00+00:00\x1fAdd feature\x1fAdd feature\x1e"
        "two\x1fparent2\x1f2026-01-02T00:00:00+00:00\x1fupdate\x1fupdate\x1e"
    )

    def fake_run(args, cwd=None):
        command = list(args)
        if command[0] == "log":
            return log
        if command[0] == "diff" and "--name-only" in command:
            return "src/app.py\n"
        if command[0] == "diff" and "--numstat" in command:
            return "1\t1\tsrc/app.py\n"
        return "diff --git a/src/app.py b/src/app.py\n+new\n"

    monkeypatch.setattr("commitgen.data.collector._run_git", fake_run)
    stats = CollectionStats()
    records = collect_repository(spec, tmp_path, max_commits=10, max_diff_bytes=1000, stats=stats)
    assert len(records) == 1
    assert records[0]["commit_subject"] == "Add feature"
    assert stats.accepted == 1
    assert stats.skipped_uninformative_subject == 1


def test_split_requires_five_repositories() -> None:
    records = [{"repository": f"repo-{index}", "diff": "d", "commit_subject": "s"} for index in range(4)]
    splits, manifest = split_by_repository(records)
    assert splits == {}
    assert manifest["valid"] is False


def test_report_is_json_serializable() -> None:
    records = [{"repository": "a", "diff": "diff", "commit_subject": "subject"}]
    report = build_report(records, CollectionStats(accepted=1), {}, [], max_diff_bytes=100)
    json.dumps(report)
    assert report["accepted_examples"] == 1
