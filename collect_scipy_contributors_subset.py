#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
from pathlib import Path


RELEASES = [
    "0.8.0",
    "1.0.0",
    "1.7.0",
    "1.11.0",
    "1.17.0",
    "1.18.0",
]

RELEASE_YEAR = {
    "0.8.0": 2010,
    "1.0.0": 2017,
    "1.7.0": 2021,
    "1.11.0": 2023,
    "1.17.0": 2026,
    "1.18.0": 2026,
}

OUTPUT_PATH = Path("scipy_contributor_tenure_subset.csv")
RELEASE_NOTES_DIR = Path(__file__).resolve().parent / "doc" / "source" / "release"
AUTHOR_LINE = re.compile(
    r"^\*\s+(?P<name>.+?)(?:\s+\((?P<count>\d+)\))?(?:\s+(?P<marker>\+))?$"
)
SUMMARY_PREFIXES = (
    "A total of",
    "People with",
    "This list",
    "NOTE:",
)

def normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip())


def extract_authors(text: str) -> list[tuple[str, int]]:
    lines = text.splitlines()
    author_index = next((i for i, line in enumerate(lines) if line.strip() == "Authors"), None)
    if author_index is None:
        return []

    authors: dict[str, int] = {}
    saw_author_line = False
    for line in lines[author_index + 1:]:
        stripped = line.strip()
        match = AUTHOR_LINE.match(stripped)
        if match:
            saw_author_line = True
            author_name = normalize_name(match.group("name"))
            if author_name in {"Name", "Name (commits)"}:
                continue
            authors[author_name] = int(match.group("marker") == "+")
            continue

        if not saw_author_line:
            continue
        if not stripped:
            continue
        if stripped.startswith(SUMMARY_PREFIXES):
            continue
        break

    return sorted(authors.items(), key=lambda item: item[0].casefold())


def load_release_text(release: str) -> tuple[str, str]:
    local_path = RELEASE_NOTES_DIR / f"{release}-notes.rst"
    if local_path.exists():
        return str(local_path), local_path.read_text(encoding="utf-8")
    raise RuntimeError(f"Missing local release notes file for {release}: {local_path}")


def main() -> None:
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    first_seen_release: dict[str, str] = {}
    first_seen_index: dict[str, int] = {}
    counts_by_release: list[tuple[str, int, str]] = []

    for release_index, release in enumerate(RELEASES):
        source, text = load_release_text(release)
        authors = extract_authors(text)
        if not authors:
            failures.append(f"{release}: no Authors section entries found in {source}")
            counts_by_release.append((release, 0, "missing-authors"))
            print(f"[status] release={release} year={RELEASE_YEAR[release]} count=0 status=missing-authors source={source}")
            continue

        counts_by_release.append((release, len(authors), "ok"))
        print(f"[status] release={release} year={RELEASE_YEAR[release]} count={len(authors)} status=ok source={source}")

        for person_raw, is_first_time_marker in authors:
            if person_raw not in first_seen_release:
                first_seen_release[person_raw] = release
                first_seen_index[person_raw] = release_index

            rows.append(
                {
                    "release": release,
                    "year": RELEASE_YEAR[release],
                    "person_raw": person_raw,
                    "is_first_time_marker": is_first_time_marker,
                    "first_seen_release": first_seen_release[person_raw],
                    "tenure_release_index": release_index - first_seen_index[person_raw],
                }
            )

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "release",
                "year",
                "person_raw",
                "is_first_time_marker",
                "first_seen_release",
                "tenure_release_index",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[output] wrote {len(rows)} rows to {OUTPUT_PATH}")
    print("[summary] parsed counts by release:")
    for release, count, status in counts_by_release:
        print(f"[summary] {release}: {count} contributors ({status})")

    print("[sample] first rows:")
    for row in rows[:10]:
        print(f"[sample] {row}")

    if failures:
        print("[failures] encountered:")
        for failure in failures:
            print(f"[failures] {failure}")
    else:
        print("[failures] none")


if __name__ == "__main__":
    main()
