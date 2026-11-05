"""Connected query groups and the approved SHA-256 train/development assignment."""

import hashlib
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass


def normalize_query(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


@dataclass(frozen=True, order=True)
class Query:
    locale: str
    query_id: int
    text: str


@dataclass(frozen=True)
class Assignment:
    locale: str
    query_id: int
    normalized_query: str
    group_key: str
    group_sha256: str
    partition: str


def assign_queries(train: Iterable[Query], test: Iterable[Query]) -> list[Assignment]:
    rows = sorted(set(train))
    parents = list(range(len(rows)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    by_id: dict[tuple[str, int], int] = {}
    by_text: dict[tuple[str, str], int] = {}
    for i, row in enumerate(rows):
        normalized = normalize_query(row.text)
        if not normalized:
            raise ValueError("Empty normalized query")
        id_key = (row.locale, row.query_id)
        if id_key in by_id:
            parents[find(i)] = find(by_id[id_key])
        else:
            by_id[id_key] = i
        text_key = (row.locale, normalized)
        if text_key in by_text:
            parents[find(i)] = find(by_text[text_key])
        else:
            by_text[text_key] = i
    tests = list(test)
    test_ids = {(row.locale, row.query_id) for row in tests}
    test_texts = {(row.locale, normalize_query(row.text)) for row in tests}
    groups: dict[int, list[Query]] = {}
    for i, row in enumerate(rows):
        groups.setdefault(find(i), []).append(row)
    assignments: list[Assignment] = []
    for group in groups.values():
        canonical = min(normalize_query(row.text) for row in group)
        locale = group[0].locale
        digest = hashlib.sha256(f"42|{locale}|{canonical}".encode()).hexdigest()
        quarantine = any(
            (row.locale, row.query_id) in test_ids
            or (row.locale, normalize_query(row.text)) in test_texts
            for row in group
        )
        partition = (
            "quarantine"
            if quarantine
            else ("development" if int(digest, 16) * 5 < 2**256 else "train")
        )
        assignments.extend(
            Assignment(
                locale, row.query_id, normalize_query(row.text), canonical, digest, partition
            )
            for row in group
        )
    return sorted(set(assignments), key=lambda a: (a.locale, a.query_id, a.normalized_query))
