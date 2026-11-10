"""Tags and approaches that grow while papers are analysed (open coding).

Tags are plain names stored per paper (`tags`); the list of tags is whatever the papers use.
Approaches are numbered, with an optional description, in data/approaches.json; papers store the
numbers (`approaches`), so renaming an approach or editing its description changes one place only.
Numbers are never reused, so "3." keeps meaning the same approach in notes and exports.
"""

import json
import os
import tempfile
from collections import Counter
from itertools import combinations

import pandas as pd

from . import store
from .config import Settings


def _key(name: str) -> str:
    return " ".join(name.split()).casefold()


def clean(name: str) -> str:
    return " ".join(str(name).split())


def canonical(name: str, vocab: list[str]) -> str:
    """The existing name that differs only by case or spaces ('petri net' -> 'Petri net'), else the cleaned name."""
    return {_key(v): v for v in vocab}.get(_key(name), clean(name))


def canonical_list(names: list[str], vocab: list[str]) -> list[str]:
    out = []
    for name in names:
        name = canonical(name, [*vocab, *out])
        if name and name not in out:
            out.append(name)
    return out


# ---------- tags ----------

def tag_counts(saved: dict[str, dict]) -> Counter:
    return Counter(t for ann in saved.values() for t in ann.get("tags", []))


def tag_vocabulary(saved: dict[str, dict]) -> list[str]:
    return sorted(tag_counts(saved), key=str.casefold)


def _rewrite(settings: Settings, attr: str, change) -> int:
    """Apply `change(list) -> list` to `attr` of every saved annotation; returns the number of papers changed."""
    n = 0
    for ann in store.load_all(settings).values():
        old = ann.get(attr, [])
        new = change(old)
        if new != old:
            ann[attr] = new
            store.save(ann, settings)
            n += 1
    return n


def _replace(values: list, old, new) -> list:
    """`old` replaced by `new` in place, without duplicates (merge)."""
    out = []
    for v in values:
        v = new if v == old else v
        if v not in out:
            out.append(v)
    return out


def rename_tag(settings: Settings, old: str, new: str) -> int:
    """Rename a tag on all papers. If `new` is already a tag, the two are merged."""
    new = canonical(new, [t for t in tag_vocabulary(store.load_all(settings)) if t != old])
    return _rewrite(settings, "tags", lambda tags: _replace(tags, old, new))


def remove_tag(settings: Settings, name: str) -> int:
    return _rewrite(settings, "tags", lambda tags: [t for t in tags if t != name])


# ---------- approaches ----------

def _path(settings: Settings):
    return settings.data_dir / "approaches.json"


def load_approaches(settings: Settings) -> list[dict]:
    """[{"number", "name", "description"}, ...] in number order."""
    path = _path(settings)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))["approaches"]


def save_approaches(settings: Settings, items: list[dict], next_number: int | None = None) -> None:
    """Atomic write. `next_number` is kept in the file so numbers of removed approaches are never reused."""
    path = _path(settings)
    old_next = json.loads(path.read_text(encoding="utf-8")).get("next_number", 1) if path.exists() else 1
    data = {
        "next_number": max(next_number or 1, old_next, *(a["number"] + 1 for a in items)),
        "approaches": sorted(items, key=lambda a: a["number"]),
    }
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=settings.data_dir, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def approach_label(a: dict) -> str:
    return f"{a['number']}. {a['name']}"


def add_approach(settings: Settings, name: str, description: str = "") -> int:
    """Number of a new approach; an existing name (ignoring case and spaces) returns that approach's number."""
    items = load_approaches(settings)
    for a in items:
        if _key(a["name"]) == _key(name):
            return a["number"]
    path = _path(settings)
    number = json.loads(path.read_text(encoding="utf-8")).get("next_number", 1) if path.exists() else 1
    items.append({"number": number, "name": clean(name), "description": description.strip()})
    save_approaches(settings, items, number + 1)
    return number


def update_approach(settings: Settings, number: int, name: str | None = None,
                    description: str | None = None) -> None:
    items = load_approaches(settings)
    for a in items:
        if a["number"] == number:
            if name is not None and clean(name):
                a["name"] = clean(name)
            if description is not None:
                a["description"] = description.strip()
    save_approaches(settings, items)


def merge_approaches(settings: Settings, src: int, dst: int) -> int:
    """Move all papers of approach `src` to `dst`, then delete `src`. Returns the number of papers changed."""
    if src == dst:
        return 0
    n = _rewrite(settings, "approaches", lambda nums: _replace(nums, src, dst))
    save_approaches(settings, [a for a in load_approaches(settings) if a["number"] != src])
    return n


def remove_approach(settings: Settings, number: int) -> int:
    n = _rewrite(settings, "approaches", lambda nums: [x for x in nums if x != number])
    save_approaches(settings, [a for a in load_approaches(settings) if a["number"] != number])
    return n


def approach_counts(saved: dict[str, dict]) -> Counter:
    return Counter(x for ann in saved.values() for x in ann.get("approaches", []))


def approach_names(numbers: list[int], approaches: list[dict]) -> str:
    """'1. Name;3. Other' for export (a number missing from the registry is written as is)."""
    by_number = {a["number"]: a for a in approaches}
    return ";".join(approach_label(by_number[x]) if x in by_number else str(x) for x in numbers)


# ---------- overview ----------

def cooccurrence(saved: dict[str, dict], rows: str, cols: str, row_names: dict | None = None,
                 col_names: dict | None = None) -> pd.DataFrame:
    """Number of papers having both a `rows` item and a `cols` item ("tags" or "approaches").
    With rows == cols, the diagonal is how many papers have that item. `*_names` map items to labels."""
    counts: Counter = Counter()
    for ann in saved.values():
        r_items, c_items = ann.get(rows, []), ann.get(cols, [])
        if rows == cols:
            counts.update((x, x) for x in r_items)
            for x, y in combinations(r_items, 2):
                counts.update([(x, y), (y, x)])
        else:
            counts.update((x, y) for x in r_items for y in c_items)
    order = lambda v: v.casefold() if isinstance(v, str) else v  # noqa: E731  (tags by name, approaches by number)
    row_items = sorted({x for x, _ in counts}, key=order)
    col_items = sorted({y for _, y in counts}, key=order)
    return pd.DataFrame([[counts[(x, y)] for y in col_items] for x in row_items],
                        index=[(row_names or {}).get(x, x) for x in row_items],
                        columns=[(col_names or {}).get(y, y) for y in col_items], dtype=int)


def export_columns(ann: dict, approaches: list[dict]) -> dict:
    return {"tags": ";".join(ann.get("tags", [])), "approaches": approach_names(ann.get("approaches", []), approaches)}
