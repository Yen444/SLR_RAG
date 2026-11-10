"""Tags and numbered approaches (open coding)."""

import json

from slr_rag import coding, store
from slr_rag.catalog import scan_papers

from .conftest import make_pdf


def _papers(settings, review, n=3):
    for i in range(n):
        make_pdf(settings.papers_dir / f"p{i}.pdf", [f"paper {i}"])
    return [store.get(p, settings, review) for p in scan_papers(settings)]


def _tag(settings, ann, tags=None, approaches=None):
    ann["tags"] = tags if tags is not None else ann["tags"]
    ann["approaches"] = approaches if approaches is not None else ann["approaches"]
    store.save(ann, settings)


def test_canonical_reuses_existing_spelling():
    vocab = ["Petri net", "data-driven"]
    assert coding.canonical("  petri   NET ", vocab) == "Petri net"
    assert coding.canonical(" new  tag ", vocab) == "new tag"
    assert coding.canonical_list(["petri net", "Petri net", "simulation", "Simulation "], vocab) == \
        ["Petri net", "simulation"]


def test_tag_counts_rename_merge_and_remove(settings, review):
    a, b, c = _papers(settings, review)
    _tag(settings, a, ["Petri net", "simulation"])
    _tag(settings, b, ["petrinet"])
    _tag(settings, c, ["petrinet", "Petri net"])
    assert coding.tag_counts(store.load_all(settings)) == {"Petri net": 2, "simulation": 1, "petrinet": 2}

    assert coding.rename_tag(settings, "petrinet", "petri  NET") == 2  # merge into the existing spelling
    saved = store.load_all(settings)
    assert coding.tag_vocabulary(saved) == ["Petri net", "simulation"]
    assert saved[c["paper_id"]]["tags"] == ["Petri net"]  # no duplicate after the merge

    assert coding.remove_tag(settings, "simulation") == 1
    assert coding.tag_vocabulary(store.load_all(settings)) == ["Petri net"]


def test_approach_numbers_are_stable_and_never_reused(settings, review):
    assert coding.add_approach(settings, "Process-model simulation", " Simulate a model ") == 1
    assert coding.add_approach(settings, "Deep learning") == 2
    assert coding.add_approach(settings, "deep  learning") == 2  # same name: no duplicate
    coding.update_approach(settings, 2, description="Generative neural models")
    items = coding.load_approaches(settings)
    assert [coding.approach_label(a) for a in items] == ["1. Process-model simulation", "2. Deep learning"]
    assert items[0]["description"] == "Simulate a model" and items[1]["description"] == "Generative neural models"

    a, b, _ = _papers(settings, review)
    _tag(settings, a, approaches=[2])
    assert coding.remove_approach(settings, 2) == 1
    assert store.load_all(settings)[a["paper_id"]]["approaches"] == []
    assert coding.add_approach(settings, "Rule-based") == 3  # 2 is not reused
    coding.update_approach(settings, 3, name="Rule-based generation")
    assert coding.load_approaches(settings)[-1]["name"] == "Rule-based generation"


def test_merge_approaches_moves_papers(settings, review):
    one, two = coding.add_approach(settings, "A"), coding.add_approach(settings, "B")
    a, b, _ = _papers(settings, review)
    _tag(settings, a, approaches=[one])
    _tag(settings, b, approaches=[one, two])
    assert coding.merge_approaches(settings, one, two) == 2
    saved = store.load_all(settings)
    assert saved[a["paper_id"]]["approaches"] == [two] and saved[b["paper_id"]]["approaches"] == [two]
    assert [x["number"] for x in coding.load_approaches(settings)] == [two]
    assert json.loads((settings.data_dir / "approaches.json").read_text(encoding="utf-8"))["next_number"] == 3


def test_cooccurrence_and_export_columns(settings, review):
    n = coding.add_approach(settings, "Simulation")
    a, b, _ = _papers(settings, review)
    _tag(settings, a, ["Petri net", "simulation"], [n])
    _tag(settings, b, ["simulation"], [n])
    saved = store.load_all(settings)

    tags = coding.cooccurrence(saved, "tags", "tags")
    assert tags.loc["simulation", "simulation"] == 2 and tags.loc["Petri net", "simulation"] == 1
    per_approach = coding.cooccurrence(saved, "approaches", "tags", row_names={n: "1. Simulation"})
    assert per_approach.loc["1. Simulation", "simulation"] == 2

    assert coding.export_columns(saved[a["paper_id"]], coding.load_approaches(settings)) == \
        {"tags": "Petri net;simulation", "approaches": "1. Simulation"}
    df = store.export_table(scan_papers(settings), settings, review)
    assert set(df["approaches"]) == {"1. Simulation", ""}
