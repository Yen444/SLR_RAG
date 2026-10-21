"""Screening CSVs: linking rows to PDFs, scope, citation and export."""

import pandas as pd
import pytest

from slr_rag import screening, store
from slr_rag.catalog import scan_papers

from .conftest import make_pdf

COLUMNS = ["title", "author", "year", "journal", "doi", "abstract", "item_id", "final_decision", "pdf_filename",
           "reason", "note", "volume"]


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False, encoding="utf-8-sig")


def _pdf(settings, rel):
    path = settings.papers_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    make_pdf(path, [f"text of {rel}"], title=f"PDF title {rel}")


@pytest.fixture
def screened(settings):
    """papers/sd/ and papers/gs/: one folder per database, each with its CSV and PDFs."""
    for rel in ("sd/sd_1_included.pdf", "sd/pdfs/sd_2_included.pdf", "sd/other.pdf",
                "gs/paper_1.pdf", "sd/paper_1.pdf"):  # same file name in two databases
        _pdf(settings, rel)
    _write_csv(settings.papers_dir / "sd" / "results.csv", [
        ["Real Title One", "Nai, Roberto and Sulis, Emilio and Audrito, Davide", "2025", "J. One",
         "https://doi.org/10.1/x", "Abstract one.", "sd_1", "included", "sd_1_included.pdf", "", "a note", "57.0"],
        ["Real Title Two", "Jia Wei and Chun Ouyang", "2024", "J. Two", "10.2/y", "Abstract two.", "sd_2",
         "included", "", "", "", ""],  # no pdf_filename: matched by item_id, PDF in a sub-folder
        ["No PDF", "A B", "2023", "", "", "", "sd_3", "included", "sd_3_included.pdf", "", "", ""],
        ["Repeated", "", "", "", "", "", "sd_1", "included", "", "", "", ""],  # same item_id, same folder
    ])
    _write_csv(settings.papers_dir / "gs" / "results.csv", [
        ["Scholar One", "", "2022", "", "", "", "gs_1", "included", "paper_1.pdf", "", "", ""],
    ])
    papers = scan_papers(settings)
    records = screening.load_records(settings)
    return papers, records, screening.scope(papers, records)


def _by_title(sc, title):
    return next(p for p in sc.papers if p.title == title)


def test_load_records_reads_csvs_in_database_folders(screened):
    _, records, _ = screened
    assert list(zip(records["source_folder"], records["item_id"])) == [
        ("gs", "gs_1"), ("sd", "sd_1"), ("sd", "sd_2"), ("sd", "sd_3")]  # repeat within "sd" dropped
    assert list(records["source_csv"]) == ["gs/results.csv"] + ["sd/results.csv"] * 3
    assert records.loc[1, "volume"] == "57.0"  # read as text, unchanged


def test_scope_matches_pdfs_to_the_csv_of_their_own_folder(screened):
    _, _, sc = screened
    assert {p.title: p.filename for p in sc.papers} == {
        "Scholar One": "gs/paper_1.pdf",  # not sd/paper_1.pdf, which has the same name
        "Real Title One": "sd/sd_1_included.pdf",
        "Real Title Two": "sd/pdfs/sd_2_included.pdf",  # CSV in a folder above the PDF
    }
    assert _by_title(sc, "Real Title One").abstract == "Abstract one."
    assert [r["item_id"] for r in sc.missing] == ["sd_3"]
    assert sorted(p.filename for p in sc.hidden) == ["sd/other.pdf", "sd/paper_1.pdf"]


def test_no_csv_means_no_records(settings):
    _pdf(settings, "a.pdf")
    assert screening.load_records(settings).empty


def test_citation_and_doi():
    def cite(authors, year="", venue=""):
        return screening.citation({screening.AUTHORS: authors, screening.YEAR: year, screening.VENUE: venue})

    assert cite("Nai, Roberto and Sulis, Emilio and X, Y", "2025", "J") == "Nai et al. 2025 · J"
    assert cite("Jia Wei and Chun Ouyang", "2024") == "Wei & Ouyang 2024"
    assert cite("Nai, Roberto", "2025") == "Nai 2025"  # one BibTeX author, not two
    assert screening.doi_url({screening.DOI: "10.2/y"}) == "https://doi.org/10.2/y"
    assert screening.doi_url({screening.DOI: "https://doi.org/10.1/x"}) == "https://doi.org/10.1/x"


def test_export_keeps_original_columns_and_links_across_databases(screened, settings, review):
    _, records, sc = screened
    one, scholar = _by_title(sc, "Real Title One"), _by_title(sc, "Scholar One")
    ann = store.get(one, settings, review)
    ann["decision"], ann["exclusion_reason"], ann["fields"] = "exclude", "off topic", {"design": "RCT",
                                                                                       "methods": ["survey"]}
    store.save(ann, settings)
    store.link(store.get(one, settings, review), store.get(scholar, settings, review), settings)

    df = screening.export(records, sc, sc.papers, settings, review)
    assert list(df.columns[: len(COLUMNS)]) == COLUMNS and len(df) == 4
    row = df[df["title"] == "Real Title One"].iloc[0]
    assert row["final_decision"] == "included" and row["final_decision_updated"] == "excluded"
    assert row["review_exclusion_reason"] == "off topic"
    assert row["design"] == "RCT" and row["methods"] == "survey"
    assert row["linked_item_ids"] == "gs_1"  # linked across databases
    assert df[df["title"] == "Scholar One"].iloc[0]["linked_item_ids"] == "sd_1"
    assert df[df["title"] == "No PDF"].iloc[0]["review_status"] == ""


def test_column_names_differ_across_databases(settings):
    _pdf(settings, "ieee/ieee_2_included.pdf")
    _pdf(settings, "springer/springer_4_included.pdf")
    _pdf(settings, "gs/gs_65_included.pdf")
    pd.DataFrame([{"Document Title": "IEEE paper", "Authors": "S. Lee; I. Choi; H. Kim", "Publication Year": "2020",
                   "Publication Title": "IEEE Access", "DOI": "10.1109/x", "Abstract": "ieee abstract",
                   "item_id": "ieee_2", "pdf_filename": r"C:\\x\\ieee_2_included.pdf"}]).to_csv(
        settings.papers_dir / "ieee" / "included.csv", index=False)
    pd.DataFrame([{"Item Title": "Logs from&#xa0;Hybrid Models", "Authors": "Anti AlmanFabrizio Maria Maggi",
                   "Publication Year": "2024.0", "Publication Title": "BPM Workshops", "Item DOI": "10.1007/y",
                   "item_id": "springer_4", "pdf_filename": ""}]).to_csv(
        settings.papers_dir / "springer" / "included.csv", index=False)
    pd.DataFrame([{"Title": "Scholar paper", "Authors": "O Vasilecas, T Savickas, K Normantas…", "Year": "2016.0",
                   "Source": "Proc. X", "item_id": "gs_65", "pdf_filename": ""}]).to_csv(
        settings.papers_dir / "gs" / "included.csv", index=False)

    papers = scan_papers(settings)
    records = screening.load_records(settings)
    sc = screening.scope(papers, records)
    assert sorted(p.title for p in sc.papers) == ["IEEE paper", "Logs from Hybrid Models", "Scholar paper"]
    cite = {r[screening.TITLE]: screening.citation(r) for r in records.to_dict("records")}
    assert cite == {"IEEE paper": "Lee et al. 2020 · IEEE Access",
                    "Logs from Hybrid Models": "Alman & Maggi 2024 · BPM Workshops",
                    "Scholar paper": "Vasilecas et al. 2016 · Proc. X"}
    assert screening.doi_url(records[records[screening.TITLE] == "Logs from Hybrid Models"].iloc[0]) == \
        "https://doi.org/10.1007/y"
    df = screening.export(records, sc, sc.papers, settings, _review())
    assert not [c for c in df.columns if c.startswith("_")]  # internal columns are not exported
    assert "Document Title" in df.columns and "Item Title" in df.columns


def _review():
    from slr_rag.config import ReviewConfig
    return ReviewConfig(statuses=["todo", "done"], decisions=["include", "exclude"])
