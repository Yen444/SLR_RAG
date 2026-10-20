"""PDF viewer: Mozilla PDF.js, served with the PDFs through Streamlit static file serving.

Streamlit serves <project>/static/ at /app/static/ (enableStaticServing in .streamlit/config.toml).
PDF.js is downloaded there once; each PDF is copied to static/pdf/<paper_id>.pdf when first viewed.
"""

import io
import shutil
import urllib.parse
import urllib.request
import zipfile

from .catalog import Paper
from .config import ROOT

PDFJS_VERSION = "4.10.38"
PDFJS_URL = f"https://github.com/mozilla/pdf.js/releases/download/v{PDFJS_VERSION}/pdfjs-{PDFJS_VERSION}-dist.zip"
STATIC_DIR = ROOT / "static"
PDFJS_DIR = STATIC_DIR / "pdfjs"
PDF_DIR = STATIC_DIR / "pdf"
URL_PREFIX = "/app/static"


def ensure_pdfjs() -> None:
    """Download and unpack PDF.js into static/pdfjs if it isn't there yet."""
    if (PDFJS_DIR / "web" / "viewer.html").exists():
        return
    with urllib.request.urlopen(PDFJS_URL, timeout=120) as response:
        data = response.read()
    tmp = STATIC_DIR / "pdfjs.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    zipfile.ZipFile(io.BytesIO(data)).extractall(tmp)
    (tmp / "web" / "compressed.tracemonkey-pldi-09.pdf").unlink(missing_ok=True)  # demo file
    tmp.rename(PDFJS_DIR)


def publish(paper: Paper) -> str:
    """Copy the PDF into static/ (if needed) and return its URL."""
    dest = PDF_DIR / f"{paper.paper_id}.pdf"
    if not dest.exists() or dest.stat().st_size != paper.path.stat().st_size:
        PDF_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(paper.path, dest)
    return f"{URL_PREFIX}/pdf/{paper.paper_id}.pdf"


def viewer_url(paper: Paper, page: int | None = None, search: str | None = None, nonce: int = 0) -> str:
    """URL of the PDF.js viewer for this paper, optionally jumping to a page and highlighting a phrase.

    Without a page, PDF.js restores the last page and zoom used for this file."""
    url = f"{URL_PREFIX}/pdfjs/web/viewer.html?file={urllib.parse.quote(publish(paper), safe='')}"
    params = {}
    if page:
        params["page"] = page
    if search:
        params.update(search=search, phrase="true")
    if params:
        params["n"] = nonce  # ignored by PDF.js; makes repeated jumps to the same page work
        url += "#" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    return url
