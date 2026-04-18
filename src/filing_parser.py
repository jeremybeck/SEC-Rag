import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from llama_index.core import Document
from llama_index.core.node_parser import TokenTextSplitter

# =========================================================
# FILENAME PARSING
# =========================================================

_FISCAL_PERIOD_RE = re.compile(r"^(\d{4})Q(\d)$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _infer_quarter_from_month(month: int) -> int:
    if month <= 3:   return 4
    elif month <= 6: return 1
    elif month <= 9: return 2
    else:            return 3


@dataclass
class FilingInfo:
    filename: str
    ticker: str
    filing_type: str
    fiscal_year: int
    fiscal_quarter: Optional[int]
    filing_date: str
    inferred: bool


def parse_filing_filename(filename: str) -> FilingInfo:
    stem = Path(filename).stem
    stem = stem.removesuffix("_full")
    parts = stem.split("_")

    if len(parts) < 3:
        raise ValueError(f"Too few underscore-separated parts: {filename!r}")

    ticker = parts[0].upper()
    raw_type = parts[1].upper()
    filing_type = raw_type if "-" in raw_type else raw_type[:2] + "-" + raw_type[2:]

    remainder = parts[2:]

    fiscal_year = fiscal_quarter = filing_date = None
    inferred = False

    if len(remainder) == 2:
        period_token, date_token = remainder

        m = _FISCAL_PERIOD_RE.match(period_token)

        if m and _DATE_RE.match(date_token):
            fiscal_year = int(m.group(1))
            fiscal_quarter = int(m.group(2))
            filing_date = date_token
        else:
            raise ValueError(f"Unrecognised Pattern A tokens: {filename!r}")

    elif len(remainder) == 1:
        date_token = remainder[0]

        if not _DATE_RE.match(date_token):
            raise ValueError(f"Expected a date token, got {date_token!r}")

        filing_date = date_token
        filing_year, filing_month = int(date_token[:4]), int(date_token[5:7])

        inferred = True

        if filing_type == "10-K":
            fiscal_year = filing_year - 1 if filing_month <= 4 else filing_year
            fiscal_quarter = 4
        else:
            q = _infer_quarter_from_month(filing_month)
            fiscal_year = filing_year - 1 if q == 4 else filing_year
            fiscal_quarter = q

    else:
        raise ValueError(f"Unexpected token count: {filename!r}")

    return FilingInfo(
        filename=filename,
        ticker=ticker,
        filing_type=filing_type,
        fiscal_year=fiscal_year,
        fiscal_quarter=fiscal_quarter,
        filing_date=filing_date,
        inferred=inferred,
    )


# =========================================================
# FILE LOADING
# =========================================================

def load_file(filepath: Path) -> tuple[str, FilingInfo]:
    with open(filepath, "r", errors="ignore") as f:
        raw = f.read()

    info = parse_filing_filename(filepath.name)
    clean = raw.replace("\x00", " ")

    return clean, info


# =========================================================
# SEC SECTION SPLITTER
# =========================================================

def split_by_sec_items(text: str) -> list[tuple[str, str]]:
    pattern = r"(ITEM\s+\d+[A-Z]?)"

    text = text.replace("\n", " ")
    parts = re.split(pattern, text, flags=re.IGNORECASE)

    sections = []

    i = 1
    while i < len(parts) - 1:
        header = parts[i].strip().upper()
        body = parts[i + 1].strip()

        if len(body) > 1000:
            sections.append((header, body))

        i += 2

    return sections


# =========================================================
# SECTION LABELS
# =========================================================

SEC_LABELS = {
    "ITEM 1": "BUSINESS",
    "ITEM 1A": "RISK FACTORS",
    "ITEM 1B": "UNRESOLVED STAFF COMMENTS",
    "ITEM 7": "MD&A",
    "ITEM 7A": "MARKET RISK",
    "ITEM 8": "FINANCIAL STATEMENTS",
}


def format_section_header(item_code: str) -> str:
    item_code = item_code.upper().strip()
    label = SEC_LABELS.get(item_code, "UNKNOWN SECTION")
    return f"{label} ({item_code})"


# =========================================================
# DOCUMENT BUILDER
# =========================================================

def build_all_documents(
    file_list: list[str],
    corpus_dir: Path,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
) -> list[Document]:
    all_docs = []

    splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    for fname in file_list:
        filepath = corpus_dir / fname
        text, info = load_file(filepath)
        sections = split_by_sec_items(text)

        for section_code, section_text in sections:
            chunks = splitter.split_text(section_text)
            pretty_header = format_section_header(section_code)

            for i, chunk in enumerate(chunks):
                all_docs.append(
                    Document(
                        text=f"{pretty_header}\n\n{chunk}",
                        metadata={
                            "ticker": info.ticker,
                            "filing_type": info.filing_type,
                            "fiscal_year": info.fiscal_year,
                            "fiscal_quarter": info.fiscal_quarter,
                            "filing_date": info.filing_date,
                            "inferred": info.inferred,
                            "section_code": section_code,
                            "section_label": pretty_header,
                            "chunk_id": i,
                            "file": info.filename,
                        },
                    )
                )

    return all_docs


def load_manifest(corpus_dir: Path) -> list[str]:
    with open(corpus_dir / "manifest.json", "r") as f:
        manifest = json.load(f)
    return manifest["files"]
