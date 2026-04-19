import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

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

_SECTION_START_RE = re.compile(r"(ITEM\s+(\d+)[A-C]?)", re.IGNORECASE)
_CROSS_REF_SIGNALS = re.compile(
    r"\b(see|see also|pursuant to|discussed in|refer to|described in|"
    r"in accordance with|as defined in|set forth in)\s*$",
    re.IGNORECASE,
)


def _is_cross_reference(text_before: str) -> bool:
    snippet = text_before[-80:].strip()
    return bool(_CROSS_REF_SIGNALS.search(snippet))


def split_by_sec_items(text: str) -> list[tuple[str, str]]:
    # B2: Normalize whitespace/unicode without destroying paragraph structure
    text = re.sub(r"\n{3,}", "\n\n", text)   # collapse excessive blank lines
    text = re.sub(r"\xa0+", " ", text)        # normalize NBSP
    text = re.sub(r" {3,}", " ", text)        # collapse padding spaces

    # Find all ITEM N / ITEM NA matches and their positions
    matches = list(_SECTION_START_RE.finditer(text))
    sections = []

    for idx, match in enumerate(matches):
        item_num_str = match.group(2)
        try:
            item_num = int(item_num_str)
        except ValueError:
            continue

        # B1: Skip exhibit/regulation item numbers > 20
        if item_num > 20:
            continue

        # B1: Skip cross-reference occurrences (e.g. "see Item 1A", "pursuant to Item 601")
        if _is_cross_reference(text[: match.start()]):
            continue

        header = match.group(1).strip().upper()

        # Body runs from end of this match to start of the next accepted match
        body_start = match.end()
        body_end = len(text)
        for later in matches[idx + 1:]:
            later_num_str = later.group(2)
            try:
                later_num = int(later_num_str)
            except ValueError:
                continue
            if later_num > 20:
                continue
            if _is_cross_reference(text[: later.start()]):
                continue
            body_end = later.start()
            break

        body = text[body_start:body_end].strip()
        if len(body) > 1000:
            sections.append((header, body))

    return sections


# =========================================================
# SECTION LABELS
# =========================================================

_SEC_LABELS_10K: dict[str, str] = {
    "ITEM 1":  "BUSINESS",
    "ITEM 1A": "RISK FACTORS",
    "ITEM 1B": "UNRESOLVED STAFF COMMENTS",
    "ITEM 1C": "CYBERSECURITY",
    "ITEM 2":  "PROPERTIES",
    "ITEM 3":  "LEGAL PROCEEDINGS",
    "ITEM 4":  "MINE SAFETY DISCLOSURES",
    "ITEM 5":  "MARKET FOR COMMON EQUITY",
    "ITEM 6":  "SELECTED FINANCIAL DATA",
    "ITEM 7":  "MD&A",
    "ITEM 7A": "MARKET RISK",
    "ITEM 8":  "FINANCIAL STATEMENTS",
    "ITEM 9":  "ACCOUNTANT CHANGES",
    "ITEM 9A": "CONTROLS AND PROCEDURES",
    "ITEM 9B": "OTHER INFORMATION",
    "ITEM 10": "DIRECTORS AND GOVERNANCE",
    "ITEM 11": "EXECUTIVE COMPENSATION",
    "ITEM 12": "SECURITY OWNERSHIP",
    "ITEM 13": "RELATED TRANSACTIONS",
    "ITEM 14": "ACCOUNTANT FEES",
    "ITEM 15": "EXHIBITS",
}

_SEC_LABELS_10Q: dict[str, str] = {
    "ITEM 1":  "FINANCIAL STATEMENTS",
    "ITEM 2":  "MD&A",
    "ITEM 3":  "MARKET RISK",
    "ITEM 4":  "CONTROLS AND PROCEDURES",
    "ITEM 5":  "OTHER INFORMATION",
    "ITEM 6":  "EXHIBITS",
}

# Keep a legacy alias so external callers that import SEC_LABELS still work
SEC_LABELS = _SEC_LABELS_10K


def format_section_header(item_code: str, filing_type: str = "10-K") -> str:
    item_code = item_code.upper().strip()
    label_map = _SEC_LABELS_10Q if "10-Q" in filing_type.upper() else _SEC_LABELS_10K
    label = label_map.get(item_code, item_code)
    return f"{label} ({item_code})"


# =========================================================
# COMPANY NAMES — used to enrich chunk text for semantic search
# =========================================================

TICKER_NAMES: dict[str, str] = {
    "AAPL": "Apple",             "MSFT": "Microsoft",        "NVDA": "NVIDIA",
    "AMZN": "Amazon",            "GOOG": "Alphabet",         "META": "Meta",
    "AMD":  "AMD",               "INTC": "Intel",            "CRM":  "Salesforce",
    "ADBE": "Adobe",             "ORCL": "Oracle",           "CSCO": "Cisco",
    "IBM":  "IBM",               "TSLA": "Tesla",            "JPM":  "JPMorgan Chase",
    "BAC":  "Bank of America",   "GS":   "Goldman Sachs",    "MS":   "Morgan Stanley",
    "AXP":  "American Express",  "BLK":  "BlackRock",        "BRK":  "Berkshire Hathaway",
    "V":    "Visa",              "MA":   "Mastercard",       "JNJ":  "Johnson & Johnson",
    "PFE":  "Pfizer",            "MRK":  "Merck",            "LLY":  "Eli Lilly",
    "ABBV": "AbbVie",            "UNH":  "UnitedHealth",     "TMO":  "Thermo Fisher",
    "XOM":  "ExxonMobil",        "CVX":  "Chevron",          "KO":   "Coca-Cola",
    "PEP":  "PepsiCo",           "WMT":  "Walmart",          "COST": "Costco",
    "TGT":  "Target",            "PG":   "Procter & Gamble", "NKE":  "Nike",
    "MCD":  "McDonald's",        "SBUX": "Starbucks",        "DIS":  "Disney",
    "NFLX": "Netflix",           "CMCSA":"Comcast",          "T":    "AT&T",
    "VZ":   "Verizon",           "BA":   "Boeing",           "CAT":  "Caterpillar",
    "LMT":  "Lockheed Martin",   "RTX":  "RTX",              "DE":   "Deere & Company",
    "UPS":  "UPS",               "HD":   "Home Depot",       "GE":   "GE",
}


def _chunk_prefix(info: "FilingInfo", section_label: str) -> str:
    """Build a self-describing text prefix that gives the embedding model full context."""
    company = TICKER_NAMES.get(info.ticker, info.ticker)
    filing_label = "Annual Report" if "10-K" in info.filing_type else "Quarterly Report"
    if info.fiscal_quarter:
        period = f"FY{info.fiscal_year} Q{info.fiscal_quarter}"
    else:
        period = f"FY{info.fiscal_year}"
    return f"{company} ({info.ticker}) | {info.filing_type} {filing_label} | {period} | {section_label}"


# =========================================================
# DOCUMENT BUILDER
# =========================================================

def build_all_documents(
    file_list: list[str],
    corpus_dir: Path,
    chunk_size: int = 800,
    chunk_overlap: int = 200,
) -> list[Document]:
    all_docs = []

    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        paragraph_separator="\n\n",
    )

    for fname in file_list:
        filepath = corpus_dir / fname
        text, info = load_file(filepath)
        sections = split_by_sec_items(text)

        for section_code, section_text in sections:
            chunks = splitter.split_text(section_text)
            section_label = format_section_header(section_code, info.filing_type)
            prefix = _chunk_prefix(info, section_label)

            for i, chunk in enumerate(chunks):
                all_docs.append(
                    Document(
                        text=f"{prefix}\n\n{chunk}",
                        metadata={
                            "ticker": info.ticker,
                            "filing_type": info.filing_type,
                            "fiscal_year": info.fiscal_year,
                            "fiscal_quarter": info.fiscal_quarter,
                            "filing_date": info.filing_date,
                            "inferred": info.inferred,
                            "section_code": section_code,
                            "section_label": section_label,
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
