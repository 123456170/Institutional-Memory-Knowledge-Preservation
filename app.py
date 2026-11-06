import base64
import hashlib
import io
import os
import re
import tempfile
import time

import imageio
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw, ImageFont
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

APP_TITLE = "Institutional Memory & Knowledge Preservation — Live Demo"
SEED = 20261106
DEMO_NOW = pd.Timestamp("2026-11-06 09:00:00")

ROLE_LEVEL = {"Viewer": 0, "Steward": 1, "Admin": 2}
CONF_LEVEL = {"Public": 0, "Internal": 1, "Confidential": 2}

CATEGORIES = [
    "ADR",
    "RFC",
    "Meeting Minutes",
    "Policy",
    "Email Thread",
    "Ticket",
    "Architecture Note",
]

OWNERS = [
    "A. Okafor",
    "L. Chen",
    "M. Rivera",
    "S. Patel",
    "J. Kim",
    "D. Novak",
    "R. Iyer",
    "K. Tanaka",
    "H. Salem",
]

TEAMS = ["Platform", "Security", "Data", "Clinical AI", "Operations", "Records"]

STATUS_VALUES = ["Accepted", "Proposed", "Deprecated", "Superseded", "Draft", "Closed"]

TOPICS = [
    "data retention",
    "vendor risk",
    "model monitoring",
    "sensor calibration",
    "clinical review",
    "budget forecast",
    "access provisioning",
    "audit logging",
    "backup restore",
    "training data governance",
    "archive migration",
    "records classification",
    "knowledge handover",
    "incident postmortem",
    "policy exception",
]

CORE_RECORDS = [
    {
        "doc_id": "ADR-0001",
        "title": "Adopt OAuth 2.0 device flow for field devices",
        "category": "ADR",
        "owner": "A. Okafor",
        "team": "Platform",
        "date": "2021-03-12",
        "last_review": "2024-05-01",
        "status": "Accepted",
        "confidentiality": "Internal",
        "retention_days": 3650,
        "source_uri": "https://archive.example.org/records/adr-0001",
        "tags": ["authentication", "oauth", "field-devices"],
        "supersedes": "",
        "superseded_by": "",
        "text": (
            "Decision: Adopt OAuth 2.0 device flow for field devices. "
            "Rationale: reduces shared credentials and supports revocation. "
            "Alternatives considered: static tokens and mutual TLS. "
            "Consequences: onboarding needs a device broker. "
            "Owner: A. Okafor. Date: 2021-03-12."
        ),
    },
    {
        "doc_id": "ADR-0005",
        "title": "Use PostgreSQL as the archival system of record",
        "category": "ADR",
        "owner": "L. Chen",
        "team": "Data",
        "date": "2020-09-04",
        "last_review": "2024-01-15",
        "status": "Accepted",
        "confidentiality": "Internal",
        "retention_days": 3650,
        "source_uri": "https://archive.example.org/records/adr-0005",
        "tags": ["postgres", "database", "archive"],
        "supersedes": "",
        "superseded_by": "",
        "text": (
            "Decision: Use PostgreSQL as the system of record for archival metadata. "
            "Rationale: transactional integrity, JSONB, mature backups. "
            "Alternatives considered: MongoDB and Elasticsearch as primary store. "
            "Owner: L. Chen. Date: 2020-09-04."
        ),
    },
    {
        "doc_id": "ADR-0009",
        "title": "Allow legacy FTP for partner file drops",
        "category": "ADR",
        "owner": "J. Kim",
        "team": "Operations",
        "date": "2019-06-10",
        "last_review": "2020-06-10",
        "status": "Superseded",
        "confidentiality": "Internal",
        "retention_days": 365,
        "source_uri": "https://archive.example.org/records/adr-0009",
        "tags": ["ftp", "partners", "legacy"],
        "supersedes": "",
        "superseded_by": "ADR-0014",
        "text": (
            "Decision: Allow legacy FTP for partner file drops. "
            "Rationale: compatibility with older partner systems. "
            "Alternatives considered: SFTP. "
            "Owner: J. Kim. Date: 2019-06-10."
        ),
    },
    {
        "doc_id": "ADR-0014",
        "title": "Replace legacy FTP with SFTP and AS2 for partner transfers",
        "category": "ADR",
        "owner": "L. Chen",
        "team": "Security",
        "date": "2023-01-18",
        "last_review": "2025-01-20",
        "status": "Accepted",
        "confidentiality": "Internal",
        "retention_days": 3650,
        "source_uri": "https://archive.example.org/records/adr-0014",
        "tags": ["sftp", "as2", "file-transfer", "security"],
        "supersedes": "ADR-0009",
        "superseded_by": "",
        "text": (
            "Decision: Replace legacy FTP with SFTP and AS2 for partner transfers. "
            "This record superseded ADR-0009. "
            "Rationale: encryption in transit, non-repudiation, better audit trails. "
            "Alternatives considered: continuing FTP with VPN, managed file transfer appliance. "
            "Owner: L. Chen. Date: 2023-01-18."
        ),
    },
    {
        "doc_id": "RFC-002",
        "title": "Where architecture decision records are stored",
        "category": "RFC",
        "owner": "R. Iyer",
        "team": "Records",
        "date": "2022-04-02",
        "last_review": "2024-10-01",
        "status": "Accepted",
        "confidentiality": "Public",
        "retention_days": 3650,
        "source_uri": "https://archive.example.org/records/rfc-002",
        "tags": ["adr", "storage", "git", "provenance"],
        "supersedes": "",
        "superseded_by": "",
        "text": (
            "Summary: Store architecture decision records in a Git-backed archive with immutable IDs. "
            "Motivations: searchable history, provenance, review trails. "
            "Proposed change: all ADRs are stored under records/adr and linked to owners and dates. "
            "Owner: R. Iyer. Date: 2022-04-02."
        ),
    },
    {
        "doc_id": "RFC-007",
        "title": "Event bus for audit records",
        "category": "RFC",
        "owner": "M. Rivera",
        "team": "Platform",
        "date": "2023-07-11",
        "last_review": "2025-02-01",
        "status": "Accepted",
        "confidentiality": "Internal",
        "retention_days": 2555,
        "source_uri": "https://archive.example.org/records/rfc-007",
        "tags": ["kafka", "audit", "events", "stream-processing"],
        "supersedes": "",
        "superseded_by": "",
        "text": (
            "Proposed change: Use Kafka for audit event ingestion. "
            "Rationale: durable replay, partitioned ordering, ecosystem support. "
            "Alternatives considered: Apache Pulsar and batch HTTP ingest were rejected because "
            "operational maturity and replay semantics were weaker for audit use. "
            "Decision: Kafka selected for audit events. "
            "Owner: M. Rivera. Date: 2023-07-11."
        ),
    },
    {
        "doc_id": "POL-001",
        "title": "Encryption at rest standard",
        "category": "Policy",
        "owner": "S. Patel",
        "team": "Security",
        "date": "2021-11-03",
        "last_review": "2025-06-01",
        "status": "Accepted",
        "confidentiality": "Public",
        "retention_days": 3650,
        "source_uri": "https://archive.example.org/records/pol-001",
        "tags": ["encryption", "aes-256", "data-at-rest"],
        "supersedes": "",
        "superseded_by": "",
        "text": (
            "Policy: Encrypt data at rest using AES-256 or an approved equivalent. "
            "Scope: databases, object storage, backups, and export bundles. "
            "Owner: S. Patel. Date: 2021-11-03."
        ),
    },
    {
        "doc_id": "POL-003",
        "title": "Incident response playbook ownership",
        "category": "Policy",
        "owner": "M. Rivera",
        "team": "Security",
        "date": "2022-02-17",
        "last_review": "2025-03-04",
        "status": "Accepted",
        "confidentiality": "Internal",
        "retention_days": 2555,
        "source_uri": "https://archive.example.org/records/pol-003",
        "tags": ["incident-response", "playbook", "ownership"],
        "supersedes": "",
        "superseded_by": "",
        "text": (
            "Policy: The incident response playbook is owned by Security Operations. "
            "Accountable owner: M. Rivera. "
            "Review cadence: annual or after major incidents. "
            "Date: 2022-02-17."
        ),
    },
    {
        "doc_id": "POL-005",
        "title": "Review cadence for stale ADRs and policies",
        "category": "Policy",
        "owner": "R. Iyer",
        "team": "Records",
        "date": "2023-03-09",
        "last_review": "2025-04-01",
        "status": "Accepted",
        "confidentiality": "Public",
        "retention_days": 3650,
        "source_uri": "https://archive.example.org/records/pol-005",
        "tags": ["review", "staleness", "adr", "policy"],
        "supersedes": "",
        "superseded_by": "",
        "text": (
            "Policy: ADRs and policies are reviewed every 12 months. "
            "Stale items are flagged after 15 months without review. "
            "Departing-expert capture is triggered when an owner changes or leaves. "
            "Owner: R. Iyer. Date: 2023-03-09."
        ),
    },
    {
        "doc_id": "POL-008",
        "title": "Retention for confidential customer email threads",
        "category": "Policy",
        "owner": "D. Novak",
        "team": "Records",
        "date": "2024-05-21",
        "last_review": "2025-05-22",
        "status": "Accepted",
        "confidentiality": "Confidential",
        "retention_days": 2555,
        "source_uri": "https://archive.example.org/records/pol-008",
        "tags": ["retention", "email", "confidential", "customer"],
        "supersedes": "",
        "superseded_by": "",
        "text": (
            "Policy: Confidential customer email threads are retained for seven years "
            "unless a shorter legal hold applies. Redaction is required before broad sharing. "
            "Owner: D. Novak. Date: 2024-05-21."
        ),
    },
    {
        "doc_id": "MTG-001",
        "title": "Project kickoff meeting minutes",
        "category": "Meeting Minutes",
        "owner": "K. Tanaka",
        "team": "Data",
        "date": "2018-11-05",
        "last_review": "2019-05-05",
        "status": "Closed",
        "confidentiality": "Internal",
        "retention_days": 365,
        "source_uri": "https://archive.example.org/records/mtg-001",
        "tags": ["kickoff", "taxonomy", "meeting"],
        "supersedes": "",
        "superseded_by": "",
        "text": (
            "Meeting minutes: project kickoff. "
            "Decisions: use provisional taxonomy and revisit after first ingestion cycle. "
            "Owner: K. Tanaka. Date: 2018-11-05."
        ),
    },
    {
        "doc_id": "MTG-010",
        "title": "Fetal ultrasound segmentation pilot approval",
        "category": "Meeting Minutes",
        "owner": "H. Salem",
        "team": "Clinical AI",
        "date": "2025-02-14",
        "last_review": "2025-08-14",
        "status": "Accepted",
        "confidentiality": "Internal",
        "retention_days": 2555,
        "source_uri": "https://archive.example.org/records/mtg-010",
        "tags": ["ultrasound", "segmentation", "pilot", "clinical-ai"],
        "supersedes": "",
        "superseded_by": "",
        "text": (
            "Meeting minutes: approved fetal ultrasound segmentation pilot with clinician oversight. "
            "Decision: begin synthetic-data validation before prospective review. "
            "Owner: H. Salem. Date: 2025-02-14."
        ),
    },
    {
        "doc_id": "MTG-018",
        "title": "Model registry rollout accountability",
        "category": "Meeting Minutes",
        "owner": "L. Chen",
        "team": "Platform",
        "date": "2025-08-22",
        "last_review": "2025-10-01",
        "status": "Accepted",
        "confidentiality": "Internal",
        "retention_days": 2555,
        "source_uri": "https://archive.example.org/records/mtg-018",
        "tags": ["model-registry", "rollout", "accountability"],
        "supersedes": "",
        "superseded_by": "",
        "text": (
            "Meeting minutes: Platform team is accountable for model registry rollout. "
            "Decision: staging promotion requires approval from model owner and records steward. "
            "Owner: L. Chen. Date: 2025-08-22."
        ),
    },
    {
        "doc_id": "TICK-012",
        "title": "November sensor outage postmortem",
        "category": "Ticket",
        "owner": "M. Rivera",
        "team": "Operations",
        "date": "2024-11-19",
        "last_review": "2025-05-01",
        "status": "Closed",
        "confidentiality": "Internal",
        "retention_days": 1825,
        "source_uri": "https://archive.example.org/records/tick-012",
        "tags": ["sensor", "outage", "root-cause", "postmortem"],
        "supersedes": "",
        "superseded_by": "",
        "text": (
            "Ticket: November sensor outage. "
            "Root cause: moisture ingress in vibration sensor connector caused intermittent telemetry loss. "
            "Resolution: replace harness, add IP67 enclosure, and update inspection checklist. "
            "Owner: M. Rivera. Date: 2024-11-19."
        ),
    },
    {
        "doc_id": "EMAIL-004",
        "title": "Confidential customer contract renewal thread",
        "category": "Email Thread",
        "owner": "D. Novak",
        "team": "Records",
        "date": "2025-05-09",
        "last_review": "2025-05-10",
        "status": "Closed",
        "confidentiality": "Confidential",
        "retention_days": 2555,
        "source_uri": "https://archive.example.org/records/email-004",
        "tags": ["contract", "customer", "confidential", "renewal"],
        "supersedes": "",
        "superseded_by": "",
        "text": (
            "Email thread: confidential customer contract renewal terms. "
            "Decision: renew with amended data-processing addendum. "
            "Owner: D. Novak. Date: 2025-05-09."
        ),
    },
    {
        "doc_id": "ARCH-003",
        "title": "Data lineage view for institutional memory",
        "category": "Architecture Note",
        "owner": "R. Iyer",
        "team": "Data",
        "date": "2024-09-12",
        "last_review": "2025-09-01",
        "status": "Accepted",
        "confidentiality": "Public",
        "retention_days": 3650,
        "source_uri": "https://archive.example.org/records/arch-003",
        "tags": ["lineage", "graph", "provenance", "timeline"],
        "supersedes": "",
        "superseded_by": "",
        "text": (
            "Architecture note: data lineage view links records, owners, dates, and supersession chains. "
            "Purpose: make historical knowledge traceable and reduce loss of context. "
            "Owner: R. Iyer. Date: 2024-09-12."
        ),
    },
]

GOLD_QUESTIONS = [
    {
        "question": "Which authentication standard was adopted for field devices and why?",
        "expected_doc_id": "ADR-0001",
    },
    {
        "question": "Why was PostgreSQL chosen over MongoDB for archival metadata?",
        "expected_doc_id": "ADR-0005",
    },
    {
        "question": "What superseded the legacy FTP transfer decision?",
        "expected_doc_id": "ADR-0014",
    },
    {
        "question": "Who owns the incident response playbook?",
        "expected_doc_id": "POL-003",
    },
    {
        "question": "When was the fetal ultrasound segmentation pilot approved?",
        "expected_doc_id": "MTG-010",
    },
    {
        "question": "What alternative to Kafka was rejected for audit events?",
        "expected_doc_id": "RFC-007",
    },
    {
        "question": "Which document defines retention for confidential customer emails?",
        "expected_doc_id": "POL-008",
    },
    {
        "question": "What was the root cause of the November sensor outage?",
        "expected_doc_id": "TICK-012",
    },
    {
        "question": "Where are architecture decision records stored?",
        "expected_doc_id": "RFC-002",
    },
    {
        "question": "What is the review cadence for stale ADRs?",
        "expected_doc_id": "POL-005",
    },
    {
        "question": "Which team is accountable for model registry rollout?",
        "expected_doc_id": "MTG-018",
    },
    {
        "question": "What encryption algorithm is mandated for data at rest?",
        "expected_doc_id": "POL-001",
    },
]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def get_font(size: int):
    candidates = ["arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def record_text_series(df: pd.DataFrame) -> pd.Series:
    titles = df["title"].fillna("").astype(str)
    tags = df["tags"].apply(lambda x: " ".join(map(str, x)) if isinstance(x, list) else "")
    texts = df["text"].fillna("").astype(str)
    return titles + " " + tags + " " + texts


def compute_governance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["date", "last_review", "ingested_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    df["retention_days"] = (
        pd.to_numeric(df["retention_days"], errors="coerce").fillna(3650).astype(int)
    )

    df["expiry_date"] = df["date"] + pd.to_timedelta(df["retention_days"], unit="D")
    df["days_to_expiry"] = (df["expiry_date"] - DEMO_NOW).dt.days
    df["days_since_review"] = (DEMO_NOW - df["last_review"]).dt.days
    df["expired"] = df["days_to_expiry"] < 0

    draft_bonus = (df["status"].astype(str).str.lower() == "draft").astype(float) * 0.15
    review_factor = np.clip(df["days_since_review"].fillna(730) / 730.0, 0, 1)

    dte = df["days_to_expiry"].fillna(9999)
    near_expiry = np.clip((365 - dte) / 365.0, 0, 0.45)
    expiry_factor = np.where(df["expired"], 0.55, near_expiry)

    df["stale_score"] = np.clip(review_factor + expiry_factor + draft_bonus, 0, 1).round(3)
    df["stale_flagged"] = df["expired"] | (df["stale_score"] >= 0.60)
    df["permission_level"] = df["confidentiality"].map(CONF_LEVEL).fillna(1).astype(int)

    return df


def build_base_corpus():
    rng = np.random.default_rng(SEED)
    rows = []

    for rec in CORE_RECORDS:
        r = rec.copy()
        r.setdefault("team", str(rng.choice(TEAMS)))
        r.setdefault("supersedes", "")
        r.setdefault("superseded_by", "")
        r.setdefault(
            "source_uri",
            f"https://archive.example.org/records/{r['doc_id'].lower()}",
        )
        rows.append(r)

    for i in range(34):
        doc_id = f"DOC-{1000 + i}"
        category = str(rng.choice(CATEGORIES))
        topic = str(rng.choice(TOPICS))
        owner = str(rng.choice(OWNERS))
        team = str(rng.choice(TEAMS))

        days_ago = int(rng.integers(45, 2900))
        date = DEMO_NOW - pd.Timedelta(days=days_ago)

        review_span = max(1, min(540, days_ago))
        review_gap = int(rng.integers(0, review_span))
        last_review = date + pd.Timedelta(days=review_gap)
        if last_review > DEMO_NOW:
            last_review = DEMO_NOW

        status = str(rng.choice(STATUS_VALUES))
        confidentiality = str(rng.choice(["Public", "Internal", "Internal", "Confidential"]))
        retention_days = int(rng.choice([365, 730, 1825, 3650]))

        title = f"{category}: {topic.title()} ({date.year})"
        text = (
            f"{category} about {topic}. "
            "Summary: the team reviewed context and preserved rationale for future retrieval. "
            f"Outcome: {str(rng.choice(['proceed', 'defer', 'approve with conditions', 'close']))}. "
            f"Owner: {owner}. Date: {date.date()}."
        )

        rows.append(
            {
                "doc_id": doc_id,
                "title": title,
                "category": category,
                "owner": owner,
                "team": team,
                "date": date,
                "last_review": last_review,
                "status": status,
                "confidentiality": confidentiality,
                "retention_days": retention_days,
                "source_uri": f"https://archive.example.org/records/{doc_id.lower()}",
                "tags": [topic.split()[0].lower(), category.lower().replace(" ", "-"), "synthetic"],
                "supersedes": "",
                "superseded_by": "",
                "text": text,
            }
        )

    docs = pd.DataFrame(rows)
    docs["schema_version"] = "1.0"
    docs["ingested_at"] = pd.to_datetime(docs["date"]) + pd.Timedelta(days=2)
    docs["sha256"] = docs["text"].map(
        lambda x: hashlib.sha256(str(x).encode("utf-8")).hexdigest()[:16]
    )

    docs = compute_governance(docs)

    texts = record_text_series(docs)
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts).toarray()

    return docs, vectorizer, matrix


@st.cache_resource(show_spinner=False)
def load_base_corpus():
    return build_base_corpus()


def filter_mask(
    docs: pd.DataFrame,
    role: str,
    category: str,
    owner: str,
    status: str,
    include_expired: bool,
) -> np.ndarray:
    mask = docs["permission_level"] <= ROLE_LEVEL.get(role, 1)

    if category != "All":
        mask &= docs["category"] == category
    if owner != "All":
        mask &= docs["owner"] == owner
    if status != "All":
        mask &= docs["status"] == status
    if not include_expired:
        mask &= ~docs["expired"]

    return mask.to_numpy()


def rank_documents(
    query: str,
    docs: pd.DataFrame,
    matrix: np.ndarray,
    vectorizer: TfidfVectorizer,
    mask: np.ndarray,
    top_k: int = 10,
) -> pd.DataFrame:
    idx = np.where(mask)[0]

    if idx.size == 0:
        return pd.DataFrame(columns=list(docs.columns) + ["score"])

    sub = docs.iloc[idx].copy()

    if not str(query).strip():
        sub = sub.sort_values("date", ascending=False).head(top_k).copy()
        if len(sub) == 0:
            sub["score"] = []
        elif len(sub) == 1:
            sub["score"] = [1.0]
        else:
            sub["score"] = np.linspace(1.0, 0.5, len(sub))
        return sub

    qvec = vectorizer.transform([query]).toarray()
    cos = cosine_similarity(qvec, matrix[idx])[0]

    terms = tokenize(query)
    denom = max(1, len(terms))

    title_scores = []
    tag_scores = []

    for _, row in sub.iterrows():
        title = str(row["title"]).lower()
        tags = " ".join(map(str, row["tags"])).lower() if isinstance(row["tags"], list) else ""

        ts = sum(1 for t in terms if t in title) / denom
        gs = sum(1 for t in terms if t in tags) / denom

        title_scores.append(ts)
        tag_scores.append(gs)

    sub["score"] = (
        0.64 * cos
        + 0.23 * np.array(title_scores)
        + 0.13 * np.array(tag_scores)
    )

    return sub.sort_values("score", ascending=False).head(top_k).copy()


def make_snippet(text: str, query: str, max_len: int = 280):
    text = str(text)
    terms = tokenize(query)
    sentences = re.split(r"(?<=[.!?])\s+", text)

    best = ""
    best_start = 0
    best_score = -1
    pos = 0

    for sent in sentences:
        start = text.find(sent, pos)
        if start < 0:
            start = pos

        score = sum(1 for t in terms if t in sent.lower())
        if score > best_score:
            best_score = score
            best = sent
            best_start = start

        pos = start + len(sent)

    if not best:
        best = text[:max_len]
        best_start = 0

    snippet = best[:max_len]
    return snippet, int(best_start), int(best_start + len(snippet))


def evaluate_retrieval(docs: pd.DataFrame, vectorizer: TfidfVectorizer, matrix: np.ndarray):
    mask = np.ones(len(docs), dtype=bool)

    hits = []
    reciprocal_ranks = []
    ndcg_scores = []

    for item in GOLD_QUESTIONS:
        result = rank_documents(
            item["question"],
            docs,
            matrix,
            vectorizer,
            mask,
            top_k=10,
        )
        ids = result["doc_id"].tolist()
        expected = item["expected_doc_id"]

        if expected in ids:
            rank = ids.index(expected) + 1
        else:
            rank = 0

        hit = 1 if 0 < rank <= 5 else 0
        rr = 1.0 / rank if rank > 0 else 0.0
        ndcg = 1.0 / np.log2(rank + 1) if 0 < rank <= 5 else 0.0

        hits.append(hit)
        reciprocal_ranks.append(rr)
        ndcg_scores.append(ndcg)

    return {
        "recall_at_5": float(np.mean(hits)) if hits else 0.0,
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "ndcg_at_5": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
        "question_count": len(GOLD_QUESTIONS),
    }


def evaluate_governance(docs: pd.DataFrame):
    expired = docs["expired"]
    flagged = docs["stale_flagged"]

    expired_recall = float(flagged[expired].mean()) if expired.any() else 1.0

    return {
        "expired_recall": expired_recall,
        "stale_rate": float(flagged.mean()) if len(flagged) else 0.0,
        "expired_count": int(expired.sum()),
        "stale_count": int(flagged.sum()),
    }


def retrieval_details(docs: pd.DataFrame, vectorizer: TfidfVectorizer, matrix: np.ndarray):
    mask = np.ones(len(docs), dtype=bool)
    by_id = docs.set_index("doc_id")

    rows = []

    for item in GOLD_QUESTIONS:
        result = rank_documents(
            item["question"],
            docs,
            matrix,
            vectorizer,
            mask,
            top_k=10,
        )
        ids = result["doc_id"].tolist()
        expected = item["expected_doc_id"]

        rank = ids.index(expected) + 1 if expected in ids else 0
        hit = 1 if 0 < rank <= 5 else 0
        rr = 1.0 / rank if rank > 0 else 0.0
        ndcg = 1.0 / np.log2(rank + 1) if 0 < rank <= 5 else 0.0

        exp_row = by_id.loc[expected] if expected in by_id.index else None

        rows.append(
            {
                "question": item["question"],
                "expected_doc_id": expected,
                "expected_category": exp_row["category"] if exp_row is not None else "unknown",
                "expected_owner": exp_row["owner"] if exp_row is not None else "unknown",
                "rank": rank if rank > 0 else None,
                "hit": hit,
                "reciprocal_rank": rr,
                "ndcg_at_5": ndcg,
            }
        )

    return pd.DataFrame(rows)


def write_video(frames, fps: int = 12):
    last_error = None

    for codec in ("libx264", "mpeg4"):
        fd, path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)

        try:
            imageio.mimsave(
                path,
                frames,
                fps=fps,
                codec=codec,
                macro_block_size=1,
            )
            with open(path, "rb") as f:
                return f.read(), "video/mp4"
        except Exception as exc:
            last_error = exc
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    buf = io.BytesIO()
    try:
        imageio.mimsave(buf, frames, format="GIF", fps=fps, loop=0)
        return buf.getvalue(), "image/gif"
    except Exception as exc:
        raise RuntimeError(
            f"MP4 generation failed ({last_error}); GIF fallback failed ({exc})."
        ) from exc


@st.cache_data(show_spinner=False)
def build_demo_video_bytes():
    docs, vectorizer, matrix = load_base_corpus()

    query = "Which authentication standard was adopted for field devices and why?"
    mask = np.ones(len(docs), dtype=bool)
    results = rank_documents(query, docs, matrix, vectorizer, mask, top_k=3)

    retrieval = evaluate_retrieval(docs, vectorizer, matrix)
    governance = evaluate_governance(docs)

    if results.empty:
        result_rows = [
            ("NO-RESULT", "No visible result", "—", "—", 0.0),
            ("NO-RESULT", "No visible result", "—", "—", 0.0),
            ("NO-RESULT", "No visible result", "—", "—", 0.0),
        ]
    else:
        result_rows = []
        for _, row in results.iterrows():
            result_rows.append(
                (
                    str(row["doc_id"]),
                    str(row["title"]),
                    str(row["owner"]),
                    str(row["date"])[:10],
                    float(row["score"]),
                )
            )
        while len(result_rows) < 3:
            result_rows.append(("—", "—", "—", "—", 0.0))

    total_frames = 72
    fps = 12
    width, height = 960, 540

    f_header = get_font(32)
    f_sub = get_font(21)
    f_body = get_font(17)
    f_small = get_font(13)

    frames = []

    for t in range(total_frames):
        img = Image.new("RGB", (width, height), (13, 19, 33))
        d = ImageDraw.Draw(img)

        d.rectangle([0, 0, width, 72], fill=(24, 36, 64))
        d.text((24, 16), "Institutional Memory Live Demo", font=f_header, fill=(240, 246, 255))
        d.text(
            (24, 50),
            "Synthetic archive · provenance · stale detection · hybrid search",
            font=f_small,
            fill=(160, 185, 220),
        )

        progress = t / max(1, total_frames - 1)
        d.rectangle([24, 84, width - 24, 92], fill=(40, 55, 85))
        d.rectangle([24, 84, 24 + int((width - 48) * progress), 92], fill=(80, 200, 160))

        d.text((24, 108), "Ingested records", font=f_sub, fill=(220, 235, 255))

        window = (t * 2) % max(1, len(docs) - 8)
        for i in range(8):
            row = docs.iloc[(window + i) % len(docs)]
            y = 138 + i * 36
            dot_color = (255, 120, 120) if bool(row["stale_flagged"]) else (120, 220, 170)

            d.ellipse([28, y + 8, 38, y + 18], fill=dot_color)
            d.text(
                (46, y),
                f"{row['doc_id']} · {row['category']} · {str(row['date'])[:10]}",
                font=f_body,
                fill=(230, 240, 255),
            )
            d.text((46, y + 18), str(row["title"])[:58], font=f_small, fill=(150, 175, 205))

        d.text((360, 108), "Question", font=f_sub, fill=(220, 235, 255))
        d.text((360, 138), query[:54], font=f_body, fill=(255, 230, 150))
        d.text((360, 160), query[54:108] if len(query) > 54 else "", font=f_body, fill=(255, 230, 150))

        y = 194
        for j in range(3):
            doc_id, title, owner, date_str, score = result_rows[j]

            d.rectangle([360, y, width - 40, y + 64], outline=(70, 95, 135), width=1)
            d.text((370, y + 8), f"Top {j + 1}: {doc_id} · {owner}", font=f_body, fill=(240, 248, 255))
            d.text((370, y + 30), title[:56], font=f_small, fill=(180, 205, 235))
            d.text(
                (370, y + 46),
                f"score={score:.3f} · {date_str}",
                font=f_small,
                fill=(130, 210, 170),
            )
            y += 72

        d.text((360, 424), "Metrics", font=f_sub, fill=(220, 235, 255))

        metric_values = [
            ("Recall@5", retrieval["recall_at_5"]),
            ("Expired flagged", governance["expired_recall"]),
            ("Stale rate", governance["stale_rate"]),
        ]

        x = 360
        for label, val in metric_values:
            d.rectangle([x, 454, x + 170, 504], fill=(30, 45, 75))
            d.text((x + 10, 462), label, font=f_small, fill=(170, 195, 225))
            d.text((x + 10, 480), f"{val:.0%}", font=f_body, fill=(120, 230, 190))
            x += 185

        event_texts = [
            "SHA-256 fingerprint recorded",
            "Provenance span linked to source",
            "Stale ADR flagged for review",
            "Supersedes chain updated",
            "Access log written",
            "Retention policy evaluated",
        ]
        ev = event_texts[(t // 6) % len(event_texts)]

        d.rectangle([0, 510, width, height], fill=(20, 30, 50))
        ticker_x = width - ((t * 14) % (width + max(1, len(ev) * 10)))
        d.text((ticker_x, 516), ev, font=f_body, fill=(255, 210, 120))

        frames.append(np.asarray(img))

    return write_video(frames, fps=fps)


def embed_video(video_bytes: bytes, mime: str):
    if mime.startswith("video/"):
        b64 = base64.b64encode(video_bytes).decode("ascii")
        html = f'''
        <video autoplay muted loop controls playsinline
               style="width:100%; max-height:560px; border-radius:12px; background:#000;">
          <source src="data:{mime};base64,{b64}" type="{mime}">
        </video>
        '''
        components.html(html, height=580)
    else:
        st.image(video_bytes, use_container_width=True)


def init_session():
    if st.session_state.get("initialized", False):
        return

    base_docs, vectorizer, base_matrix = load_base_corpus()

    st.session_state.docs = base_docs.copy()
    st.session_state.vectorizer = vectorizer
    st.session_state.matrix = base_matrix.copy()
    st.session_state.rng = np.random.default_rng(SEED + 1)

    st.session_state.tick = 0
    st.session_state.ingest_count = 0
    st.session_state.events = []
    st.session_state.history = []
    st.session_state.review_log = []

    st.session_state.simulation_running = True
    st.session_state.role = "Steward"
    st.session_state.query = "Which authentication standard was adopted for field devices and why?"
    st.session_state.category = "All"
    st.session_state.owner = "All"
    st.session_state.status = "All"
    st.session_state.include_expired = True
    st.session_state.top_k = 8

    st.session_state.start_time = time.monotonic()
    st.session_state.last_answer = ""

    refresh_metrics()

    st.session_state.initialized = True


def append_event(event_type: str, message: str, doc_id: str = ""):
    ss = st.session_state
    sim_time = (DEMO_NOW + pd.Timedelta(seconds=ss.tick * 30)).strftime("%H:%M:%S")

    ss.events.append(
        {
            "tick": ss.tick,
            "sim_time": sim_time,
            "type": event_type,
            "doc_id": doc_id,
            "message": message,
        }
    )

    if len(ss.events) > 300:
        ss.events = ss.events[-300:]


def refresh_metrics():
    ss = st.session_state

    governance = evaluate_governance(ss.docs)
    retrieval = evaluate_retrieval(ss.docs, ss.vectorizer, ss.matrix)

    ss.metrics = {
        "documents": len(ss.docs),
        **governance,
        **retrieval,
    }


def ingest_synthetic_doc(rng, source: str = "manual", update_metrics: bool = True):
    ss = st.session_state
    ss.ingest_count += 1

    doc_id = f"SIM-{ss.ingest_count:04d}"
    category = str(rng.choice(CATEGORIES))
    topic = str(rng.choice(TOPICS))
    owner = str(rng.choice(OWNERS))
    team = str(rng.choice(TEAMS))
    status = str(rng.choice(["Proposed", "Accepted", "Draft"]))
    confidentiality = str(rng.choice(["Public", "Internal", "Internal", "Confidential"]))
    retention_days = int(rng.choice([365, 730, 1825, 3650]))

    days_ago = int(rng.integers(0, 46))
    date = DEMO_NOW - pd.Timedelta(days=days_ago)

    review_span = max(1, min(180, days_ago + 1))
    review_gap = int(rng.integers(0, review_span))
    last_review = date + pd.Timedelta(days=review_gap)
    if last_review > DEMO_NOW:
        last_review = DEMO_NOW

    text = (
        f"Simulated {category} about {topic}. "
        "Decision or note: preserve context for future retrieval. "
        "Rationale: reduce loss of organizational memory. "
        "Alternatives considered: discard thread, summarize only. "
        f"Owner: {owner}. Date: {date.date()}."
    )

    doc = {
        "doc_id": doc_id,
        "title": f"{category}: {topic.title()} (simulated)",
        "category": category,
        "owner": owner,
        "team": team,
        "date": date,
        "last_review": last_review,
        "status": status,
        "confidentiality": confidentiality,
        "retention_days": retention_days,
        "source_uri": f"https://archive.example.org/records/{doc_id.lower()}",
        "tags": [topic.split()[0].lower(), category.lower().replace(" ", "-"), "simulated"],
        "supersedes": "",
        "superseded_by": "",
        "text": text,
        "schema_version": "1.0",
        "ingested_at": DEMO_NOW,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
    }

    new_df = compute_governance(pd.DataFrame([doc]))

    ss.docs = pd.concat([ss.docs, new_df], ignore_index=True)

    vec = ss.vectorizer.transform(record_text_series(new_df)).toarray()
    ss.matrix = np.vstack([ss.matrix, vec])

    append_event("ingest", f"Ingested simulated record from {source}", doc_id)

    if update_metrics:
        refresh_metrics()


def advance_simulation():
    ss = st.session_state
    ss.tick += 1

    rng = ss.rng
    roll = float(rng.random())

    if roll < 0.22:
        ingest_synthetic_doc(rng, source="stream", update_metrics=False)

    elif roll < 0.50:
        query = str(rng.choice([ss.query] + [g["question"] for g in GOLD_QUESTIONS]))
        mask = filter_mask(ss.docs, "Admin", "All", "All", "All", True)
        top = rank_documents(query, ss.docs, ss.matrix, ss.vectorizer, mask, top_k=1)

        if not top.empty:
            row = top.iloc[0]
            append_event(
                "access",
                f"Retrieved {row['doc_id']} for query: {query[:64]}",
                str(row["doc_id"]),
            )
        else:
            append_event("access", f"No visible result for query: {query[:64]}")

    elif roll < 0.68:
        candidates = ss.docs[ss.docs["stale_score"] > 0.55]

        if not candidates.empty:
            pos = int(rng.integers(0, len(candidates)))
            row = candidates.iloc[pos]
            old_review = row["last_review"]

            ss.docs.at[row.name, "last_review"] = DEMO_NOW
            ss.review_log.append({"doc_id": str(row["doc_id"]), "old_last_review": str(old_review)})

            append_event(
                "review",
                f"Automatic review applied to {row['doc_id']}",
                str(row["doc_id"]),
            )
        else:
            append_event("review", "No stale candidate available for automatic review")

    elif roll < 0.86:
        candidates = ss.docs[ss.docs["stale_flagged"]]

        if not candidates.empty:
            pos = int(rng.integers(0, len(candidates)))
            row = candidates.iloc[pos]
            append_event(
                "stale_flag",
                f"Flagged {row['doc_id']} as stale (score {row['stale_score']:.2f})",
                str(row["doc_id"]),
            )
        else:
            append_event("stale_flag", "No stale items detected in current window")

    else:
        candidates = ss.docs[(ss.docs["days_to_expiry"] >= 0) & (ss.docs["days_to_expiry"] <= 120)]

        if not candidates.empty:
            pos = int(rng.integers(0, len(candidates)))
            row = candidates.iloc[pos]
            append_event(
                "retention_notice",
                f"Retention notice for {row['doc_id']} expires in {int(row['days_to_expiry'])} days",
                str(row["doc_id"]),
            )
        else:
            append_event("retention_notice", "No near-expiry records in current window")

    ss.docs = compute_governance(ss.docs)
    refresh_metrics()

    ss.history.append(
        {
            "tick": ss.tick,
            "records": len(ss.docs),
            "stale": int(ss.docs["stale_flagged"].sum()),
            "expired": int(ss.docs["expired"].sum()),
            "recall_at_5": ss.metrics["recall_at_5"],
        }
    )

    if len(ss.history) > 240:
        ss.history = ss.history[-240:]


def render_metrics_row():
    ss = st.session_state
    m = ss.metrics
    uptime = time.monotonic() - ss.start_time

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    c1.metric("Records", m["documents"])
    c2.metric("Recall@5", f"{m['recall_at_5']:.0%}")
    c3.metric("MRR", f"{m['mrr']:.2f}")
    c4.metric("Expired flagged", f"{m['expired_recall']:.0%}")
    c5.metric("Stale rate", f"{m['stale_rate']:.0%}")
    c6.metric("Tick / uptime", f"{ss.tick} · {uptime:.0f}s")


def live_fragment():
    ss = st.session_state

    if ss.simulation_running:
        advance_simulation()

    render_metrics_row()

    col_events, col_preview = st.columns([2.2, 1.3])

    with col_events:
        st.subheader("Live event stream")

        if ss.events:
            ev = pd.DataFrame(ss.events[-12:]).iloc[::-1]
            st.dataframe(
                ev[["sim_time", "type", "doc_id", "message"]],
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("Waiting for first simulation event...")

    with col_preview:
        st.subheader("Retrieval preview")

        mask = filter_mask(
            ss.docs,
            ss.role,
            ss.category,
            ss.owner,
            ss.status,
            ss.include_expired,
        )
        res = rank_documents(ss.query, ss.docs, ss.matrix, ss.vectorizer, mask, top_k=3)

        if res.empty:
            st.warning("No visible records for current filters.")
        else:
            for _, row in res.iterrows():
                st.markdown(
                    f"**{row['doc_id']}** · {row['category']} · {str(row['date'])[:10]}"
                )
                st.caption(
                    f"{row['owner']} · score {row['score']:.3f} · stale {row['stale_score']:.2f}"
                )


def render_sidebar():
    ss = st.session_state

    st.sidebar.title("Controls")

    st.sidebar.selectbox("Role", list(ROLE_LEVEL.keys()), key="role")

    docs = ss.docs

    categories = ["All"] + sorted(docs["category"].dropna().unique().tolist())
    if ss.category not in categories:
        ss.category = "All"
    st.sidebar.selectbox("Category", categories, key="category")

    owners = ["All"] + sorted(docs["owner"].dropna().unique().tolist())
    if ss.owner not in owners:
        ss.owner = "All"
    st.sidebar.selectbox("Owner", owners, key="owner")

    statuses = ["All"] + sorted(docs["status"].dropna().unique().tolist())
    if ss.status not in statuses:
        ss.status = "All"
    st.sidebar.selectbox("Status", statuses, key="status")

    st.sidebar.text_input("Question", key="query")
    st.sidebar.slider("Top K", min_value=3, max_value=20, key="top_k")
    st.sidebar.checkbox("Include expired records", key="include_expired")
    st.sidebar.checkbox("Live simulation", value=True, key="simulation_running")

    c1, c2 = st.sidebar.columns(2)

    if c1.button("Run 10 ticks"):
        for _ in range(10):
            advance_simulation()
        st.rerun()

    if c2.button("Ingest record"):
        ingest_synthetic_doc(ss.rng, source="manual", update_metrics=True)
        st.rerun()

    if st.sidebar.button("Reset demo"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.sidebar.caption(
        "Synthetic data only. No paid API keys. CPU-only defaults. "
        "Generated MP4/GIF demo is cached for speed."
    )


def render_search_tab():
    ss = st.session_state

    st.subheader("Search & provenance")

    mask = filter_mask(
        ss.docs,
        ss.role,
        ss.category,
        ss.owner,
        ss.status,
        ss.include_expired,
    )

    results = rank_documents(
        ss.query,
        ss.docs,
        ss.matrix,
        ss.vectorizer,
        mask,
        top_k=int(ss.top_k),
    )

    if results.empty:
        st.info("No records match the current question and filters.")
        return

    display = results.copy()
    display["date"] = display["date"].dt.strftime("%Y-%m-%d")
    display["score"] = display["score"].round(3)
    display["stale_score"] = display["stale_score"].round(2)

    st.dataframe(
        display[
            [
                "doc_id",
                "title",
                "category",
                "owner",
                "date",
                "status",
                "confidentiality",
                "score",
                "stale_score",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )

    choices = results["doc_id"].tolist()
    selected_id = st.selectbox("Inspect source record", choices)
    row = results[results["doc_id"] == selected_id].iloc[0]

    snippet, start, end = make_snippet(row["text"], ss.query)

    st.markdown(f"#### {row['title']}")
    st.markdown(
        f"**Source record:** [{row['doc_id']}]({row['source_uri']}) · "
        f"{str(row['date'])[:10]} · Owner: {row['owner']} · Team: {row['team']}"
    )
    st.markdown(
        f"**Provenance:** SHA-256 `{row['sha256']}` · span `{start}-{end}` · "
        "extractor `tfidf-hybrid-v1` · "
        f"confidence `{row['score']:.3f}` · "
        f"ingested `{str(row['ingested_at'])[:16]}`"
    )

    st.write(snippet)

    if st.button("Generate cited answer"):
        answer = (
            f"According to {row['doc_id']} (“{row['title']}”), dated "
            f"{str(row['date'])[:10]} and owned by {row['owner']}, the archive states: "
            f"“{snippet}” Source: {row['source_uri']}"
        )
        ss.last_answer = answer
        st.success(answer)

    if ss.last_answer:
        st.markdown("**Last generated answer**")
        st.info(ss.last_answer)


def render_timeline_tab():
    ss = st.session_state

    st.subheader("Timeline & supersession chains")

    mask = filter_mask(
        ss.docs,
        ss.role,
        ss.category,
        ss.owner,
        ss.status,
        ss.include_expired,
    )
    docs = ss.docs[mask].copy()

    if docs.empty:
        st.info("No visible records for timeline.")
        return

    docs = docs.sort_values("date")

    st.markdown("**Record volume over time**")
    monthly = docs.set_index("date").resample("MS").size()
    st.line_chart(monthly)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Records by category**")
        st.bar_chart(docs["category"].value_counts())

    with c2:
        st.markdown("**Recent records**")
        recent = docs.tail(12)[["doc_id", "title", "category", "owner", "date"]]
        recent = recent.copy()
        recent["date"] = recent["date"].dt.strftime("%Y-%m-%d")
        st.dataframe(recent, hide_index=True, use_container_width=True)

    st.markdown("**Supersedes / superseded-by chain**")

    choices = docs["doc_id"].tolist()
    chosen = st.selectbox("Select record", choices)
    row = docs[docs["doc_id"] == chosen].iloc[0]

    st.markdown(f"### {row['doc_id']} — {row['title']}")
    st.caption(
        f"{row['category']} · {row['owner']} · {str(row['date'])[:10]} · {row['status']}"
    )

    by_id = ss.docs.set_index("doc_id")

    if row["supersedes"]:
        target = row["supersedes"]
        if target in by_id.index:
            trow = by_id.loc[target]
            st.markdown(
                f"**Supersedes:** [{target}]({trow['source_uri']}) — {trow['title']}"
            )
        else:
            st.markdown(f"**Supersedes:** {target}")

    if row["superseded_by"]:
        target = row["superseded_by"]
        if target in by_id.index:
            trow = by_id.loc[target]
            st.markdown(
                f"**Superseded by:** [{target}]({trow['source_uri']}) — {trow['title']}"
            )
        else:
            st.markdown(f"**Superseded by:** {target}")

    if not row["supersedes"] and not row["superseded_by"]:
        st.info("This record has no declared supersession chain.")


def render_governance_tab():
    ss = st.session_state

    st.subheader("Governance, staleness & retention")

    mask = filter_mask(
        ss.docs,
        ss.role,
        ss.category,
        ss.owner,
        ss.status,
        ss.include_expired,
    )
    docs = ss.docs[mask].copy()

    gov_df = docs[docs["stale_flagged"]].sort_values("stale_score", ascending=False)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Visible records", len(docs))
    c2.metric("Stale flagged", int(docs["stale_flagged"].sum()))
    c3.metric("Expired", int(docs["expired"].sum()))
    c4.metric("Expired recall", f"{ss.metrics['expired_recall']:.0%}")

    if gov_df.empty:
        st.success("No stale or expired records visible under current filters.")
    else:
        out = gov_df.copy()
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")
        out["expiry_date"] = out["expiry_date"].dt.strftime("%Y-%m-%d")
        out["last_review"] = out["last_review"].dt.strftime("%Y-%m-%d")
        out["days_to_expiry"] = out["days_to_expiry"].fillna(9999).astype(int)
        out["stale_score"] = out["stale_score"].round(2)

        st.dataframe(
            out[
                [
                    "doc_id",
                    "title",
                    "owner",
                    "date",
                    "expiry_date",
                    "days_to_expiry",
                    "last_review",
                    "stale_score",
                    "expired",
                    "confidentiality",
                ]
            ],
            hide_index=True,
            use_container_width=True,
        )

        stale_ids = out["doc_id"].tolist()

        chosen = st.selectbox("Select stale record to review", stale_ids)

        b1, b2 = st.columns(2)

        if b1.button("Mark reviewed now"):
            idx = ss.docs.index[ss.docs["doc_id"] == chosen][0]
            old = ss.docs.at[idx, "last_review"]

            ss.review_log.append({"doc_id": chosen, "old_last_review": str(old)})
            ss.docs.at[idx, "last_review"] = DEMO_NOW

            ss.docs = compute_governance(ss.docs)
            refresh_metrics()
            append_event("review", f"Manual review applied to {chosen}", chosen)

            st.success("Review recorded and stale score recomputed.")
            st.rerun()

        if b2.button("Undo last review") and ss.review_log:
            last = ss.review_log.pop()
            doc_id = last["doc_id"]

            if doc_id in ss.docs["doc_id"].values:
                idx = ss.docs.index[ss.docs["doc_id"] == doc_id][0]
                ss.docs.at[idx, "last_review"] = pd.Timestamp(last["old_last_review"])

                ss.docs = compute_governance(ss.docs)
                refresh_metrics()
                append_event("review", f"Undid manual review for {doc_id}", doc_id)

                st.info("Last review undone.")
                st.rerun()

    st.markdown("**Access log**")
    access_events = [e for e in ss.events if e["type"] == "access"]

    if access_events:
        st.dataframe(
            pd.DataFrame(access_events[-20:]).iloc[::-1],
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No access events yet.")


def render_evaluation_tab():
    ss = st.session_state

    st.subheader("Evaluation")

    m = ss.metrics

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recall@5", f"{m['recall_at_5']:.0%}")
    c2.metric("MRR", f"{m['mrr']:.2f}")
    c3.metric("nDCG@5", f"{m['ndcg_at_5']:.2f}")
    c4.metric("Expired flagged", f"{m['expired_recall']:.0%}")

    st.markdown("**Gold question diagnostics**")

    details = retrieval_details(ss.docs, ss.vectorizer, ss.matrix)
    st.dataframe(details, hide_index=True, use_container_width=True)

    st.markdown("**Error analysis by expected category**")

    by_cat = (
        details.groupby("expected_category")
        .agg(
            recall_at_5=("hit", "mean"),
            count=("hit", "size"),
            avg_rank=("rank", lambda x: x.dropna().mean() if x.notna().any() else np.nan),
        )
        .reset_index()
    )
    by_cat["recall_at_5"] = by_cat["recall_at_5"].round(3)
    by_cat["avg_rank"] = by_cat["avg_rank"].round(2)

    st.dataframe(by_cat, hide_index=True, use_container_width=True)

    st.markdown("**Governance checks**")
    st.write(
        f"- Expired records: **{m['expired_count']}**\n"
        f"- Stale-flagged records: **{m['stale_count']}**\n"
        f"- Expired-item detection: **{m['expired_recall']:.0%}**\n"
        f"- Gold questions: **{m['question_count']}**"
    )

    st.markdown("**Known limitations**")
    st.info(
        "This prototype uses clearly labelled synthetic institutional records. "
        "It demonstrates provenance-linked retrieval, staleness detection, access tiers, "
        "and live simulation. It does not prove performance on real confidential archives, "
        "regulated clinical data, production graph stores, or multi-user authorization."
    )


def render_events_tab():
    ss = st.session_state

    st.subheader("Full event log")

    if not ss.events:
        st.info("No events yet.")
        return

    df = pd.DataFrame(ss.events).iloc[::-1]
    st.dataframe(df, hide_index=True, use_container_width=True)

    if ss.history:
        st.markdown("**Simulation history**")
        hist = pd.DataFrame(ss.history).set_index("tick")
        st.line_chart(hist[["records", "stale", "expired"]])


def main():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🗂️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    with st.spinner("Initializing synthetic institutional memory..."):
        init_session()

    st.title(APP_TITLE)
    st.caption(
        "CPU-only · synthetic data · no API keys · provenance-linked retrieval · "
        "live simulation · generated demo video"
    )

    with st.spinner("Generating live demo video..."):
        try:
            video_bytes, mime = build_demo_video_bytes()
        except Exception as exc:
            video_bytes, mime = None, None
            st.error(f"Video generation failed: {exc}")

    if video_bytes is not None:
        with st.expander("Generated live demo video (auto-playing)", expanded=True):
            embed_video(video_bytes, mime)

            ext = "mp4" if mime == "video/mp4" else "gif"
            st.download_button(
                "Download demo video",
                data=video_bytes,
                file_name=f"institutional_memory_demo.{ext}",
                mime=mime,
            )

    render_sidebar()

    if hasattr(st, "fragment"):
        st.fragment(run_every="3s")(live_fragment)()
    else:
        live_fragment()

    st.divider()

    tabs = st.tabs(
        [
            "Search & provenance",
            "Timeline & chains",
            "Governance & retention",
            "Evaluation",
            "Event log",
        ]
    )

    with tabs[0]:
        render_search_tab()

    with tabs[1]:
        render_timeline_tab()

    with tabs[2]:
        render_governance_tab()

    with tabs[3]:
        render_evaluation_tab()

    with tabs[4]:
        render_events_tab()


main()