"""
rag_utils.py
------------
Core utilities for PowerSense AI:
  * PDF / image text extraction (PyPDF + OCR)
  * RAG knowledge base pipeline (chunking, embeddings, FAISS)
  * Groq LLM helper functions used by the different "agents" in app.py

No tariff rates, policies, or complaint URLs are hardcoded here. Anything
that depends on official rules is retrieved from the `knowledge/` folder
at runtime, or is explicitly left for the LLM to mark as
"Requires Verification" when it is not supported by retrieved context.
"""

import io
import os
import json
import re
import glob
from typing import List, Dict, Tuple, Optional

import streamlit as st
from pypdf import PdfReader
from PIL import Image
import pytesseract

# These imports are wrapped in try/except because different langchain
# versions have moved these classes between packages (langchain vs
# langchain_text_splitters, langchain_community vs langchain_huggingface).
# This keeps the app working across the version range in requirements.txt.
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.documents import Document

from groq import Groq


# ---------------------------------------------------------------------------
# CONFIGURATION (change these in one place if needed)
# ---------------------------------------------------------------------------

# Put the Groq model name in a single configuration variable so it can be
# swapped easily if Groq changes their available/free model line-up.
GROQ_MODEL = "llama-3.3-70b-versatile"

# Embedding model used for the RAG knowledge base (small, free, CPU-friendly)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Folder that holds official reference PDFs (tariff schedules, NEPRA
# notifications, FCA/QTA decisions, complaint procedures, etc.)
KNOWLEDGE_DIR = "knowledge"

# Text splitting parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# How many chunks to retrieve per query
TOP_K = 5


# ---------------------------------------------------------------------------
# TEXT EXTRACTION: PDF
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract raw text from an uploaded PDF electricity bill using PyPDF.

    Returns an empty string (never raises) so the calling UI code can show
    a friendly message instead of a stack trace.
    """
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        text_parts = []
        for page in reader.pages:
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""
            text_parts.append(page_text)
        return "\n".join(text_parts).strip()
    except Exception as e:
        st.session_state.setdefault("errors", []).append(f"PDF extraction failed: {e}")
        return ""


# ---------------------------------------------------------------------------
# TEXT EXTRACTION: IMAGE (OCR)
# ---------------------------------------------------------------------------

def extract_text_from_image(file_bytes: bytes) -> str:
    """Extract text from a bill photo/screenshot using Tesseract OCR.

    NOTE: On Streamlit Community Cloud, the `tesseract-ocr` binary must be
    installed via a `packages.txt` file (apt package), otherwise pytesseract
    will raise a TesseractNotFoundError. This is handled gracefully below.
    """
    try:
        image = Image.open(io.BytesIO(file_bytes))
        # Basic preprocessing: convert to grayscale to improve OCR accuracy
        image = image.convert("L")
        text = pytesseract.image_to_string(image)
        return text.strip()
    except pytesseract.TesseractNotFoundError:
        st.session_state.setdefault("errors", []).append(
            "OCR engine (Tesseract) is not installed on this server. "
            "Add a packages.txt file with 'tesseract-ocr' for Streamlit Cloud."
        )
        return ""
    except Exception as e:
        st.session_state.setdefault("errors", []).append(f"OCR failed: {e}")
        return ""


def extract_bill_text(uploaded_file) -> str:
    """Dispatch to the correct extractor based on uploaded file type."""
    if uploaded_file is None:
        return ""

    file_bytes = uploaded_file.getvalue()
    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        text = extract_text_from_pdf(file_bytes)
    elif name.endswith((".jpg", ".jpeg", ".png")):
        text = extract_text_from_image(file_bytes)
    else:
        st.session_state.setdefault("errors", []).append(
            f"Unsupported file type: {uploaded_file.name}"
        )
        return ""

    if not text:
        st.session_state.setdefault("errors", []).append(
            "Could not extract any readable text from the uploaded bill. "
            "Please enter the bill details manually below."
        )
    return text


# ---------------------------------------------------------------------------
# RAG PIPELINE: KNOWLEDGE BASE LOADING + FAISS
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_embeddings():
    """Load and cache the HuggingFace sentence-transformer embedding model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def _load_knowledge_documents() -> List[Document]:
    """Read every PDF in the knowledge/ folder and split it into chunks.

    Each chunk keeps a `source` metadata field (the file name) so the UI
    can show a "Sources Used" section.
    """
    documents: List[Document] = []
    pdf_paths = sorted(glob.glob(os.path.join(KNOWLEDGE_DIR, "*.pdf")))

    if not pdf_paths:
        return documents

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    for path in pdf_paths:
        try:
            reader = PdfReader(path)
            full_text = ""
            for page in reader.pages:
                full_text += (page.extract_text() or "") + "\n"
            if not full_text.strip():
                continue
            chunks = splitter.split_text(full_text)
            source_name = os.path.basename(path)
            for chunk in chunks:
                documents.append(Document(page_content=chunk, metadata={"source": source_name}))
        except Exception as e:
            st.session_state.setdefault("errors", []).append(
                f"Could not read knowledge file {os.path.basename(path)}: {e}"
            )

    return documents


@st.cache_resource(show_spinner=False)
def build_or_load_vectorstore():
    """Build the FAISS vector store from the knowledge/ folder.

    Cached with st.cache_resource so it is only rebuilt once per app
    session/deployment, not on every user interaction.
    Returns None if the knowledge base is empty (graceful fallback).
    """
    documents = _load_knowledge_documents()
    if not documents:
        return None

    embeddings = get_embeddings()
    try:
        vectorstore = FAISS.from_documents(documents, embeddings)
        return vectorstore
    except Exception as e:
        st.session_state.setdefault("errors", []).append(f"FAISS index build failed: {e}")
        return None


def retrieve_relevant_chunks(query: str, vectorstore, k: int = TOP_K) -> List[Dict]:
    """Similarity search over the knowledge base.

    Returns a list of {"text": ..., "source": ...} dicts. Returns an empty
    list (not an error) if there is no knowledge base yet, so the rest of
    the app can still run in a degraded ("Requires Verification"-heavy) mode.
    """
    if vectorstore is None or not query:
        return []
    try:
        results = vectorstore.similarity_search(query, k=k)
        return [
            {"text": r.page_content, "source": r.metadata.get("source", "Unknown source")}
            for r in results
        ]
    except Exception as e:
        st.session_state.setdefault("errors", []).append(f"Knowledge retrieval failed: {e}")
        return []


# ---------------------------------------------------------------------------
# GROQ LLM HELPERS
# ---------------------------------------------------------------------------

def get_groq_client(api_key: str) -> Optional[Groq]:
    if not api_key:
        return None
    try:
        return Groq(api_key=api_key)
    except Exception as e:
        st.session_state.setdefault("errors", []).append(f"Could not initialize Groq client: {e}")
        return None


def call_groq_chat(
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    model: str = GROQ_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 1800,
) -> Optional[str]:
    """Send a chat completion request to Groq and return the text response.

    Returns None (never raises) on any failure so the UI can show a
    friendly error message instead of crashing.
    """
    client = get_groq_client(api_key)
    if client is None:
        return None

    try:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        st.session_state.setdefault("errors", []).append(f"Groq API call failed: {e}")
        return None


def safe_json_parse(text: Optional[str]) -> Optional[dict]:
    """Extract and parse the first JSON object found in an LLM response."""
    if not text:
        return None
    # Strip markdown code fences if present
    cleaned = re.sub(r"```json|```", "", text).strip()
    # Try direct parse first
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    # Fallback: find the outermost {...} block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# AGENT 1 — BILL EXTRACTION AGENT
# ---------------------------------------------------------------------------

BILL_FIELDS_SCHEMA = [
    "consumer_number", "meter_number", "provider", "billing_month",
    "issue_date", "due_date", "previous_reading", "current_reading",
    "units_consumed", "tariff_category", "previous_bill_amount",
    "current_bill_amount", "amount_payable", "electricity_charges",
    "taxes", "surcharges", "fca", "quarterly_adjustment",
    "fixed_charges", "arrears", "other_charges",
]


def extract_bill_fields_with_llm(api_key: str, bill_text: str) -> Dict:
    """Agent 1: Ask the LLM to pull structured fields out of raw bill text.

    Only fields that are actually present in the text should be filled in;
    everything else must be returned as null so the UI can ask the user
    to fill it manually rather than the model inventing numbers.
    """
    system_prompt = (
        "You are a careful data-extraction assistant for Pakistani electricity bills. "
        "Extract ONLY information that is explicitly present in the provided bill text. "
        "Never guess or invent a number. If a field is not present or unclear, set it to null. "
        "Respond with STRICT JSON ONLY, no explanation, no markdown fences."
    )
    user_prompt = (
        "Extract the following fields from this electricity bill text and return them as a "
        f"JSON object with exactly these keys: {json.dumps(BILL_FIELDS_SCHEMA)}.\n\n"
        "Numeric fields (units_consumed, previous_reading, current_reading, "
        "previous_bill_amount, current_bill_amount, amount_payable, electricity_charges, "
        "taxes, surcharges, fca, quarterly_adjustment, fixed_charges, arrears, other_charges) "
        "should be numbers (no currency symbols/commas) or null.\n\n"
        f"BILL TEXT:\n{bill_text[:6000]}"
    )

    raw = call_groq_chat(api_key, system_prompt, user_prompt, temperature=0.0)
    parsed = safe_json_parse(raw)
    if not parsed:
        return {field: None for field in BILL_FIELDS_SCHEMA}

    # Ensure every expected key exists
    for field in BILL_FIELDS_SCHEMA:
        parsed.setdefault(field, None)
    return parsed


# ---------------------------------------------------------------------------
# AGENT 2 & 3 — TARIFF ANALYSIS + CHARGE VERIFICATION
# ---------------------------------------------------------------------------

LANGUAGE_INSTRUCTIONS = {
    "English": "Respond in clear, simple English.",
    "Urdu": "Respond in Urdu script (اردو).",
    "Roman Urdu": "Respond in Roman Urdu (Urdu written using English/Latin letters).",
}


def analyze_and_verify_bill(
    api_key: str,
    bill_data: Dict,
    provider: str,
    consumer_category: str,
    language: str,
    context_chunks: List[Dict],
) -> Optional[Dict]:
    """Agent 2 (Tariff Analysis) + Agent 3 (Charge Verification) combined.

    Uses retrieved knowledge-base context to explain each bill component and
    classify it as Explained / Applicable / Requires Verification /
    Potential Billing Issue. Never invents tariff rates — if the context
    does not support a specific rate/rule, the item must be marked
    "Requires Verification".
    """
    context_text = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks
    ) or "NO OFFICIAL KNOWLEDGE DOCUMENTS WERE RETRIEVED FOR THIS QUERY."

    lang_instruction = LANGUAGE_INSTRUCTIONS.get(language, LANGUAGE_INSTRUCTIONS["English"])

    system_prompt = (
        "You are a cautious electricity-bill analysis assistant for Pakistani consumers. "
        "You must NEVER call a charge illegal, fake, or fraudulent. Use only these status "
        "labels: 'Explained', 'Applicable', 'Requires Verification', 'Potential Billing Issue'. "
        "Only use retrieved official context to justify tariff rates or rules. If the context "
        "does not clearly support a charge, classify it as 'Requires Verification' rather than "
        "guessing. Do not fabricate tariff rates, percentages, or policies that are not present "
        "in the provided context. " + lang_instruction + " "
        "Respond with STRICT JSON ONLY, no explanation outside the JSON, no markdown fences."
    )

    user_prompt = f"""
Consumer category: {consumer_category}
Electricity provider/DISCO: {provider}

Extracted bill data (numbers are in PKR unless null):
{json.dumps(bill_data, indent=2)}

Retrieved official knowledge context (may be empty):
{context_text}

Return a JSON object with exactly this structure:
{{
  "bill_summary": {{
     "provider": string,
     "billing_month": string or null,
     "units_consumed": number or null,
     "total_bill": number or null,
     "due_date": string or null
  }},
  "charge_breakdown": [
     {{
        "component": string,
        "amount": number or null,
        "status": "Explained" | "Applicable" | "Requires Verification" | "Potential Billing Issue",
        "explanation": string,
        "source_used": string or null
     }}
  ],
  "consumption_analysis": {{
     "previous_units": number or null,
     "current_units": number or null,
     "difference": number or null,
     "percent_change": number or null,
     "note": string
  }},
  "why_bill_is_high": string,
  "attention_items": [
     {{
        "charge_name": string,
        "amount": number or null,
        "flag_reason": string,
        "status": "Requires Verification" | "Potential Billing Issue",
        "source_used": string or null,
        "what_to_verify": string
     }}
  ],
  "sources_used": [string]
}}

Rules:
- Only include components in "charge_breakdown" that were actually detected in the bill data (non-null or clearly mentioned).
- "attention_items" should only include items that are unclear or need the consumer/DISCO to verify — do not force items into this list if everything looks explainable.
- If consumption/reading data is insufficient, set consumption_analysis note to explain that previous consumption data was not available.
- "sources_used" must list only the source file names actually used to justify a claim; if no knowledge documents were retrieved, return an empty list.
"""

    raw = call_groq_chat(api_key, system_prompt, user_prompt, temperature=0.2, max_tokens=2200)
    return safe_json_parse(raw)


# ---------------------------------------------------------------------------
# AGENT 4 — COMPLAINT ASSISTANT AGENT
# ---------------------------------------------------------------------------

def generate_complaint_package(
    api_key: str,
    bill_data: Dict,
    analysis: Dict,
    provider: str,
    language: str,
) -> Optional[Dict]:
    """Agent 4: Decide whether a complaint is warranted and draft supporting text.

    This never claims a complaint will succeed, and never invents an
    official complaint URL — the URL is supplied by the calling UI code
    from a fixed, verified constant, not by the LLM.
    """
    lang_instruction = LANGUAGE_INSTRUCTIONS.get(language, LANGUAGE_INSTRUCTIONS["English"])

    system_prompt = (
        "You are a consumer-assistance agent helping a Pakistani electricity consumer decide "
        "whether to file a complaint about their bill. You must never guarantee a complaint will "
        "succeed. You must never invent a URL — do not include any URLs in your response. " +
        lang_instruction + " Respond with STRICT JSON ONLY, no markdown fences."
    )

    user_prompt = f"""
Provider/DISCO: {provider}
Bill data: {json.dumps(bill_data, indent=2)}
Prior bill analysis (charge breakdown + attention items): {json.dumps(analysis, indent=2)}

Return JSON with exactly this structure:
{{
  "should_consider_complaint": true or false,
  "likely_complaint_category": string,
  "reason_summary": string,
  "evidence_to_keep": [string],
  "draft_complaint_text": string,
  "contact_provider_first_note": string
}}

"draft_complaint_text" should be a short (4-6 sentence), polite, factual description the
consumer could submit, referencing only the flagged items from the analysis — do not
accuse the provider of wrongdoing, only ask for verification/clarification.
"""

    raw = call_groq_chat(api_key, system_prompt, user_prompt, temperature=0.3, max_tokens=1200)
    return safe_json_parse(raw)
