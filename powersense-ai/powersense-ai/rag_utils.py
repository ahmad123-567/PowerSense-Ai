"""
rag_utils.py
------------
Core utilities for PowerSense AI:
  * PDF / image text extraction (PyPDF + OCR)
  * OCR quality checks and bill-reading validation
  * RAG knowledge base pipeline (chunking, embeddings, FAISS)
  * Groq LLM helper functions used by the different agents in app.py

The OCR pipeline is deliberately conservative: it improves the uploaded
image before OCR, performs several OCR passes, and then validates important
meter-reading relationships before the values are used for analysis.

No tariff rates, policies, or complaint URLs are hardcoded here. Anything
that depends on official rules is retrieved from the knowledge/ folder at
runtime, or is explicitly left for the LLM to mark as "Requires Verification"
when it is not supported by retrieved context.
"""

import io
import os
import json
import re
import glob
from typing import List, Dict, Optional

import streamlit as st
from pypdf import PdfReader
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import pytesseract

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from groq import Groq


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

GROQ_MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Base directory setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Check multiple potential knowledge paths
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")

# Fallback path if files are nested inside powersense-ai/powersense-ai/knowledge
if not os.path.exists(KNOWLEDGE_DIR) or not os.listdir(KNOWLEDGE_DIR):
    alt_path = os.path.join(BASE_DIR, "powersense-ai", "powersense-ai", "knowledge")
    if os.path.exists(alt_path):
        KNOWLEDGE_DIR = alt_path
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 5


# ---------------------------------------------------------------------------
# ERROR HELPER
# ---------------------------------------------------------------------------

def add_error(message: str) -> None:
    """Store a friendly error message without crashing the Streamlit app."""
    st.session_state.setdefault("errors", []).append(message)


# ---------------------------------------------------------------------------
# TEXT EXTRACTION: PDF
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract raw text from an uploaded PDF electricity bill."""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        text_parts = []
        for page in reader.pages:
            try:
                text_parts.append(page.extract_text() or "")
            except Exception:
                text_parts.append("")
        return "\n".join(text_parts).strip()
    except Exception as e:
        add_error(f"PDF extraction failed: {e}")
        return ""


# ---------------------------------------------------------------------------
# OCR HELPERS
# ---------------------------------------------------------------------------

def _prepare_ocr_variants(image: Image.Image) -> List[Image.Image]:
    """Create several OCR-friendly versions of a bill image."""
    image = image.convert("RGB")

    width, height = image.size

    # Upscale small text. Limit the largest dimension to avoid excessive
    # memory/CPU use on Streamlit Community Cloud.
    scale = 2.0
    max_dimension = 5000
    if max(width, height) * scale > max_dimension:
        scale = max_dimension / max(width, height)

    image = image.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))),
        Image.Resampling.LANCZOS,
    )

    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray, cutoff=1)
    gray = ImageEnhance.Contrast(gray).enhance(1.7)
    gray = gray.filter(ImageFilter.SHARPEN)

    # A thresholded copy can help with faint printed digits and table text.
    threshold = gray.point(lambda p: 255 if p > 180 else 0)

    # A softer contrast variant helps when the original bill has gray boxes.
    soft = ImageEnhance.Contrast(ImageOps.grayscale(image)).enhance(1.35)
    soft = soft.filter(ImageFilter.SHARPEN)

    return [gray, soft, threshold]


def _clean_ocr_text(text: str) -> str:
    """Normalize obvious OCR whitespace without changing bill numbers."""
    text = text.replace("\x0c", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _deduplicate_ocr_lines(text: str) -> str:
    """Remove exact duplicate OCR lines while preserving order."""
    seen = set()
    output = []
    for line in text.splitlines():
        normalized = re.sub(r"\s+", " ", line).strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(normalized)
    return "\n".join(output)


def extract_text_from_image(file_bytes: bytes) -> str:
    """
    Extract bill text with enhanced OCR.

    Three preprocessed image variants and multiple Tesseract page-segmentation
    modes are used. The output is combined and lightly deduplicated so the LLM
    gets more useful bill text without blindly trusting a single OCR pass.
    """
    try:
        image = Image.open(io.BytesIO(file_bytes))
        variants = _prepare_ocr_variants(image)

        results = []
        configs = ["--oem 3 --psm 6", "--oem 3 --psm 11"]

        for variant_index, variant in enumerate(variants, start=1):
            for config in configs:
                try:
                    text = pytesseract.image_to_string(
                        variant,
                        lang="eng",
                        config=config,
                    )
                    text = _clean_ocr_text(text)
                    if text:
                        results.append(
                            f"[OCR PASS {variant_index} / {config}]\n{text}"
                        )
                except pytesseract.TesseractError:
                    # Continue with the other OCR passes.
                    continue

        if not results:
            return ""

        combined = "\n\n".join(results)
        return _deduplicate_ocr_lines(combined)

    except pytesseract.TesseractNotFoundError:
        add_error(
            "OCR engine (Tesseract) is not installed on this server. "
            "Add tesseract-ocr and tesseract-ocr-eng to packages.txt."
        )
        return ""
    except Exception as e:
        add_error(f"OCR failed: {e}")
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
        add_error(f"Unsupported file type: {uploaded_file.name}")
        return ""

    if not text:
        add_error(
            "Could not extract readable text from the uploaded bill. "
            "Please try a clearer photo/PDF, or complete the fields manually."
        )
    return text


# ---------------------------------------------------------------------------
# BILL OCR QUALITY CHECKS + NUMERIC VALIDATION
# ---------------------------------------------------------------------------

_NUMBER = r"([0-9][0-9, ]*)"


def _to_number(value) -> Optional[float]:
    """Convert a number-like value to float, or return None."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        cleaned = re.sub(r"[^0-9.\-]", "", str(value))
        return float(cleaned) if cleaned else None
    except Exception:
        return None


def _find_number_after_labels(text: str, labels: List[str]) -> Optional[float]:
    """Find a number appearing near one of the supplied labels."""
    for label in labels:
        pattern = rf"{label}[^0-9]{{0,80}}{_NUMBER}"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            try:
                return float(match.group(1).replace(",", "").replace(" ", ""))
            except ValueError:
                pass
    return None


def quick_bill_ocr_checks(bill_text: str) -> Dict:
    """
    Perform lightweight checks on raw OCR output.

    This does not replace the LLM extraction. It simply warns when key fields
    look suspicious or when meter readings imply a different unit count.
    """
    detected = {}
    warnings = []

    if not bill_text:
        return {"detected": detected, "warnings": ["No OCR text is available."]}

    previous = _find_number_after_labels(
        bill_text,
        [
            r"previous\s+reading",
            r"previous\s+meter\s+reading",
            r"prev(?:ious)?\s*reading",
        ],
    )
    current = _find_number_after_labels(
        bill_text,
        [
            r"present\s+reading",
            r"current\s+reading",
            r"current\s+meter\s+reading",
            r"present\s+meter\s+reading",
        ],
    )
    units = _find_number_after_labels(
        bill_text,
        [r"\bunits\b", r"units\s+consumed", r"consumption"],
    )

    if previous is not None:
        detected["previous_reading"] = int(previous) if previous.is_integer() else previous
    if current is not None:
        detected["current_reading"] = int(current) if current.is_integer() else current
    if units is not None:
        detected["units_consumed"] = int(units) if units.is_integer() else units

    if previous is not None and current is not None:
        calculated = current - previous
        if calculated < 0:
            warnings.append(
                "Previous and current meter readings do not form a normal increasing sequence. "
                "Please verify the readings manually."
            )
        elif units is not None and abs(calculated - units) > 0.5:
            warnings.append(
                f"OCR reading check found a mismatch: current reading ({int(current)}) - "
                f"previous reading ({int(previous)}) = {int(calculated)} units, "
                f"but OCR also detected about {int(units)} units. Please verify the bill."
            )

    # A bill OCR result with lots of text but no obvious reading labels is not
    # necessarily wrong, but it is worth showing a gentle verification notice.
    if len(bill_text.strip()) < 100:
        warnings.append("Very little text was detected. A clearer/higher-resolution bill image may help.")

    return {"detected": detected, "warnings": warnings}


def validate_bill_data(bill_data: Dict) -> Dict:
    """
    Validate important numeric relationships after LLM/manual extraction.

    If both meter readings are available and the current reading is higher,
    calculated consumption is returned. The UI can use that value rather than
    trusting a conflicting OCR/LLM unit count.
    """
    previous = _to_number(bill_data.get("previous_reading"))
    current = _to_number(bill_data.get("current_reading"))
    units = _to_number(bill_data.get("units_consumed"))

    warnings = []
    notes = []
    calculated_units = None

    if previous is not None and current is not None:
        if current < previous:
            warnings.append(
                "Meter reading check failed: current reading is lower than previous reading. "
                "Please verify both readings from the bill."
            )
        else:
            calculated_units = current - previous

            if units is not None and abs(calculated_units - units) > 0.5:
                # OCR can confuse a single digit in a long meter reading. If the
                # current reading and the printed unit count agree with each other,
                # derive the previous reading as a consistency check. Likewise, if
                # previous reading + printed units matches current, keep the current.
                candidate_previous = current - units
                candidate_current = previous + units

                if candidate_previous >= 0:
                    notes.append(
                        f"OCR values were inconsistent. Current reading ({int(current)}) "
                        f"minus printed units ({int(units)}) gives a consistency-check "
                        f"previous reading of {int(candidate_previous)}."
                    )

                # Prefer the derived value only when it is a clean integer and the
                # discrepancy looks like a normal OCR digit error. This is still shown
                # as a verification note rather than a claim that the original OCR was correct.
                if candidate_previous >= 0 and abs(candidate_previous - previous) <= 100:
                    previous = candidate_previous
                    calculated_units = units
                    notes.append(
                        "The previous reading was corrected for analysis using the "
                        "current reading and printed unit count; visually verify it on the bill."
                    )
                elif abs(candidate_current - current) <= 100:
                    current = candidate_current
                    calculated_units = units
                    notes.append(
                        "The current reading was corrected for analysis using the "
                        "previous reading and printed unit count; visually verify it on the bill."
                    )
                else:
                    warnings.append(
                        f"Units mismatch: the readings calculate to {int(calculated_units)} units, "
                        f"while the extracted units value is {int(units)}. Please visually verify "
                        "the meter readings and units."
                    )

            notes.append(
                f"Calculated consumption from meter readings: {int(current)} - "
                f"{int(previous)} = {int(calculated_units)} units."
            )

    # Basic sanity checks. These do not claim that a bill is wrong.
    if units is not None and units < 0:
        warnings.append("Units consumed cannot be negative. Please verify the extracted value.")

    return {
        "previous_reading": int(previous) if previous is not None and previous.is_integer() else previous,
        "current_reading": int(current) if current is not None and current.is_integer() else current,
        "calculated_units": int(calculated_units) if calculated_units is not None and calculated_units.is_integer() else calculated_units,
        "warnings": warnings,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# RAG PIPELINE
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_embeddings():
    """Load and cache the HuggingFace embedding model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def _load_knowledge_documents():
    documents = []

    # Always resolve knowledge folder relative to rag_utils.py
    base_dir = os.path.dirname(os.path.abspath(__file__))
    knowledge_dir = os.path.join(base_dir, "knowledge")

    # Search PDFs recursively
    pdf_paths = sorted(
        glob.glob(os.path.join(knowledge_dir, "**", "*.pdf"), recursive=True)
    )

    print("======================================")
    print("PowerSense Knowledge Base")
    print("Base directory:", base_dir)
    print("Knowledge directory:", knowledge_dir)
    print("PDF files found:", len(pdf_paths))
    print("PDF paths:", pdf_paths)
    print("======================================")

    if not pdf_paths:
        return documents

    for pdf_path in pdf_paths:
        try:
            reader = PdfReader(pdf_path)

            for page_number, page in enumerate(reader.pages):
                text = page.extract_text() or ""

                if text.strip():
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": os.path.basename(pdf_path),
                                "file_path": pdf_path,
                                "page": page_number + 1,
                            },
                        )
                    )

        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")

    return documents

@st.cache_resource(show_spinner=False)
def build_or_load_vectorstore():
    """Build and cache the FAISS vector store."""
    documents = _load_knowledge_documents()
    if not documents:
        return None

    try:
        embeddings = get_embeddings()
        return FAISS.from_documents(documents, embeddings)
    except Exception as e:
        add_error(f"FAISS index build failed: {e}")
        return None


def retrieve_relevant_chunks(query: str, vectorstore, k: int = TOP_K) -> List[Dict]:
    """Retrieve relevant knowledge-base chunks."""
    if vectorstore is None or not query:
        return []

    try:
        results = vectorstore.similarity_search(query, k=k)
        return [
            {
                "text": r.page_content,
                "source": r.metadata.get("source", "Unknown source"),
            }
            for r in results
        ]
    except Exception as e:
        add_error(f"Knowledge retrieval failed: {e}")
        return []


# ---------------------------------------------------------------------------
# GROQ HELPERS
# ---------------------------------------------------------------------------


def get_groq_client(api_key: str) -> Optional[Groq]:
    if not api_key:
        return None
    try:
        return Groq(api_key=api_key)
    except Exception as e:
        add_error(f"Could not initialize Groq client: {e}")
        return None


def call_groq_chat(
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    model: str = GROQ_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 1800,
) -> Optional[str]:
    """Send a chat completion request to Groq and return the text response."""
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
        add_error(f"Groq API call failed: {e}")
        return None


def safe_json_parse(text: Optional[str]) -> Optional[dict]:
    """Extract and parse the first JSON object found in an LLM response."""
    if not text:
        return None

    cleaned = re.sub(r"```json|```", "", text).strip()

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    return None


# ---------------------------------------------------------------------------
# AGENT 1 — BILL EXTRACTION
# ---------------------------------------------------------------------------

BILL_FIELDS_SCHEMA = [
    "consumer_number", "meter_number", "provider", "billing_month",
    "issue_date", "due_date", "previous_reading", "current_reading",
    "units_consumed", "tariff_category", "previous_bill_amount",
    "current_bill_amount", "amount_payable", "electricity_charges",
    "taxes", "surcharges", "fca", "quarterly_adjustment",
    "fixed_charges", "arrears", "other_charges",
]

NUMERIC_EXTRACTION_FIELDS = {
    "units_consumed", "previous_reading", "current_reading",
    "previous_bill_amount", "current_bill_amount", "amount_payable",
    "electricity_charges", "taxes", "surcharges", "fca",
    "quarterly_adjustment", "fixed_charges", "arrears", "other_charges",
}


def extract_bill_fields_with_llm(api_key: str, bill_text: str) -> Dict:
    """Agent 1: extract structured fields and validate meter readings."""
    system_prompt = (
        "You are a careful data-extraction assistant for Pakistani electricity bills. "
        "Extract ONLY information explicitly present in the provided OCR text. "
        "Never guess or invent a number. If a field is missing or unclear, use null. "
        "Pay special attention to labels immediately next to numbers. Do not confuse "
        "QR-code numbers, consumer numbers, meter serial numbers, page numbers, or "
        "dates with monetary amounts. Preserve leading zeros in consumer/meter IDs. "
        "For numeric bill amounts, return numbers without currency symbols or commas. "
        "Respond with STRICT JSON ONLY, with no explanation and no markdown fences."
    )

    user_prompt = (
        "Extract the following fields from this electricity bill OCR text and return "
        f"a JSON object with exactly these keys: {json.dumps(BILL_FIELDS_SCHEMA)}.\n\n"
        "Important extraction rules:\n"
        "1. previous_reading must come from the field labelled previous reading.\n"
        "2. current_reading must come from the field labelled present/current reading.\n"
        "3. units_consumed must come from the bill's units/consumption field when clearly readable.\n"
        "4. If previous reading, current reading, and units are all visible, check whether "
        "current reading - previous reading = units. If one value conflicts, use repeated OCR "
        "evidence and the bill's labelled fields to resolve an obvious OCR digit error; record "
        "the correction in validation_notes rather than silently treating it as certain.\n"
        "5. Ignore numbers belonging to QR codes/barcodes unless they are explicitly labelled as a bill field.\n"
        "6. If OCR is contradictory and cannot be resolved from the bill's labelled values, use null.\n\n"
        "Numeric fields should be numbers or null. Text fields should be strings or null.\n\n"
        f"BILL OCR TEXT:\n{bill_text[:12000]}"
    )

    raw = call_groq_chat(
        api_key,
        system_prompt,
        user_prompt,
        temperature=0.0,
        max_tokens=2200,
    )

    parsed = safe_json_parse(raw)
    if not parsed:
        return {field: None for field in BILL_FIELDS_SCHEMA}

    # Keep the schema strict and normalize numeric fields.
    cleaned = {}
    for field in BILL_FIELDS_SCHEMA:
        value = parsed.get(field)
        if field in NUMERIC_EXTRACTION_FIELDS:
            number = _to_number(value)
            cleaned[field] = (
                int(number) if number is not None and number.is_integer() else number
            )
        else:
            cleaned[field] = value if value not in ("", "unknown", "Unknown") else None

    # Validate meter readings. If the bill provides both readings, calculated
    # consumption is more trustworthy than an OCR/LLM unit mismatch.
    validation = validate_bill_data(cleaned)
    if validation.get("calculated_units") is not None:
        cleaned["units_consumed"] = validation["calculated_units"]

    cleaned["validation_notes"] = validation.get("notes", [])
    return cleaned


# ---------------------------------------------------------------------------
# AGENTS 2 & 3 — TARIFF ANALYSIS + CHARGE VERIFICATION
# ---------------------------------------------------------------------------

LANGUAGE_INSTRUCTIONS = {
    "English": "Respond in clear, simple English.",
    "Urdu": "Respond in Urdu script (اردو).",
    "Roman Urdu": "Respond in Roman Urdu (Urdu written using English/Latin characters).",
}


def analyze_and_verify_bill(
    api_key: str,
    bill_data: Dict,
    provider: str,
    consumer_category: str,
    language: str,
    context_chunks: List[Dict],
) -> Optional[Dict]:
    """Analyze bill components using retrieved knowledge-base context."""
    context_text = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks
    ) or "NO OFFICIAL KNOWLEDGE DOCUMENTS WERE RETRIEVED FOR THIS QUERY."

    lang_instruction = LANGUAGE_INSTRUCTIONS.get(
        language,
        LANGUAGE_INSTRUCTIONS["English"],
    )

    system_prompt = (
        "You are a cautious electricity-bill analysis assistant for Pakistani consumers. "
        "Never call a charge illegal, fake, or fraudulent. Use only these status labels: "
        "'Explained', 'Applicable', 'Requires Verification', 'Potential Billing Issue'. "
        "Only retrieved official context can justify tariff rates or rules. If the context "
        "does not clearly support a specific rate or rule, classify it as 'Requires Verification'. "
        "Do not fabricate tariff rates, percentages, policies, or calculations that are not "
        "supported by the provided bill data/context. Do not invent missing bill values. "
        "" + lang_instruction + " "
        "Respond with STRICT JSON ONLY, with no explanation outside the JSON and no markdown fences."
    )

    user_prompt = f"""
Consumer category: {consumer_category}
Electricity provider/DISCO: {provider}

Extracted bill data (numbers are in PKR unless the field name indicates readings/units):
{json.dumps(bill_data, indent=2)}

Retrieved official knowledge context:
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
- Include only bill components actually detected in bill_data.
- Do not treat QR/barcode numbers as charges.
- If previous and current readings are available, use their difference as consumption.
- Do not accuse the provider of wrongdoing.
- Only flag an item when it is genuinely unclear or needs verification.
- If knowledge context does not support a rule/rate, say that verification is needed.
- sources_used must contain only actual retrieved source file names used for claims.
"""

    raw = call_groq_chat(
        api_key,
        system_prompt,
        user_prompt,
        temperature=0.2,
        max_tokens=2200,
    )
    return safe_json_parse(raw)


# ---------------------------------------------------------------------------
# AGENT 4 — COMPLAINT ASSISTANT
# ---------------------------------------------------------------------------


def generate_complaint_package(
    api_key: str,
    bill_data: Dict,
    analysis: Dict,
    provider: str,
    language: str,
) -> Optional[Dict]:
    """Prepare a cautious complaint-assistance package."""
    lang_instruction = LANGUAGE_INSTRUCTIONS.get(
        language,
        LANGUAGE_INSTRUCTIONS["English"],
    )

    system_prompt = (
        "You are a consumer-assistance agent helping a Pakistani electricity consumer "
        "decide whether to file a complaint about their bill. Never guarantee that a "
        "complaint will succeed. Never invent a URL and do not include URLs in your response. "
        + lang_instruction + " Respond with STRICT JSON ONLY, no markdown fences."
    )

    user_prompt = f"""
Provider/DISCO: {provider}
Bill data: {json.dumps(bill_data, indent=2)}
Prior bill analysis: {json.dumps(analysis, indent=2)}

Return JSON with exactly this structure:
{{
  "should_consider_complaint": true or false,
  "likely_complaint_category": string,
  "reason_summary": string,
  "evidence_to_keep": [string],
  "draft_complaint_text": string,
  "contact_provider_first_note": string
}}

The draft complaint must be short, polite, and factual. Reference only flagged
items from the analysis. Do not accuse the provider of wrongdoing; ask for
verification or clarification instead.
"""

    raw = call_groq_chat(
        api_key,
        system_prompt,
        user_prompt,
        temperature=0.3,
        max_tokens=1200,
    )
    return safe_json_parse(raw)
