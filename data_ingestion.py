import os
import re
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_pdf_pages(pdf_path):
    """Extract text page by page with 1-based page numbers."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    reader = PdfReader(pdf_path)
    pages = []
    for page_num, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        cleaned_text = re.sub(r"[ \t]+", " ", raw_text)
        cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text).strip()
        if cleaned_text:
            pages.append(
                {
                    "page_number": page_num,
                    "text": cleaned_text,
                }
            )
    return pages


def chunk_pages(pages, chunk_size=1000, chunk_overlap=150):
    """Split pages into overlapping chunks while preserving page_number metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for page in pages:
        page_num = page["page_number"]
        splits = splitter.split_text(page["text"])
        for chunk_idx, text_piece in enumerate(splits):
            clean_piece = text_piece.strip()
            if clean_piece:
                chunks.append(
                    {
                        "text": clean_piece,
                        "metadata": {
                            "page_number": page_num,
                            "chunk_index": chunk_idx,
                        },
                    }
                )
    return chunks


def generate_quick_summary(pages, max_chars=2200, max_sentences=5):
    """Create a concise summary from the early content of the PDF."""
    if not pages:
        return "No readable text found in the uploaded PDF."

    combined_parts = []
    for page in pages:
        combined_parts.append(f"[Page {page['page_number']}] {page['text']}")
    combined_text = "\n\n".join(combined_parts).strip()

    if not combined_text:
        return "No readable text found in the uploaded PDF."

    excerpt = combined_text[:max_chars].replace("\n", " ")
    raw_sentences = re.split(r"(?<=[.!?])\s+", excerpt)

    selected = []
    for sent in raw_sentences:
        s = sent.strip()
        if len(s) >= 20:
            selected.append(s)
        if len(selected) >= max_sentences:
            break

    if selected:
        return " ".join(selected)

    fallback = excerpt[:400].strip()
    return fallback + ("..." if len(excerpt) > 400 else "")


def generate_relevant_questions(pages, max_questions=4):
    """Generate dynamic starter questions based on key lines in the uploaded PDF."""
    candidates = []
    for page in pages[:6]:
        for raw_line in page["text"].splitlines():
            line = re.sub(r"\s+", " ", raw_line).strip("•-*#·●▪▸► ")
            if 25 <= len(line) <= 105 and not line.endswith("?"):
                if not re.fullmatch(r"[\d\W_]+", line):
                    candidates.append(line)

    questions = []
    seen = set()

    for line in candidates:
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        questions.append(f"Can you explain '{line}' from the document?")
        if len(questions) >= max_questions:
            break

    default_questions = [
        "What are the main topics covered in this PDF?",
        "Can you summarize the key points from this document?",
        "What important definitions or concepts are explained here?",
        "Are there any key steps, rules, or examples in this PDF?",
    ]

    for dq in default_questions:
        if len(questions) >= max_questions:
            break
        if dq.lower() not in seen:
            seen.add(dq.lower())
            questions.append(dq)

    return questions[:max_questions]


def ingest_pdf(pdf_path, chunk_size=1000, chunk_overlap=150):
    """Process a PDF and return total pages, chunks with page numbers, summary, and starter questions."""
    pages = extract_pdf_pages(pdf_path)
    chunks = chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    summary = generate_quick_summary(pages)
    relevant_questions = generate_relevant_questions(pages)

    return {
        "total_pages": len(pages),
        "chunks": chunks,
        "summary": summary,
        "relevant_questions": relevant_questions,
    }