import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

FALLBACK_OPTIONS = [
    "Upload a relevant PDF that covers this topic.",
    "Ask something from your document related to the uploaded PDF.",
    "Try one of the auto-generated relevant questions suggested for your document.",
]


def _get_gemini_client():
    """Initialize Google GenAI client using GEMINI_API_KEY or GOOGLE_API_KEY from .env."""
    api_key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or ""
    ).strip()

    if not api_key:
        return None

    return genai.Client(api_key=api_key)


def _is_answer_missing_from_context(answer_text):
    """Detect if the model response indicates missing context."""
    cleaned = (answer_text or "").strip().lower()
    if not cleaned:
        return True
    missing_markers = [
        "not mentioned in the provided document",
        "not found in the provided context",
        "not available in the context",
        "not covered in your uploaded document",
        "i don't know",
        "i do not know",
        "cannot find",
        "can't find",
        "no information about",
    ]
    return any(marker in cleaned for marker in missing_markers)


def generate_rag_answer(
    query,
    context_text,
    page_numbers=None,
    suggested_questions=None,
    model_name=DEFAULT_GEMINI_MODEL,
):
    """
    Generate an answer from retrieved PDF context using Gemini.
    Returns a dict with answer, page_numbers, is_fallback, options, and suggested_questions.
    """
    pages = sorted(set(page_numbers or []))
    starter_questions = suggested_questions or []

    question = (query or "").strip()
    if not question:
        return {
            "answer": "Please enter or speak a question so RAGMentor can help.",
            "page_numbers": [],
            "is_fallback": True,
            "options": FALLBACK_OPTIONS,
            "suggested_questions": starter_questions,
        }

    if not (context_text or "").strip():
        return {
            "answer": (
                "No relevant information found in your uploaded document. "
                "What would you like to do?"
            ),
            "page_numbers": [],
            "is_fallback": True,
            "options": FALLBACK_OPTIONS,
            "suggested_questions": starter_questions,
        }

    prompt = (
        "You are RAGMentor, a precise document study assistant.\n"
        "Answer the user's question strictly using the provided document context.\n"
        "If the answer cannot be found or inferred reliably from the provided context, "
        "reply strictly with: NOT_FOUND_IN_DOCUMENT\n\n"
        f"Document Context:\n{context_text}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )

    client = _get_gemini_client()

    if client is not None:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=600,
                ),
            )
            answer_text = (response.text or "").strip()

            if (
                not answer_text
                or "NOT_FOUND_IN_DOCUMENT" in answer_text
                or _is_answer_missing_from_context(answer_text)
            ):
                return {
                    "answer": (
                        "No relevant information found in your uploaded document for this question. "
                        "What would you like to do?"
                    ),
                    "page_numbers": [],
                    "is_fallback": True,
                    "options": FALLBACK_OPTIONS,
                    "suggested_questions": starter_questions,
                }

            return {
                "answer": answer_text,
                "page_numbers": pages,
                "is_fallback": False,
                "options": [],
                "suggested_questions": starter_questions,
            }

        except Exception as exc:
            excerpt = context_text[:600].strip()
            return {
                "answer": (
                    f"Gemini API notice ({exc}). Closest matching section from your PDF:\n\n"
                    f"{excerpt}..."
                ),
                "page_numbers": pages,
                "is_fallback": False,
                "options": [],
                "suggested_questions": starter_questions,
            }

    excerpt = context_text[:600].strip()
    return {
        "answer": (
            "GEMINI_API_KEY wasn't found in `.env`. Showing the closest matching section from your PDF:\n\n"
            f"{excerpt}..."
        ),
        "page_numbers": pages,
        "is_fallback": False,
        "options": [],
        "suggested_questions": starter_questions,
    }