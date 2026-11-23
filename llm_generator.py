import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

load_dotenv()

DEFAULT_HF_REPO_ID = "HuggingFaceH4/zephyr-7b-beta"

FALLBACK_OPTIONS = [
    "Ask questions directly related to the content of your uploaded PDF.",
    "Upload a more relevant PDF document that covers this topic.",
    "Try one of the auto-generated relevant questions suggested for your document.",
]


def _get_llm(repo_id=DEFAULT_HF_REPO_ID, temperature=0.2, max_new_tokens=512):
    """Initialize Hugging Face Zephyr-7b-beta endpoint if token is available."""
    hf_token = (
        os.getenv("HUGGINGFACEHUB_API_TOKEN")
        or os.getenv("HF_TOKEN")
        or ""
    ).strip()

    if not hf_token:
        return None

    return HuggingFaceEndpoint(
        repo_id=repo_id,
        huggingfacehub_api_token=hf_token,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )


def _is_answer_missing_from_context(answer_text):
    """Detect if the model indicates the information is not found in the context."""
    cleaned = (answer_text or "").strip().lower()
    if not cleaned:
        return True
    missing_markers = [
        "not mentioned in the provided document",
        "not found in the provided context",
        "not available in the context",
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
    repo_id=DEFAULT_HF_REPO_ID,
):
    """
    Generate an answer from retrieved PDF context using Zephyr-7b-beta.
    Returns a structured dictionary with answer, page_numbers, is_fallback, and options.
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
                "I couldn't find information for your question in the current document context. "
                "Here are a few ways you can continue:"
            ),
            "page_numbers": [],
            "is_fallback": True,
            "options": FALLBACK_OPTIONS,
            "suggested_questions": starter_questions,
        }

    prompt_template = PromptTemplate.from_template(
        """You are RAGMentor, a helpful study and research assistant.
Answer the user's question strictly using the provided document context.
If the answer is not clearly present in the provided context, reply exactly with:
NOT_FOUND_IN_DOCUMENT

Context:
{context}

Question:
{question}

Answer:"""
    )

    llm = _get_llm(repo_id=repo_id)

    if llm is not None:
        try:
            formatted_prompt = prompt_template.format(
                context=context_text,
                question=question,
            )
            raw_response = llm.invoke(formatted_prompt)
            answer_text = (
                raw_response.content
                if hasattr(raw_response, "content")
                else str(raw_response)
            ).strip()

            if (
                not answer_text
                or "NOT_FOUND_IN_DOCUMENT" in answer_text
                or _is_answer_missing_from_context(answer_text)
            ):
                return {
                    "answer": (
                        "This topic does not appear to be covered in the uploaded PDF. "
                        "Instead of stopping here, you can try any of these options:"
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
                    f"LLM API notice ({exc}). Based on the closest matching section in your PDF:\n\n"
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
            "Hugging Face API token isn't configured in `.env` yet, so here is the most relevant "
            f"excerpt retrieved from your PDF:\n\n{excerpt}..."
        ),
        "page_numbers": pages,
        "is_fallback": False,
        "options": [],
        "suggested_questions": starter_questions,
    }