
import time
import os
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"

FALLBACK_OPTIONS = [
    "Upload a Relevant PDF",
    "Ask Something from Your Document",
    "Search the Web",
]

GEMINI_API_BASE = (
    "https://generativelanguage.googleapis.com/v1beta"
)

def _get_gemini_api_key():
    """Read Gemini API key from .env."""

    return (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or ""
    ).strip()


def get_available_gemini_models(api_key):
    """Find text-generation Flash models only."""

    url = f"{GEMINI_API_BASE}/models"

    response = requests.get(
        url,
        params={"key": api_key, "pageSize": 100},
        timeout=30,
    )

    if not response.ok:
        raise RuntimeError(
            f"Model detection failed: "
            f"HTTP {response.status_code}: "
            f"{response.text[:400]}"
        )

    data = response.json()

    available_models = []

    # Never use audio-only or image-generation models.
    excluded_keywords = [
        "image",
        "tts",
        "audio",
        "live",
        "embedding",
        "robotics",
    ]

    for model in data.get("models", []):

        name = model.get("name", "").replace(
            "models/", ""
        )

        methods = model.get(
            "supportedGenerationMethods", []
        )

        if "generateContent" not in methods:
            continue

        if "flash" not in name.lower():
            continue

        if any(
            keyword in name.lower()
            for keyword in excluded_keywords
        ):
            continue

        available_models.append(name)

    if not available_models:
        raise RuntimeError(
            "No compatible text-generation Flash models found."
        )

    preferred_models = [
        "gemini-3.8-flash",
        "gemini-3.1-flash",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ]

    ordered = [
        model for model in preferred_models
        if model in available_models
    ]

    ordered += sorted(
        model for model in available_models
        if model not in ordered
    )

    print("Text-compatible models:", ordered)

    return ordered


def get_available_gemini_model(api_key):
    """Return preferred text-generation model."""

    return get_available_gemini_models(api_key)[0]

    """Detect Flash models available for this API key."""

    url = f"{GEMINI_API_BASE}/models"

    available_models = []
    page_token = None

    while True:
        params = {
            "key": api_key,
            "pageSize": 100,
        }

        if page_token:
            params["pageToken"] = page_token

        response = requests.get(
            url,
            params=params,
            timeout=30,
        )

        if not response.ok:
            raise RuntimeError(
                f"Model detection failed: "
                f"HTTP {response.status_code}: "
                f"{response.text[:400]}"
            )

        data = response.json()

        for model in data.get("models", []):

            methods = model.get(
                "supportedGenerationMethods", []
            )

            name = model.get("name", "").replace(
                "models/", ""
            )

            if (
                "generateContent" in methods
                and "flash" in name.lower()
            ):
                available_models.append(name)

        page_token = data.get("nextPageToken")

        if not page_token:
            break

    if not available_models:
        raise RuntimeError(
            "No compatible Gemini Flash models found "
            "for this API key."
        )

    # Prefer the model recommended by your API error.
    preferred = [
        "gemini-3.8-flash",
        "gemini-3.1-flash",
        "gemini-2.5-flash",
    ]

    ordered_models = [
        model for model in preferred
        if model in available_models
    ]

    ordered_models += sorted(
        model for model in available_models
        if model not in ordered_models
    )

    return ordered_models

def get_available_gemini_model(api_key):
    """Return the preferred available Flash model."""

    return get_available_gemini_models(api_key)[0]

def _fallback_result(starter_questions):
    """Return three alternative options."""

    return {
        "answer": (
            "The answer could not be verified from your "
            "uploaded document. What would you like to do?"
        ),
        "page_numbers": [],
        "is_fallback": True,
        "is_error": False,
        "options": FALLBACK_OPTIONS,
        "suggested_questions": starter_questions,
    }

def _is_missing_answer(answer):
    """Detect common missing-context responses."""

    text = (answer or "").lower()

    markers = [
        "not mentioned in the provided document",
        "not found in the provided context",
        "not available in the context",
        "not covered in your uploaded document",
        "i don't know",
        "i do not know",
        "cannot find the information",
        "no information about",
    ]

    return any(marker in text for marker in markers)

def _call_gemini(prompt, api_key, model_name=None):
    """
    Call Gemini with retry and alternative model support.
    Retries temporary 429, 500 and 503 errors.
    """

    models = get_available_gemini_models(api_key)

    # Try requested model first if it is available.
    if model_name and model_name in models:
        models.remove(model_name)
        models.insert(0, model_name)

    last_error = None

    for model in models:

        url = (
            f"{GEMINI_API_BASE}/models/"
            f"{model}:generateContent"
        )

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 1000,
            }
        }

        for attempt in range(3):

            try:
                print(
                    f"Trying Gemini model: {model} "
                    f"(Attempt {attempt + 1}/3)"
                )

                response = requests.post(
                    url,
                    params={"key": api_key},
                    json=payload,
                    headers={
                        "Content-Type": "application/json"
                    },
                    timeout=60,
                )

                if response.ok:

                    data = response.json()

                    candidates = data.get(
                        "candidates", []
                    )

                    if not candidates:
                        last_error = (
                            "Gemini returned no candidates."
                        )
                        break

                    parts = (
                        candidates[0]
                        .get("content", {})
                        .get("parts", [])
                    )

                    answer = " ".join(
                        part.get("text", "")
                        for part in parts
                        if part.get("text")
                    ).strip()

                    if answer:
                        print(
                            f"Gemini success using: {model}"
                        )
                        return answer

                    last_error = (
                        "Gemini returned an empty response."
                    )
                    break

                status = response.status_code

                last_error = (
                    f"HTTP {status}: {response.text[:400]}"
                )

                # Temporary server or rate-limit errors.
                if status in (429, 500, 502, 503, 504):

                    if attempt < 2:
                        wait_time = 2 ** (attempt + 1)

                        print(
                            f"Temporary error {status}. "
                            f"Retrying in {wait_time} seconds..."
                        )

                        time.sleep(wait_time)
                        continue

                    # Try the next available model.
                    break

                # Model unavailable: try another model.
                if status == 404:
                    print(
                        f"Model {model} unavailable. "
                        "Trying another model..."
                    )
                    break

                # Invalid API key or permission problem.
                raise RuntimeError(last_error)

            except requests.Timeout:
                last_error = "Gemini request timed out."

                if attempt < 2:
                    time.sleep(2 ** (attempt + 1))
                    continue

                break

            except requests.RequestException as exc:
                last_error = str(exc)

                if attempt < 2:
                    time.sleep(2 ** (attempt + 1))
                    continue

                break

    raise RuntimeError(
        "All available Gemini Flash models failed. "
        f"Last error: {last_error}"
    )

def generate_rag_answer(
    query,
    context_text,
    page_numbers=None,
    suggested_questions=None,
    model_name=DEFAULT_GEMINI_MODEL,
):
    """Generate strict document-grounded answers."""

    question = (query or "").strip()
    context = (context_text or "").strip()

    starter_questions = suggested_questions or []

    if not question:
        return {
            "answer": "Please enter a question.",
            "page_numbers": [],
            "is_fallback": False,
            "is_error": False,
            "options": [],
            "suggested_questions": starter_questions,
        }

    # No retrieved context.
    if not context:
        return _fallback_result(starter_questions)

    api_key = _get_gemini_api_key()

    if not api_key:
        return {
            "answer": (
                "Gemini API key is missing. "
                "Configure GEMINI_API_KEY in your .env file."
            ),
            "page_numbers": [],
            "is_fallback": False,
            "is_error": True,
            "options": [],
            "suggested_questions": starter_questions,
        }

    prompt = f"""
You are RAGMentor, a strict document-based AI assistant.

RULES:

1. Answer ONLY using the provided PDF context.
2. Never invent information.
3. Never use outside knowledge.
4. Verify that the context supports your answer.
5. If the answer is missing, return exactly:
   NOT_FOUND_IN_DOCUMENT
6. If only part of the answer is supported,
   provide only the supported information.
7. Give clear and understandable answers.

DOCUMENT CONTEXT:
<context>
{context}
</context>

USER QUESTION:
{question}

ANSWER:
"""

    try:
        print("Starting RAGMentor answer generation...")

        answer = _call_gemini(
            prompt=prompt,
            api_key=api_key,
            model_name=model_name,
        )

        normalized = answer.strip().upper()

        if (
            "NOT_FOUND_IN_DOCUMENT" in normalized
            or not answer.strip()
            or _is_missing_answer(answer)
        ):
            return _fallback_result(starter_questions)

        return {
            "answer": answer,
            "page_numbers": sorted(
                set(page_numbers or [])
            ),
            "is_fallback": False,
            "is_error": False,
            "options": [],
            "suggested_questions": starter_questions,
        }

    except Exception as exc:

        return {
            "answer": (
                f"Gemini API Error: {str(exc)}\n\n"
                "Please check API availability, "
                "model access, internet connection "
                "and API quota."
            ),
            "page_numbers": [],
            "is_fallback": False,
            "is_error": True,
            "options": [],
            "suggested_questions": starter_questions,
        }