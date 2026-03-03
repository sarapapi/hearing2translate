from google import genai
from google.genai import types
import os
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "gemini-2.5-flash"

# Retry configuration
MAX_RETRIES = 5
INITIAL_BACKOFF = 2   # seconds
BACKOFF_MULTIPLIER = 2
MAX_BACKOFF = 60      # seconds

# HTTP status codes that are worth retrying
RETRYABLE_STATUS_CODES = {400, 500, 503}  # 400 can be transient on file upload


def load_model():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set.")
    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        seed=42,
    )
    return config, client


def generate(model_processor, model_input):
    config, client = model_processor
    last_exception = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            audio_file = client.files.upload(
                file=model_input["sample"],
                config=types.UploadFileConfig(mime_type="audio/wav"),
            )
            contents = [
                audio_file,
                model_input["prompt"],
            ]
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=config,
            )
            if response.text is None:
                return ""
            return response.text.strip()

        except Exception as e:
            last_exception = e
            status_code = _extract_status_code(e)
            error_name = type(e).__name__

            if status_code in RETRYABLE_STATUS_CODES:
                # Server-side / overload errors — safe to retry
                backoff = min(INITIAL_BACKOFF * (BACKOFF_MULTIPLIER ** (attempt - 1)), MAX_BACKOFF)
                logger.warning(
                    "Attempt %d/%d failed with %s (HTTP %s). Retrying in %.1fs...",
                    attempt, MAX_RETRIES, error_name, status_code, backoff,
                )
                time.sleep(backoff)

    logger.error("All %d retry attempts exhausted. Last error: %s", MAX_RETRIES, last_exception)
    raise last_exception


def _extract_status_code(exception):
    """Pull the HTTP status code out of whatever exception type the SDK throws."""
    # google-genai SDK wraps errors in google.api_core.exceptions or similar
    if hasattr(exception, "status_code"):
        return exception.status_code
    if hasattr(exception, "code"):
        return exception.code
    # Fall back to parsing the string representation
    msg = str(exception)
    for code in (400, 500, 503):
        if str(code) in msg:
            return code
    return None