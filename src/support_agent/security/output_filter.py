"""
Output Filter — LLM response sanitization guardrails.
Blocks secrets, internal paths, env vars, system prompt fragments,
and fabricated citations from reaching the end user.
"""

import re
import hashlib
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pattern catalogue (compiled once at import time)
# ---------------------------------------------------------------------------

_BLOCKED_PATTERNS: List[Tuple[re.Pattern, str, str]] = [
    # ENV-VAR values / names
    (re.compile(r'sk-[A-Za-z0-9_-]{20,}', re.S), "openai_key", "***BLOCKED_KEY***"),
    (re.compile(r'OPENAI_API_KEY\s*[=:]\s*\S+', re.I), "env_openai", "***BLOCKED_ENV***"),
    (re.compile(r'JWT_SECRET\s*[=:]\s*\S+', re.I), "env_jwt", "***BLOCKED_ENV***"),
    (re.compile(r'ENCRYPTION_KEY\s*[=:]\s*\S+', re.I), "env_enc", "***BLOCKED_ENV***"),
    (re.compile(r'MASTER_PASSWORD\s*[=:]\s*\S+', re.I), "env_master", "***BLOCKED_ENV***"),
    (re.compile(r'REDIS_PASSWORD\s*[=:]\s*\S+', re.I), "env_redis", "***BLOCKED_ENV***"),
    (re.compile(r'POSTGRES_PASSWORD\s*[=:]\s*\S+', re.I), "env_pg", "***BLOCKED_ENV***"),
    (re.compile(
        r'(?:os\.environ|os\.getenv|process\.env)\s*[\[\(]\s*[\'"]?\w+[\'"]?\s*[\]\)]',
        re.I,
    ), "code_env_access", "***BLOCKED***"),

    # Absolute filesystem paths (Unix & Windows)
    (re.compile(r'(?:/app/|/home/\w+/|C:\\Users\\)\S{5,}', re.I), "abs_path", "[path_redacted]"),

    # Python tracebacks / stack frames
    (re.compile(r'Traceback \(most recent call last\).*?(?=\n\n|\Z)', re.S), "traceback", "[internal_error]"),
    (re.compile(r'File "[^"]+", line \d+', re.S), "stack_frame", "[internal_error]"),

    # Common system-prompt leak markers
    (re.compile(r'(?:system prompt|system message|instruções internas|hidden prompt)', re.I), "sysprompt_ref", None),

    # Base64 blobs ≥ 40 chars (likely encoded secrets)
    (re.compile(r'(?:[A-Za-z0-9+/]{40,}={0,2})'), "base64_blob", None),
]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def filter_llm_output(text: str) -> str:
    """
    Sanitise an LLM-generated response before it reaches the user.

    Returns the cleaned text.  If a hard-block pattern fires, the
    entire response is replaced with a safe fallback.
    """
    if not text:
        return text

    hard_block = False
    cleaned = text

    for pattern, tag, replacement in _BLOCKED_PATTERNS:
        match = pattern.search(cleaned)
        if match:
            logger.warning(
                "[OUTPUT_FILTER] blocked pattern=%s matched_len=%d",
                tag, len(match.group()),
            )
            if replacement is not None:
                cleaned = pattern.sub(replacement, cleaned)
            else:
                # Patterns without replacement trigger hard block
                hard_block = True

    if hard_block:
        logger.warning("[OUTPUT_FILTER] hard_block triggered — full response replaced")
        return (
            "Desculpe, não consigo fornecer essa informação. "
            "Posso ajudar com dúvidas sobre produtos e serviços do Jota. "
            "Como posso te ajudar?"
        )

    return cleaned


def check_citation_fabrication(response: str, valid_citation_ids: List[str]) -> str:
    """
    Detect citations in the response (e.g. [C1], [C42]) that do NOT
    appear in *valid_citation_ids*.  Replace fabricated ones with a
    warning note.
    """
    found = re.findall(r'\[C(\d+)\]', response)
    for cid_num in found:
        cid = f"[C{cid_num}]"
        if cid not in valid_citation_ids:
            logger.warning("[OUTPUT_FILTER] fabricated citation detected: %s", cid)
            response = response.replace(cid, "[fonte não verificada]")
    return response


def hash_session_id(session_id: str, length: int = 8) -> str:
    """Return a short SHA-256 hash for safe logging of session identifiers."""
    return hashlib.sha256(session_id.encode()).hexdigest()[:length]
