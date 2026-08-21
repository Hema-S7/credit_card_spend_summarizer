"""
Guardrails layer for the agentic RAG API.

Input Guard:
    - ToxicLanguage

Output Guard:
    - Partial PII masking
    - GuardrailsPII validation for high-risk entities
"""

import os
import re
import uuid

from dotenv import load_dotenv

load_dotenv(override=True)


# ==========================================================
# Guardrails imports
# ==========================================================

try:
    from guardrails.errors import ValidationError

except Exception:
    ValidationError = Exception


# ==========================================================
# Configuration
# ==========================================================

# Do not include:
# EMAIL_ADDRESS
# PHONE_NUMBER
# PERSON
# CREDIT_CARD
#
# because we handle them with partial masking.

PII_ENTITIES = [
    "US_SSN",
    "IBAN_CODE",
    "IP_ADDRESS",
]


TOXICITY_THRESHOLD = float(os.getenv("GUARDRAIL_TOXICITY_THRESHOLD", "0.5"))


# ==========================================================
# Regex Patterns
# ==========================================================


# Account/customer IDs
ACCOUNT_ID_RE = re.compile(r"\b\d{8,18}\b")


# NorthStar credit card IDs
# Example:
# CC-886001
# CC-882001

NORTHSTAR_CARD_ID_RE = re.compile(
    r"\b(CC-)(\d{2})(\d{4})\b",
    re.IGNORECASE,
)


# Email

EMAIL_RE = re.compile(
    r"\b([A-Za-z0-9._%+-])[A-Za-z0-9._%+-]*(@[A-Za-z0-9.-]+\.[A-Za-z]{2,})\b"
)


# Phone

PHONE_RE = re.compile(r"(\+?\d{1,3}[- ]?)?(\d{2})(\d{6})(\d{2})")


# ==========================================================
# Exception
# ==========================================================


class GuardrailViolation(Exception):

    def __init__(self, guard: str, message: str):

        self.guard = guard

        self.message = message

        super().__init__(f"[{guard}] {message}")


# ==========================================================
# Mask Functions
# ==========================================================


def mask_card_id(match):
    return match.group(1) + "**" + match.group(3)


def mask_account_id(match):

    value = match.group(0)

    return "*" * (len(value) - 4) + value[-4:]


def mask_email(match):

    first_char = match.group(1)

    domain = match.group(2)

    return first_char + "*****" + domain


def mask_phone(match):

    prefix = match.group(1) or ""

    first = match.group(2)

    last = match.group(4)

    return prefix + first + "******" + last


# ==========================================================
# Lazy Guard Construction
# ==========================================================


_guards = None


def _ensure_guardrails_configured():

    api_key = os.getenv("GUARDRAILS_API_KEY")

    if not api_key:

        return

    os.environ.setdefault("GUARDRAILS_TOKEN", api_key)

    rc_path = os.path.expanduser("~/.guardrailsrc")

    if os.path.exists(rc_path):

        return

    try:

        with open(rc_path, "w") as rc_file:

            rc_file.write(
                f"id={uuid.uuid4()}\n" f"token={api_key}\n" "enable_metrics=false\n"
            )

    except OSError:

        pass


def _build_guards():

    _ensure_guardrails_configured()

    try:

        from guardrails import Guard

        from guardrails.hub import GuardrailsPII, ToxicLanguage

    except ImportError as exc:

        raise RuntimeError("Guardrails validators are not installed.") from exc

    return {
        # Output guard
        "pii": Guard().use(GuardrailsPII(entities=PII_ENTITIES, on_fail="fix")),
        # Input guard
        "toxicity": Guard().use(
            ToxicLanguage(
                threshold=TOXICITY_THRESHOLD,
                validation_method="sentence",
                on_fail="exception",
            )
        ),
    }


def _get_guards():

    global _guards

    if _guards is None:

        _guards = _build_guards()

    return _guards


# ==========================================================
# Input Guard
# ==========================================================


def guard_input(query: str):

    guards = _get_guards()

    try:

        guards["toxicity"].validate(query)

    except ValidationError as exc:

        raise GuardrailViolation(
            "toxic_language",
            "Your message was flagged as abusive or toxic and cannot be processed.",
        ) from exc


# ==========================================================
# Output Guard
# ==========================================================


def guard_output(response: str) -> str:

    if not response:

        return response

    # ----------------------------------
    # Partial masking
    # ----------------------------------

    response = NORTHSTAR_CARD_ID_RE.sub(mask_card_id, response)

    response = ACCOUNT_ID_RE.sub(mask_account_id, response)

    response = EMAIL_RE.sub(mask_email, response)

    response = PHONE_RE.sub(mask_phone, response)

    # ----------------------------------
    # Guardrails validation
    # ----------------------------------

    guards = _get_guards()

    outcome = guards["pii"].validate(response)

    response = getattr(outcome, "validated_output", None) or response

    return response
