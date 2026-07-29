"""Safety guardrails: emergency short-circuit, diagnosis-seeking flag, and an LLM self-check
pass on the drafted answer before it's shown to the user.
"""

from backend import config

EMERGENCY_KEYWORDS = [
    "chest pain", "can't breathe", "cannot breathe", "difficulty breathing",
    "severe bleeding", "won't stop bleeding", "unconscious", "not breathing",
    "suicidal", "want to kill myself", "overdose", "stroke symptoms",
    "face drooping", "slurred speech", "severe allergic reaction", "anaphylaxis",
    "seizure", "choking",
]

DIAGNOSIS_PATTERNS = [
    "what disease", "do i have cancer", "diagnose me", "what's wrong with me",
    "am i dying", "is this serious", "what illness", "my diagnosis",
    "what condition do i have", "exactly what disease",
]

DISCLAIMER = (
    "\n\n_This is general health information, not a medical diagnosis. "
    "Please consult a licensed healthcare professional for advice specific to you._"
)

EMERGENCY_RESPONSE = (
    "This sounds like it could be a medical emergency. Please call your local emergency "
    "number (e.g. 911 / 112 / 108) or go to the nearest emergency room right away. "
    "I'm not able to provide emergency medical care - please seek immediate in-person help."
)

NO_CONTEXT_RESPONSE = (
    "I don't have specific, reliable information on that in my knowledge base, so I don't "
    "want to guess. Please check with a healthcare professional for accurate guidance on this."
    + DISCLAIMER
)

SAFETY_FALLBACK_RESPONSE = (
    "I want to be careful not to overstate this. In general, please treat what I say here as "
    "background information only, and bring specific symptoms or concerns to a healthcare "
    "professional who can properly evaluate you." + DISCLAIMER
)


def check_emergency(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in EMERGENCY_KEYWORDS)


def check_diagnosis_seeking(query: str) -> bool:
    q = query.lower()
    return any(p in q for p in DIAGNOSIS_PATTERNS)


def llm_safety_review(draft_answer: str, groq_client) -> bool:
    """Second-pass check: does the draft answer read like a diagnosis or a specific drug/dose
    recommendation? Returns True if SAFE to show as-is, False if it should be swapped out."""
    review_prompt = f'''Answer with only one word: SAFE or UNSAFE.

UNSAFE means the text below states or strongly implies a specific diagnosis for the reader
("you have X"), or recommends a specific prescription medication or exact dosage.
SAFE means it stays at the level of general information and appropriately suggests seeing a doctor.

TEXT:
{draft_answer}
'''
    try:
        resp = groq_client.chat.completions.create(
            model=config.GROQ_SAFETY_MODEL,
            messages=[{"role": "user", "content": review_prompt}],
            temperature=0,
            max_tokens=config.SAFETY_CHECK_MAX_TOKENS,
            reasoning_effort=config.REASONING_EFFORT_UTILITY_CALLS,
        )
        verdict = (resp.choices[0].message.content or "").strip().upper()
        return verdict.startswith("SAFE")
    except Exception:
        # Fail OPEN (treat as safe) so a transient API hiccup doesn't block every answer.
        # Flip to False (fail closed) if you'd rather be stricter at the cost of more false blocks.
        return True
