"""Chat orchestration - the single function the Streamlit app calls per turn.

Keeps two histories deliberately separate:
- display_history: full conversation shown in the UI (unchanged shape, includes debug info)
- llm_plain_history: PLAIN text only (no retrieved-context blocks), rebuilt fresh each turn as
  system prompt + last N plain Q&A pairs + the CURRENT turn's retrieved context. Old retrieved
  chunks never leak into later turns - this is what prevents context bloat and stale-context
  hallucination across a long conversation.
"""

from backend import config, safety
from backend.prompts import SYSTEM_PROMPT, build_user_message
from backend.retrieval import retrieve


def chat_turn(user_query, llm_plain_history, embed_model, index, metadata, groq_client):
    """
    Returns: (answer_text, debug_info)
    debug_info is meant for the "View Retrieved Context / Sources" expander in the UI.
    """
    if safety.check_emergency(user_query):
        return safety.EMERGENCY_RESPONSE, {"path": "emergency_shortcircuit"}, llm_plain_history

    graded_chunks, candidates, grader_raw = retrieve(user_query, embed_model, index, metadata, groq_client)
    diag_flag = safety.check_diagnosis_seeking(user_query)

    debug = {
        "candidates_found": len(candidates),
        "all_candidates": [
            {"focus": c["focus"], "score": c["score"], "source": c.get("source"), "url": c.get("url")}
            for c in candidates
        ],
        "kept_chunks": [
            {"focus": c["focus"], "score": c["score"], "content": c["content"],
             "source": c.get("source"), "url": c.get("url")}
            for c in graded_chunks
        ],
        "grader_raw_response": grader_raw,
        "diagnosis_flag": diag_flag,
    }

    if not graded_chunks:
        debug["path"] = "no_relevant_context"
        new_history = llm_plain_history + [
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": safety.NO_CONTEXT_RESPONSE},
        ]
        return safety.NO_CONTEXT_RESPONSE, debug, new_history[-(config.MAX_HISTORY_TURNS * 2):]

    user_message = build_user_message(user_query, graded_chunks, diag_flag)
    recent = llm_plain_history[-(config.MAX_HISTORY_TURNS * 2):]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + recent + [
        {"role": "user", "content": user_message}
    ]

    try:
        response = groq_client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=messages,
            temperature=0.4,
            max_tokens=config.ANSWER_MAX_TOKENS,
        )
        draft = response.choices[0].message.content or ""
        if not draft.strip():
            draft = ("Sorry, I wasn't able to generate a response for that - please try rephrasing "
                     "your question.")
    except Exception as e:
        fallback = ("Sorry, I'm having trouble reaching the model right now "
                    f"(possibly a rate limit or connection issue: {type(e).__name__}). "
                    "Please try again in a moment.")
        debug["path"] = "llm_call_failed"
        return fallback, debug, llm_plain_history

    is_safe = safety.llm_safety_review(draft, groq_client)
    debug["path"] = "answered" if is_safe else "answered_but_selfcheck_swapped_out"
    answer = (draft if is_safe else safety.SAFETY_FALLBACK_RESPONSE) + safety.DISCLAIMER

    new_history = llm_plain_history + [
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": draft},   # store the draft, not the disclaimer-suffixed version
    ]

    return answer, debug, new_history[-(config.MAX_HISTORY_TURNS * 2):]
