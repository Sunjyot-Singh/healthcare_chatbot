"""Retrieval: FAISS recall stage + LLM relevance-grading precision stage.

Ported from the tested Colab notebook. FAISS casts a wide net (cheap), the grader decides what
actually reaches the answering model (the real safety gate) - see config.py for tuning constants.
"""

import json
import re

import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

from backend import config


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embed_model():
    return SentenceTransformer(config.EMBED_MODEL_NAME)


@st.cache_resource(show_spinner="Loading knowledge base index...")
def load_index_and_metadata():
    import pickle

    index = faiss.read_index(config.FAISS_INDEX_PATH)
    with open(config.METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


def retrieve_candidates(query, embed_model, index, metadata, top_k=config.TOP_K_CANDIDATES,
                         floor=config.SOFT_FLOOR):
    q_vec = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_vec)
    scores, idxs = index.search(q_vec, top_k)

    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1 or score < floor:
            continue
        entry = metadata[idx]
        results.append({**entry, "score": float(score)})
    return results


GRADER_PROMPT_TEMPLATE = """You are grading retrieved passages for relevance to a health question.

USER QUESTION: {query}

PASSAGES:
{listing}

For EACH passage, ask: would a careful doctor actually use THIS passage to answer THIS specific
question, or is it just a related-sounding but distinct topic? Passages about a different disease
or condition that merely shares vocabulary (e.g. a headache question vs. a passage about brain
tumors) are NOT relevant even though the words overlap - exclude them. If you are not clearly
confident a passage answers this specific question, exclude it - when in doubt, leave it out.

SPECIFICITY MATCHING: check whether the user's question names a specific real condition, or only
describes something vague/hypothetical (e.g. "a rare condition", "something obscure", "whatever
this is"). A passage about ONE specific named disease is only relevant if the user's question
names that same disease (or a clear synonym) - it is NOT made relevant just because both are
loosely about "rare diseases" as a category. If the question is vague and doesn't name a real
condition, only a general/overview passage about that category (not a passage naming one specific
disease) could ever qualify, and even then only if it doesn't invite a specific diagnosis.

IMPORTANT CARVE-OUT: this specificity rule is about passages naming a DIFFERENT, narrower diagnosis
than what the user described (e.g. excluding a "Brain Tumors" passage for a plain headache
question). It does NOT mean excluding a passage whose topic IS the exact symptom or word the user
used - if the user says "headache", KEEP a passage titled "Headache"; if they say "cough", KEEP a
passage titled "Cough". Those are the correct general-information match, even if part of the
user's question also asks for something you must decline (a diagnosis or a dosage) - answering the
general part while declining the rest is the desired behavior, not a reason to drop the passage.

Respond with ONLY a JSON object, no other text, in exactly this format:
{{"relevant_ids": ["<id>", "<id>"]}}
Use an empty array if none are relevant."""


def grade_chunks(query, candidates, groq_client, return_raw=False):
    """LLM relevance grader. Fails CLOSED (returns []) on any error or unparseable response -
    silently letting ungraded context through is worse than an occasional unnecessary
    'I don't have info on that.'"""
    if not candidates:
        return ([], "NO_CANDIDATES") if return_raw else []

    listing = "\n".join(
        f"[{c['chunk_id']}] Topic: {c['focus']} - {c['content'][:250]}"
        for c in candidates
    )
    prompt = GRADER_PROMPT_TEMPLATE.format(query=query, listing=listing)

    try:
        resp = groq_client.chat.completions.create(
            model=config.GROQ_SAFETY_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=config.GRADER_MAX_TOKENS,
            reasoning_effort=config.REASONING_EFFORT_UTILITY_CALLS,
        )
        raw = (resp.choices[0].message.content or "").strip()

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return ([], raw) if return_raw else []

        parsed = json.loads(match.group(0))
        keep_ids = set(parsed.get("relevant_ids", []))
        kept = [c for c in candidates if c["chunk_id"] in keep_ids]
        return (kept, raw) if return_raw else kept
    except Exception as e:
        raw_err = f"GRADER_ERROR_OR_UNPARSEABLE: {type(e).__name__}"
        return ([], raw_err) if return_raw else []


def retrieve(query, embed_model, index, metadata, groq_client):
    """Full retrieval pipeline: FAISS recall -> LLM grading.
    Returns (graded_chunks, all_candidates, grader_raw_response).
    """
    candidates = retrieve_candidates(query, embed_model, index, metadata)
    graded, raw = grade_chunks(query, candidates, groq_client, return_raw=True)
    return graded, candidates, raw
