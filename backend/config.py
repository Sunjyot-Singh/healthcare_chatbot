"""Central configuration - model names, paths, and tunable constants.

Change values here rather than hunting through the other modules.
"""

import os

# ---- Models (Groq) ----
GROQ_MODEL = "openai/gpt-oss-120b"          # main answering model
GROQ_SAFETY_MODEL = "openai/gpt-oss-20b"    # grading + self-check (cheaper/faster, still reasoning-capable)

# gpt-oss models spend tokens on internal reasoning before writing a final answer - if max_tokens
# runs out during that reasoning phase, content comes back empty. Keep these generous.
GRADER_MAX_TOKENS = 600
SAFETY_CHECK_MAX_TOKENS = 300
ANSWER_MAX_TOKENS = 500
REASONING_EFFORT_UTILITY_CALLS = "low"   # grading + self-check are classification tasks, not deep reasoning

# ---- Retrieval ----
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "faiss_index.index")
METADATA_PATH = os.environ.get("METADATA_PATH", "metadata.pkl")
TOP_K_CANDIDATES = 6
SOFT_FLOOR = 0.15   # cheap pre-filter to cut obvious noise before the LLM grader sees candidates

# ---- Memory ----
MAX_HISTORY_TURNS = 3   # prior exchanges kept in the LLM's context (plain text only, no old context blocks)
