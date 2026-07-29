# Healthcare Information Assistant

A RAG-based healthcare chatbot: Streamlit frontend, Groq-hosted LLMs, FAISS retrieval over the
MedQuAD dataset, with an LLM relevance-grading (C-RAG style) guardrail layer.

**Live demo:** https://healthcarechatbot-jdjbkarhta3vrzazopvnfk.streamlit.app/

## Optional enhancements implemented

- ✅ Prompt engineering (system prompt with strict no-diagnosis/no-dosage rules, diagnosis-seeking
  detection with tailored instructions)
- ✅ Context-aware conversations (multi-turn memory, see "Why memory is split" below)
- ✅ Chat history (multiple named conversations, New Chat / Delete Chat in the sidebar)
- ✅ Retrieval-Augmented Generation (RAG) over the MedQuAD dataset
- ✅ Vector database (FAISS)
- ✅ Medical knowledge base integration (MedQuAD, filtered to common-condition NIH sources)
- ✅ Conversation memory (separated display vs. LLM-context history, see below)
- ✅ Response guardrails (emergency short-circuit, C-RAG relevance grading, LLM self-check pass)
- ✅ Citation of sources (source name + URL shown per chunk in the "View Retrieved Context /
  Sources" expander under each answer)

## Data source & attribution

Knowledge base built from [MedQuAD](https://huggingface.co/datasets/lavita/MedQuAD) (Medical
Question Answering Dataset), a publicly available research dataset compiled from NIH websites
(MedlinePlus, cancer.gov, NIDDK, NHLBI, NINDS, NIH SeniorHealth, and others), used here for
educational/assignment purposes. No proprietary or copyrighted material is included.

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Get a free Groq API key at https://console.groq.com/keys
3. Run the data pipeline (`notebooks/healthcare_chatbot_rag_v2.ipynb`, Part A) in Google Colab to
   download MedQuAD, build the FAISS index, and produce `faiss_index.index` + `metadata.pkl`.
   Place both files in this project's root directory (next to `app.py`).
4. Run the app:
   ```
   streamlit run app.py
   ```
   Paste your Groq API key in the sidebar (or set it as `GROQ_API_KEY` in `.streamlit/secrets.toml`
   locally, or in the Streamlit Cloud app's Secrets settings when deployed - the hosted demo above
   already has this configured, so visitors don't need to enter a key).

## Architecture

```
User (Streamlit chat UI)
      |
      v
Emergency keyword check --------> (if matched) canned emergency response, no LLM call
      |
      v
FAISS retrieval (recall stage, top_k=6, loose score floor)
      |
      v
LLM relevance grader (openai/gpt-oss-20b) - precision stage, C-RAG pattern
      |
      v
(if nothing passes grading) --> "no relevant info" fallback
      |
      v
Answering model (openai/gpt-oss-120b) generates draft using only graded-relevant context
      |
      v
LLM self-check (openai/gpt-oss-20b) - flags diagnosis/dosage language in the draft
      |
      v
Final answer + disclaimer -> shown in UI, with retrieved chunks/sources in an expander
```

### Why two retrieval stages instead of one similarity threshold
A fixed cosine-similarity cutoff doesn't reliably separate "relevant" from "superficially similar"
in medical text - unrelated conditions often score similarly to genuinely relevant passages because
medical vocabulary overlaps heavily across topics. FAISS is used only for cheap recall; an LLM grades
each candidate against the actual question before it's allowed into the answering prompt.

### Why memory is split into two histories
`display_history` (shown in the UI) and `llm_plain_history` (fed back into the model) are kept
separate. The LLM history holds only plain question/answer text - never the bulky retrieved-context
blocks from previous turns - so old retrieved chunks can't bleed into later, unrelated answers.

## Known limitations

- The relevance grader is a small/fast model steered by prompt instructions; on unusual or highly
  ambiguous phrasing it can occasionally over- or under-include a borderline passage. This is
  mitigated by: the emergency short-circuit (bypasses retrieval/generation entirely for red-flag
  language), and the self-check pass on the drafted answer (catches diagnosis/dosage language
  even if imperfect context slipped through).
- Knowledge base is filtered to MedQuAD's more common-condition sources (MedlinePlus, cancer.gov,
  NIDDK, NHLBI, NINDS, SeniorHealth); very rare/genetic conditions are intentionally underrepresented
  since that content skews the bot toward alarming, low-probability answers for common symptoms.
- No persistent storage - conversations reset when the Streamlit session ends.

## Project structure

```
app.py                  - Streamlit UI, conversation state, chat rendering
backend/
  config.py             - model names, paths, tunable constants
  retrieval.py           - FAISS recall + LLM relevance grading
  safety.py               - emergency/diagnosis detection, LLM self-check, disclaimers
  prompts.py               - system prompt + prompt construction
  chat.py                   - orchestrates a single chat turn end-to-end
notebooks/
  healthcare_chatbot_rag_v2.ipynb  - data pipeline (Part A) + guardrail development/testing (Part B)
requirements.txt
```
