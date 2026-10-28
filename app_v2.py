import hashlib
import json
import os
import re

import numpy as np
import pandas as pd
import plotly.express as px
import pymupdf
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-5-5"


# ==========================================================
# 1. MODELS (cached)
# ==========================================================
@st.cache_resource(show_spinner="Loading AI models (first run only)...")
def load_models():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    mdl.eval()
    emb = SentenceTransformer("all-MiniLM-L6-v2")
    return tok, mdl, emb


# ==========================================================
# 2. TEXT HELPERS
# ==========================================================
def extract_pdf_text(file_bytes: bytes):
    with pymupdf.open(stream=file_bytes, filetype="pdf") as doc:
        n_pages = len(doc)
        text = "\n".join(page.get_text() for page in doc)
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text, n_pages


CITE_RE = re.compile(
    r"\[(?:[A-Za-z\+]+\d{2,4}[a-z]?(?:,\s*)?)+\]|\[\d+(?:\s*[,–-]\s*\d+)*\]"
)
MATH_CHARS = set("=≤≥∑∫∞ϵεΣ∈∀∃±×÷∂√^_{}|<>")


def clean_display(s: str) -> str:
    """Remove citation markers and spacing glitches from paper text."""
    s = CITE_RE.sub("", s)
    s = re.sub(r"\b(or|and)\s+([.,;])", r"\2", s)
    s = re.sub(r"\s+([.,;:!?])", r"\1", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()


def is_prose(s: str) -> bool:
    """Reject sentences that are mostly math or have a section heading fused in."""
    if re.search(r"\b\d+\s+(Conclusions?|Introduction|Related Work|References|Preliminaries)\b", s):
        return False
    letters = sum(c.isalpha() for c in s)
    math = sum(c in MATH_CHARS for c in s)
    return letters / max(len(s), 1) > 0.65 and math <= 2


def trim_to_sentence(s: str) -> str:
    """Cut a generated summary back to its last complete sentence."""
    s = s.strip()
    if s and s[-1] in ".!?":
        return s
    cut = max(s.rfind(". "), s.rfind("? "), s.rfind("! "))
    return s[: cut + 1] if cut > 40 else s.rstrip(",;: ") + "."


def near_dup(a: str, b: str, thresh: float = 0.7) -> bool:
    wa, wb = set(re.findall(r"\w+", a.lower())), set(re.findall(r"\w+", b.lower()))
    return bool(wa and wb) and len(wa & wb) / len(wa | wb) >= thresh


def heading_regex(names):
    alt = "|".join(names)
    return re.compile(
        rf"(?im)^[ \t]*(?:(?:\d+(?:\.\d+)*|[IVX]+)\.?[ \t]*\n?[ \t]*)?"
        rf"(?:{alt})s?[ \t]*:?[ \t]*$"
    )


STOP_HEADINGS = heading_regex(
    ["references", "bibliography", "acknowledg\\w*", "appendix\\w*", "supplementary\\w*"]
)


def strip_references(text: str) -> str:
    matches = list(heading_regex(["references", "bibliography"]).finditer(text))
    return text[: matches[-1].start()] if matches else text


def get_section(text, names, stop_names=None, max_chars=4000) -> str:
    heads = list(heading_regex(names).finditer(text))
    stop_re = heading_regex(stop_names) if stop_names else None
    for h in reversed(heads):
        rest = text[h.end():]
        ends = [len(rest)]
        for rx in (STOP_HEADINGS, stop_re):
            if rx:
                m = rx.search(rest)
                if m:
                    ends.append(m.start())
        body = rest[: min(ends)].strip()
        if len(body) >= 150:
            return body[:max_chars]
    return ""


def get_abstract(text: str) -> str:
    m = re.search(
        r"(?is)\babstract\b[\s:.\-—]*(.*?)"
        r"(?=\n[ \t]*(?:(?:1|I)\.?[ \t]*\n?[ \t]*)?introduction\b|\bkeywords?\b|\bindex terms\b|\Z)",
        text[:9000],
    )
    return re.sub(r"\s+", " ", m.group(1)).strip()[:2500] if m else ""


def sentences(text: str):
    text = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [p.strip() for p in parts if len(p.strip()) > 25]


def chunk_by_tokens(text, tok, max_tokens=700):
    chunks, cur, n_cur = [], [], 0
    for s in sentences(text):
        n = len(tok.tokenize(s))
        if cur and n_cur + n > max_tokens:
            chunks.append(" ".join(cur))
            cur, n_cur = [], 0
        cur.append(s)
        n_cur += n
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def make_passages(text: str, size: int = 700, limit: int = 400):
    """Split the paper into ~700-character passages for retrieval (Ask this paper)."""
    out, cur = [], ""
    for s in sentences(text):
        if cur and len(cur) + len(s) > size:
            out.append(cur)
            cur = ""
        cur += " " + s
    if cur:
        out.append(cur)
    return [p.strip() for p in out][:limit]


# ==========================================================
# 3. LOCAL NLP: summary, claims, future work, keywords
# ==========================================================
def summarize(text, tok, mdl, max_len=150, min_len=40) -> str:
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        ids = mdl.generate(
            **inputs, max_length=max_len, min_length=min_len,
            num_beams=4, no_repeat_ngram_size=3,
            length_penalty=2.0, early_stopping=True, do_sample=False,
        )
    return tok.decode(ids[0], skip_special_tokens=True)


def summarize_long(text, tok, mdl, detail="Medium", max_chunks=5) -> str:
    final_max, final_min = {"Short": (110, 50), "Medium": (260, 110), "Detailed": (380, 180)}[detail]
    chunks = chunk_by_tokens(text, tok)[:max_chunks]
    if not chunks:
        return ""
    parts = [summarize(c, tok, mdl, 150, 50) for c in chunks]
    return parts[0] if len(parts) == 1 else summarize(" ".join(parts), tok, mdl, final_max, final_min)


CLAIM_CUES = re.compile(
    r"\b(we (?:show|prove|demonstrate|establish|obtain|achieve|find|found|derive|give|propose|present|introduce)|"
    r"our (?:results?|analysis|experiments?|approach|method|bounds?)|outperform\w*|"
    r"state[- ]of[- ]the[- ]art|results? (?:show|indicate|demonstrate|suggest)|"
    r"tight bound|cannot be improved|lower bound|upper bound|optimal)\b", re.I)

FUTURE_CUES = re.compile(
    r"(future work|future research|future direction|open problem|open question|"
    r"remains? (?:open|unclear|unknown|to be)|we leave|left for future|leave .{0,40} open|"
    r"it would be interesting|would be interesting|promising (?:direction|avenue)|"
    r"limitations?|a natural (?:next step|question)|further (?:study|investigation|research)|"
    r"we conjecture|can be extended|could be extended|beyond the scope)", re.I)


LIMIT_CUES = re.compile(
    r"(limitations?|drawbacks?|shortcomings?|weakness\w*|caveats?|unfortunately|"
    r"not enough|not sufficient|insufficient|fails? to|does not (?:hold|apply|extend|work|generali[sz]e)|"
    r"only (?:holds|works|applies)|restricted to|under the assumption|we assume|assuming that|"
    r"strong assumption|cannot be (?:extended|used|applied)|is not (?:possible|known|tight)|"
    r"worst[- ]case|however,? (?:this|our|the|it))", re.I)


def pick(pool, regex, limit, exclude=()):
    out = []
    for s in pool:
        s = clean_display(s)
        if not is_prose(s) or not regex.search(s):
            continue
        if any(near_dup(s, o) for o in list(out) + list(exclude)):
            continue
        out.append(s)
        if len(out) >= limit:
            break
    return out


def get_keywords(text, emb, top_n=10):
    cv = CountVectorizer(stop_words="english", ngram_range=(1, 2), max_features=300,
                         token_pattern=r"(?u)\b[A-Za-z][A-Za-z\-]{3,}\b")
    try:
        X = cv.fit_transform([text[:30000]])
    except ValueError:
        return []
    terms = cv.get_feature_names_out()
    counts = np.asarray(X.sum(axis=0)).ravel()
    cands = [terms[i] for i in counts.argsort()[::-1][:80]]
    doc_emb = emb.encode([text[:4000]], show_progress_bar=False)
    cand_emb = emb.encode(cands, show_progress_bar=False)
    sims = cosine_similarity(cand_emb, doc_emb).ravel()
    selected = [int(sims.argmax())]
    while len(selected) < min(top_n, len(cands)):
        rest = [i for i in range(len(cands)) if i not in selected]
        div = cosine_similarity(cand_emb[rest], cand_emb[selected]).max(axis=1)
        selected.append(rest[int((0.6 * sims[rest] - 0.4 * div).argmax())])
    return [(cands[i], round(float(sims[i]), 3)) for i in selected]


# ==========================================================
# 4. OPTIONAL CLAUDE FEATURES
# ==========================================================
def call_claude(api_key, model, system, user, max_tokens=3000) -> str:
    import anthropic  # pip install anthropic
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


ANALYSIS_SYSTEM = """You are an expert research assistant. Use ONLY the paper text provided.
Return valid JSON only (no markdown fences) with exactly these keys:
{
 "one_line": "one-sentence plain-English takeaway",
 "problem": "what problem the paper addresses (2-3 sentences)",
 "method": "approach/technique used (2-3 sentences)",
 "contribution": "main contribution (2-3 sentences)",
 "detailed_summary": "a complete 6-8 sentence summary covering background, method, results and conclusion",
 "main_results": ["3-5 concrete results/findings, with numbers if given"],
 "limitations": ["2-5 limitations, restrictive assumptions or weaknesses of the paper"],
 "pitfalls_to_avoid": ["2-4 mistakes, dead ends or approaches a new researcher should avoid, grounded in the paper (e.g. conditions shown to be insufficient)"],
 "future_research": [{"idea": "a concrete follow-up research direction", "source": "authors" or "suggested"}],
 "audience": "who should read this paper and the background needed"
}
Give 3-5 future_research items. Mark source "authors" only if the paper itself states it; otherwise "suggested"."""


def llm_analysis(ctx, api_key, model):
    raw = call_claude(api_key, model, ANALYSIS_SYSTEM, ctx)
    start, end = raw.find("{"), raw.rfind("}")
    return json.loads(raw[start:end + 1])


QA_SYSTEM = """Answer the question using ONLY the numbered excerpts from the paper.
If the excerpts do not contain the answer, say so. Cite excerpt numbers like [1]. Be concise."""


def answer_question(question, r, emb, cfg):
    qv = emb.encode([question], show_progress_bar=False)
    sims = cosine_similarity(qv, r["passage_emb"]).ravel()
    top = sims.argsort()[::-1][:4]
    passages = [r["passages"][j] for j in top]
    if cfg["use_llm"] and cfg["api_key"]:
        ctx = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
        try:
            ans = call_claude(cfg["api_key"], cfg["model"], QA_SYSTEM,
                              f"Excerpts:\n{ctx}\n\nQuestion: {question}", 600)
            return ans, passages
        except Exception as e:
            return f"⚠️ Claude call failed ({e}). Showing the most relevant excerpts instead.", passages
    return None, passages


# ==========================================================
# 5. FULL ANALYSIS
# ==========================================================
def analyze(file_bytes, cfg, progress):
    tok, mdl, emb = load_models()

    progress(0.05, "Extracting text...")
    raw, n_pages = extract_pdf_text(file_bytes)
    if len(raw) < 500:
        return {"error": "Could not extract text. It may be a scanned (image-only) PDF."}
    text = strip_references(raw)

    progress(0.15, "Detecting sections...")
    abstract = get_abstract(text)
    intro = get_section(text, ["introduction"],
                        ["related work", "background", "preliminaries", "preliminary"], 4000)
    conclusion = get_section(text, ["conclusion", "concluding remarks", "discussion and conclusion",
                                    "conclusions and future work", "summary and conclusion"], None, 3500)
    results = get_section(text, ["results", "experiments", "experimental results", "evaluation"],
                          ["conclusion", "discussion", "related work"], 4000)
    future_sec = get_section(text, ["future work", "limitations", "open problems",
                                    "open questions", "discussion"], None, 3000)

    progress(0.30, "Summarizing the paper...")
    core = "\n".join(x for x in (abstract, intro, conclusion) if x)
    if len(core) < 800:
        start = max(text.lower().find(abstract[:40].lower()), 0) if abstract else 0
        core = text[start:start + 12000]
    summary = trim_to_sentence(clean_display(summarize_long(core, tok, mdl, cfg["detail"])))

    progress(0.42, "Summarizing each section...")
    section_summaries = {}
    for label, body in (("Introduction", intro), ("Results", results), ("Conclusion", conclusion)):
        body_ok = " ".join(x for x in sentences(body) if is_prose(clean_display(x)))
        if len(body_ok) >= 300:
            section_summaries[label] = trim_to_sentence(
                clean_display(summarize(body_ok[:4000], tok, mdl, 130, 40)))

    progress(0.55, "Extracting results and conclusions...")
    non_future = lambda t: " ".join(s for s in sentences(t) if not FUTURE_CUES.search(s) and is_prose(s))
    res_src = (non_future(conclusion) + " " + non_future(results) + " " + non_future(abstract)).strip()
    conclusion_summary = clean_display(summarize(res_src[:4000], tok, mdl, 150, 45)) if len(res_src) >= 250 else ""
    claims = pick(sentences(abstract) + sentences(conclusion) + sentences(results) + sentences(intro),
                  CLAIM_CUES, 5, exclude=[conclusion_summary] if conclusion_summary else ())

    progress(0.70, "Finding open problems and future directions...")
    future = pick(sentences(conclusion) + sentences(future_sec) + sentences(abstract) + sentences(text),
                  FUTURE_CUES, 6)

    limitations = pick(sentences(future_sec) + sentences(conclusion) + sentences(abstract)
                       + sentences(intro) + sentences(text), LIMIT_CUES, 6, exclude=future)

    progress(0.80, "Extracting keywords and indexing passages...")
    kw_source = (abstract + " " + intro) if len(abstract + intro) > 500 else text
    keywords = get_keywords(kw_source, emb)
    passages = make_passages(text)
    passage_emb = emb.encode(passages, batch_size=32, show_progress_bar=False)

    llm, llm_error = None, None
    if cfg["use_llm"] and cfg["api_key"]:
        progress(0.90, "Running deep analysis with Claude...")
        ctx = "\n\n".join(f"{k.upper()}:\n{v[:3500]}" for k, v in
                          [("abstract", abstract), ("introduction", intro),
                           ("results", results), ("conclusion", conclusion)] if v)
        if len(ctx) < 800:
            ctx = text[:12000]
        try:
            llm = llm_analysis(ctx, cfg["api_key"], cfg["model"])
        except Exception as e:
            llm_error = str(e)

    progress(1.0, "Done")
    words = len(text.split())
    return {
        "n_pages": n_pages, "words": words, "read_min": max(1, round(words / 200)),
        "abstract": abstract, "summary": summary, "conclusion_summary": conclusion_summary,
        "claims": claims, "future": future, "limitations": limitations,
        "section_summaries": section_summaries, "keywords": keywords,
        "passages": passages, "passage_emb": passage_emb,
        "found": {"Abstract": bool(abstract), "Introduction": bool(intro),
                  "Results": bool(results), "Conclusion": bool(conclusion)},
        "llm": llm, "llm_error": llm_error,
    }


def to_markdown(name, r) -> str:
    L, md = r["llm"], [f"# Analysis of {name}", ""]
    if L:
        md += [f"**{L.get('one_line', '')}**", "", "## Problem", L.get("problem", ""), "",
               "## Method", L.get("method", ""), "", "## Contribution", L.get("contribution", ""), ""]
    md += ["## Summary", r["summary"], ""]
    results = L.get("main_results") if L else None
    if results:
        md += ["## Main Results"] + [f"- {x}" for x in results] + [""]
    else:
        md += ["## Results & Conclusions", r["conclusion_summary"] or "n/a"] + \
              [f"- {c}" for c in r["claims"]] + [""]
    if L and L.get("limitations"):
        md += ["## Limitations"] + [f"- {x}" for x in L["limitations"]] + [""]
    md += ["## Drawbacks & Pitfalls"]
    if L and (L.get("limitations") or L.get("pitfalls_to_avoid")):
        md += [f"- {x}" for x in L.get("limitations", [])]
        md += [f"- AVOID: {x}" for x in L.get("pitfalls_to_avoid", [])]
    else:
        md += [f"- {x}" for x in r["limitations"]] or ["- none detected"]
    md += ["", "## Future Research"]
    if L and L.get("future_research"):
        md += [f"- {f['idea']} _({f['source']})_" for f in L["future_research"]]
    else:
        md += [f"- {f}" for f in r["future"]]
    md += ["", "## Keywords", ", ".join(k for k, _ in r["keywords"])]
    return "\n".join(md)


# ==========================================================
# 6. UI
# ==========================================================
st.set_page_config(page_title="AI Paper Analyzer", page_icon="📄", layout="wide")
st.markdown(
    """<style>
    div[data-testid="stMetric"] {background:#f6f8fb; border:1px solid #e3e8ef;
        padding:12px 16px; border-radius:12px;}
    .block-container {padding-top:2rem;}
    </style>""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Settings")
    detail = st.select_slider("Summary detail", ["Short", "Medium", "Detailed"], value="Medium")
    st.divider()
    st.subheader("🤖 Deep analysis (optional)")
    use_llm = st.toggle("Use Claude for structured analysis & Q&A", value=False)
    api_key = st.text_input("Anthropic API key", type="password",
                            value=os.environ.get("ANTHROPIC_API_KEY", ""),
                            disabled=not use_llm)
    claude_model = st.text_input("Model", value=DEFAULT_CLAUDE_MODEL, disabled=not use_llm)
    st.caption("Off = fully local analysis. On = text excerpts of your paper are sent to the Claude API.")

cfg = {"detail": detail, "use_llm": use_llm, "api_key": api_key, "model": claude_model}

st.title("📄 AI Academic Paper Analyzer")
st.markdown("Upload research papers for summaries, key results, future research directions, "
            "keyword insights, Q&A and comparison.")

uploaded = st.file_uploader("Upload research papers (PDF)", type=["pdf"], accept_multiple_files=True)

if uploaded:
    cache = st.session_state.setdefault("analysis_cache", {})
    results = []

    for idx, f in enumerate(uploaded):
        data = f.getvalue()
        key = hashlib.md5(data + repr((detail, use_llm and bool(api_key), claude_model)).encode()).hexdigest()

        if key not in cache:
            bar = st.progress(0.0, text=f"Analyzing {f.name}...")
            cache[key] = analyze(data, cfg, lambda p, m: bar.progress(p, text=f"{f.name}: {m}"))
            bar.empty()
        r = cache[key]

        if "error" in r:
            st.error(f"{f.name}: {r['error']}")
            continue
        results.append({"filename": f.name, **r})
        L = r["llm"]

        st.header(f"📘 {f.name}")
        if cfg["use_llm"] and r["llm_error"]:
            st.warning(f"Claude analysis failed, showing local analysis instead: {r['llm_error']}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pages", r["n_pages"])
        c2.metric("Reading time", f"~{r['read_min']} min")
        c3.metric("Sections found", f"{sum(r['found'].values())}/4")
        c4.metric("Keywords", len(r["keywords"]))
        missing = [k for k, v in r["found"].items() if not v]
        if missing:
            st.caption(f"ℹ️ Not detected: {', '.join(missing)}. Results may rely on the abstract.")

        t_over, t_res, t_lim, t_fut, t_kw, t_qa = st.tabs(
            ["🧭 Overview", "🎯 Results", "⚠️ Drawbacks & Pitfalls", "🔭 Future Research",
             "🏷️ Keywords", "💬 Ask this paper"])

        with t_over:
            if L:
                st.info(f"**In one line:** {L.get('one_line', '')}")
                a, b, c = st.columns(3)
                a.markdown("**Problem**"); a.write(L.get("problem", ""))
                b.markdown("**Method**"); b.write(L.get("method", ""))
                c.markdown("**Contribution**"); c.write(L.get("contribution", ""))
                if L.get("audience"):
                    st.caption(f"👥 Who should read this: {L['audience']}")
            if L and L.get("detailed_summary"):
                st.subheader("Detailed summary")
                st.write(L["detailed_summary"])
            st.subheader("AI summary of the full paper" if L else "Summary")
            st.write(r["summary"])
            if r["section_summaries"]:
                st.subheader("Section by section")
                for label, text_ in r["section_summaries"].items():
                    st.markdown(f"**{label}:** {text_}")
            with st.expander("Original abstract", expanded=not L):
                st.write(r["abstract"] or "Abstract not detected.")
            if not L:
                st.caption("💡 For a fuller, more fluent summary, turn on Claude in the sidebar (» icon, top left).")

        with t_res:
            if L and L.get("main_results"):
                st.subheader("Main results")
                for x in L["main_results"]:
                    st.markdown(f"- {x}")
                if L.get("limitations"):
                    st.subheader("Limitations")
                    for x in L["limitations"]:
                        st.markdown(f"- {x}")
            else:
                st.subheader("Results & conclusions")
                st.write(r["conclusion_summary"] or "_No separate conclusion section found._")
                if r["claims"]:
                    st.markdown("**Key claims made by the authors:**")
                    for c in r["claims"]:
                        st.markdown(f"- {c}")

        with t_lim:
            st.caption("Weaknesses and dead ends to keep in mind so your own work doesn't repeat them.")
            if L and (L.get("limitations") or L.get("pitfalls_to_avoid")):
                if L.get("limitations"):
                    st.subheader("Limitations of this paper")
                    for x in L["limitations"]:
                        st.markdown(f"- ⚠️ {x}")
                if L.get("pitfalls_to_avoid"):
                    st.subheader("Mistakes to avoid")
                    for x in L["pitfalls_to_avoid"]:
                        st.markdown(f"- 🚫 {x}")
            elif r["limitations"]:
                for x in r["limitations"]:
                    st.markdown(f"- ⚠️ {x}")
                st.caption("Extracted from the paper's own wording. Turn on Claude for deeper analysis.")
            else:
                st.info("The paper doesn't state explicit limitations. Turn on Claude in the sidebar "
                        "for an AI assessment of weaknesses and pitfalls.")

        with t_fut:
            if L and L.get("future_research"):
                st.caption("🟢 stated by the authors · 💡 suggested by AI based on the paper")
                for item in L["future_research"]:
                    badge = "🟢" if item.get("source") == "authors" else "💡"
                    st.markdown(f"- {badge} {item['idea']}")
            elif r["future"]:
                st.caption("Open questions and limitations stated in the paper:")
                for s in r["future"]:
                    st.markdown(f"- {s}")
            else:
                st.info("No explicit future work found. Turn on Claude in the sidebar for suggested directions.")

        with t_kw:
            if r["keywords"]:
                kdf = pd.DataFrame(r["keywords"], columns=["keyword", "relevance"]).sort_values("relevance")
                fig = px.bar(kdf, x="relevance", y="keyword", orientation="h",
                             title="Keyword relevance to the paper")
                fig.update_layout(height=420, yaxis_title=None, margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig, key=f"kw_{idx}")

        with t_qa:
            st.caption("Ask anything about this paper. Answers come from the paper's own text.")
            q = st.text_input("Your question", key=f"q_{idx}",
                              placeholder="e.g. What is the main theorem? What dataset was used?")
            if q:
                _, _, emb = load_models()
                with st.spinner("Searching the paper..."):
                    ans, passages = answer_question(q, r, emb, cfg)
                if ans:
                    st.markdown(ans)
                else:
                    st.caption("Showing the most relevant passages (enable Claude for written answers).")
                with st.expander("Source passages", expanded=ans is None):
                    for i, p in enumerate(passages, 1):
                        st.markdown(f"**[{i}]** {clean_display(p)}")

        st.download_button("⬇️ Download analysis (Markdown)", to_markdown(f.name, r),
                           file_name=f"{f.name}_analysis.md", key=f"dl_{idx}")
        st.divider()

    # ---------------- comparison ----------------
    if len(results) > 1:
        st.header("🔍 Cross-Paper Comparison")
        _, _, emb = load_models()
        names = [r["filename"] for r in results]
        texts = [(r["abstract"] or r["summary"]) + " " + r["conclusion_summary"] for r in results]
        sim = cosine_similarity(emb.encode(texts, show_progress_bar=False))
        fig = px.imshow(sim, x=names, y=names, text_auto=".2f", zmin=0, zmax=1,
                        color_continuous_scale="Blues", title="Topic similarity (1.0 = identical)")
        st.plotly_chart(fig, key="sim_heatmap")

        kw_sets = [set(k for k, _ in r["keywords"]) for r in results]
        shared = set.intersection(*kw_sets)
        st.markdown("**Shared keywords:** " + (", ".join(sorted(shared)) if shared else "none"))

        st.markdown("**Side-by-side main results**")
        cols = st.columns(len(results))
        for col, r in zip(cols, results):
            with col:
                st.markdown(f"**{r['filename']}**")
                if r["llm"] and r["llm"].get("main_results"):
                    for x in r["llm"]["main_results"][:4]:
                        st.markdown(f"- {x}")
                else:
                    st.write(r["conclusion_summary"] or r["summary"])
