"""Streamlit UI: risk estimate + cited explanations, and free-form document Q&A.

Run:  python -m streamlit run src/hcai/app/streamlit_app.py
"""

from __future__ import annotations

import html

import streamlit as st

from hcai import DISCLAIMER
from hcai.app.service import Services
from hcai.config import load_config
from hcai.errors import HcaiError, PatientInputError
from hcai.ml.schema import CLINICAL_TEST_FIELDS, LABELS, PatientInput
from hcai.rag.pipeline import Answer

NEEDS_TEST = "🩺 needs a clinical test"
YES_NO = {"No": 0, "Yes": 1}
CP = {"Typical angina": 1, "Atypical angina": 2, "Non-anginal pain": 3, "Asymptomatic": 4}
ECG = {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}
SLOPE = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}
VESSELS = {"Unknown / not tested": None, "0": 0, "1": 1, "2": 2, "3": 3}
THAL = {"Unknown / not tested": None, "Normal": 3, "Fixed defect": 6, "Reversible defect": 7}


CSS = """
<style>
.block-container {padding-top: 1.2rem; max-width: 1180px;}
.hero {background: linear-gradient(135deg, #1f6fb2 0%, #4a9fd8 60%, #8cc6ec 100%);
  border-radius: 24px; padding: 2.2rem 2.4rem; color: #fff; margin-bottom: 1rem;}
.hero h1 {color: #fff; font-size: 2.3rem; line-height: 1.15; margin: 0 0 .6rem 0; font-weight: 650;}
.hero p {color: #eaf4fb; font-size: 1.02rem; max-width: 46rem; margin: 0 0 1.2rem 0;}
.tiles {display: flex; gap: .8rem; flex-wrap: wrap;}
.tile {background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.35);
  border-radius: 16px; padding: .7rem 1rem; min-width: 9.5rem;}
.tile b {display: block; font-size: 1.25rem; color: #fff;}
.tile span {font-size: .78rem; color: #dcecf8;}
.steps {display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: .8rem;
  margin: .2rem 0 1rem 0;}
.step {background: #fff; border-radius: 16px; padding: .9rem 1rem; border: 1px solid #d5e4f0;}
.step i {font-style: normal; color: #1f6fb2; font-weight: 700; font-size: .85rem;}
.step b {display: block; margin: .15rem 0; color: #12263a;}
.step span {font-size: .85rem; color: #4a6076;}
.card {background: #fff; border-radius: 20px; padding: 1.2rem 1.4rem; border: 1px solid #d5e4f0;
  margin-bottom: .9rem;}
.risk {display: flex; align-items: center; gap: 1.4rem; flex-wrap: wrap;}
.risk .pct {font-size: 3.2rem; font-weight: 700; line-height: 1;}
.band {display: inline-block; padding: .2rem .8rem; border-radius: 999px; font-weight: 600;
  font-size: .85rem; color: #fff;}
.band.low {background: #2e9e6b;} .band.moderate {background: #d99a1c;} .band.high {background: #c2413b;}
.bar {height: 10px; border-radius: 999px; background: #e1ebf4; overflow: hidden; min-width: 14rem;}
.bar > div {height: 100%; background: linear-gradient(90deg, #4a9fd8, #1f6fb2);}
div[data-testid="stForm"] {background: #fff; border-radius: 20px; border: 1px solid #d5e4f0;
  padding: 1.2rem 1.4rem;}
button[kind="primary"] {border-radius: 999px !important; padding: .45rem 1.6rem !important;}
.stTabs [data-baseweb="tab"] {border-radius: 999px; padding: .4rem 1.1rem;}
.disclaimer {font-size: .8rem; color: #4a6076;}
</style>
"""

BAND_NOTE = {
    "low": "Below the demo low cut-point",
    "moderate": "Between the demo cut-points",
    "high": "Above the demo high cut-point",
}


@st.cache_resource(show_spinner="Loading model and index…")
def get_services() -> Services:
    return Services(load_config())


def hero(svc: Services) -> None:
    """Header: real figures from the loaded model/index only (nothing typed by hand)."""
    prov = svc.provenance()
    tiles = [
        ("Index", f"{prov.get('index_chunks', 'n/a')} chunks"),
        ("Retrieval", prov.get("retrieval", "n/a")),
        ("Answers", "LLM + citations" if prov.get("answer_mode") == "LLM" else "Extractive"),
        ("Model", prov.get("model_version", "n/a")[:28]),
    ]
    tile_html = "".join(
        f"<div class='tile'><b>{html.escape(v)}</b><span>{k}</span></div>" for k, v in tiles
    )
    st.markdown(
        "<div class='hero'><h1>Heart-health risk estimate,<br>explained with sources.</h1>"
        "<p>A demonstration prototype: a model estimates risk from clinical fields, and every "
        "explanation is drawn from public WHO, NIDDK and NHLBI documents, with citations, or says "
        "it is not in them.</p>"
        f"<div class='tiles'>{tile_html}</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='steps'>"
        "<div class='step'><i>01</i><b>Enter patient fields</b><span>Validated; out-of-range values "
        "are rejected, not clamped.</span></div>"
        "<div class='step'><i>02</i><b>Get a model estimate</b><span>Probability and demo band, with "
        "the factors that raise it.</span></div>"
        "<div class='step'><i>03</i><b>Read cited explanations</b><span>Passages from the documents "
        "for each factor.</span></div>"
        "<div class='step'><i>04</i><b>Or ask a question</b><span>Answers carry citations, or "
        "state that the documents do not cover it.</span></div></div>",
        unsafe_allow_html=True,
    )


def show_answer(ans: Answer) -> None:
    if ans.mode == "emergency":
        st.error(ans.text, icon="🚨")
    elif ans.refused:
        st.warning(ans.text)
    else:
        st.markdown(ans.text)
        if ans.mode == "extractive":
            st.caption("No LLM key configured: showing passages verbatim from the documents.")
    for s in ans.sources:
        with st.expander(f"[{s.cite}] {s.title} — {s.section}"):
            st.write(s.snippet + ("…" if len(s.snippet) >= 300 else ""))
            st.caption(f"{s.url} · license: {s.license} · similarity {s.score:.2f}")
    if ans.warnings:
        st.caption("Notes: " + ", ".join(ans.warnings))


def risk_tab(svc: Services) -> None:
    st.info(f"{NEEDS_TEST}: " + ", ".join(LABELS[f] for f in CLINICAL_TEST_FIELDS))
    spec = PatientInput.model_fields
    with st.form("patient"):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age (years)", 18, 120, 55)
        sex = c1.selectbox("Sex", {"Female": 0, "Male": 1})
        cp = c1.selectbox("Chest pain type", list(CP))
        trestbps = c2.number_input("Resting blood pressure (mm Hg)", 70.0, 250.0, 130.0)
        chol = c2.number_input("Serum cholesterol (mg/dl)", 100.0, 700.0, 240.0)
        fbs = c2.selectbox("Fasting blood sugar > 120 mg/dl", list(YES_NO))
        restecg = c2.selectbox(f"Resting ECG ({NEEDS_TEST})", list(ECG))
        thalach = c3.number_input(f"Max heart rate achieved ({NEEDS_TEST})", 60.0, 250.0, 150.0)
        exang = c3.selectbox("Exercise-induced angina", list(YES_NO))
        oldpeak = c3.number_input(f"ST depression ({NEEDS_TEST})", 0.0, 10.0, 1.0, step=0.1)
        slope = c1.selectbox("Peak exercise ST slope", list(SLOPE))
        ca = c2.selectbox(f"Major vessels coloured ({NEEDS_TEST})", list(VESSELS))
        thal = c3.selectbox(f"Thalassemia test ({NEEDS_TEST})", list(THAL))
        submitted = st.form_submit_button("Estimate risk", type="primary")
    del spec
    if not submitted:
        return
    patient: dict[str, object] = {
        "age": int(age), "sex": {"Female": 0, "Male": 1}[sex], "cp": CP[cp], "trestbps": trestbps,
        "chol": chol, "fbs": YES_NO[fbs], "restecg": ECG[restecg], "thalach": thalach,
        "exang": YES_NO[exang], "oldpeak": oldpeak, "slope": SLOPE[slope], "ca": VESSELS[ca],
        "thal": THAL[thal],
    }  # fmt: skip
    try:
        with st.spinner("Assessing…"):
            res = svc.assess(patient)
    except PatientInputError as exc:
        st.error(str(exc))
        return
    except HcaiError as exc:
        st.error(str(exc))
        return
    r = res.risk
    st.metric("Model risk estimate", f"{r.probability:.0%}", help="Demo cut-points, not clinical.")
    band = r.band if r.band in BAND_NOTE else "moderate"
    st.markdown(
        "<div class='card risk'>"
        f"<div class='pct'>{r.probability:.0%}</div>"
        f"<div><span class='band {band}'>{band.title()} band</span>"
        f"<div class='disclaimer'>{BAND_NOTE[band]}. Model <code>{html.escape(str(r.model_version))}"
        "</code>. Not a diagnosis.</div></div>"
        f"<div class='bar'><div style='width:{min(max(r.probability, 0.0), 1.0) * 100:.0f}%'></div></div>"
        "</div>",
        unsafe_allow_html=True,
    )
    for n in res.notes:
        st.warning(n)
    if r.top_factors:
        st.subheader("Factors raising the estimate")
        st.table({"Factor": [f.label for f in r.top_factors],
                  "Your value": [f"{f.patient_value:g}" for f in r.top_factors],
                  "Added risk vs. typical": [f"+{f.delta:.0%}" for f in r.top_factors]})  # fmt: skip
    for e in res.explanations:
        st.subheader(f"About: {e.factor.label}")
        if e.answer:
            show_answer(e.answer)
        else:
            st.info(e.note)
    st.caption(res.disclaimer)


def qa_tab(svc: Services) -> None:
    q = st.text_input("Ask a question about the indexed health documents")
    st.caption(
        "Try: What are risk factors for high blood pressure? · When should I see a doctor about diabetes?"
    )
    if st.button("Ask") and q.strip():
        try:
            with st.spinner("Searching documents…"):
                show_answer(svc.ask(q))
        except HcaiError as exc:
            st.error(str(exc))
    st.caption(DISCLAIMER)


def main() -> None:
    st.set_page_config(page_title="Healthcare AI prototype", page_icon="🫀", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    svc = get_services()
    hero(svc)
    st.warning(DISCLAIMER, icon="⚠️")
    tab1, tab2 = st.tabs(["Risk assessment", "Ask the documents"])
    with tab1:
        risk_tab(svc)
    with tab2:
        qa_tab(svc)
    with st.sidebar:
        st.subheader("Provenance")
        for k, v in svc.provenance().items():
            st.caption(f"**{k}**: {v}")


main()
