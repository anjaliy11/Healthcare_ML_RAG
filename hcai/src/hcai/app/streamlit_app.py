"""Streamlit UI: risk estimate + cited explanations, and free-form document Q&A.

Run:  python -m streamlit run src/hcai/app/streamlit_app.py
"""

from __future__ import annotations

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


@st.cache_resource(show_spinner="Loading model and index…")
def get_services() -> Services:
    return Services(load_config())


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
    st.write(f"**{r.band.title()} band** · model `{r.model_version}`")
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
    st.title("🫀 Healthcare AI prototype")
    st.warning(DISCLAIMER, icon="⚠️")
    svc = get_services()
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
