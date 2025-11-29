
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from zyntra_reliability_engine import ZyntraReliabilityEngine

st.set_page_config(page_title="Zyntra AI Reliability Engine", layout="wide")
st.sidebar.header("Upload Your File")
uploaded_file = st.sidebar.file_uploader("Upload Excel or CSV", type=["xlsx", "csv"])

engine = ZyntraReliabilityEngine()

st.title("⚙️ Zyntra AI — Reliability & Issue Intelligence Engine")

if uploaded_file:
    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    st.success("File Uploaded Successfully!")

    result = engine.run(df)
    kpis = result["kpis"]
    recs = result["recs"]
    annotated_df = result["annotated_df"]

    st.subheader("Global KPIs")
    col1, col2 = st.columns(2)
    col1.metric("Global MTTR (days)", round(kpis.get("global_mttr_days", 0), 2))
    col2.metric("SLA Breach Rate (%)", round(kpis.get("sla_breach_rate_pct", 0), 2))

    st.subheader("Top Root Causes (Pareto)")
    for rc in recs["top_root_causes"]:
        st.write(f"{rc['root_cause']} — {rc['count']} issues ({rc['pct']:.1f}%)")

    st.subheader("Download Annotated File")
    towrite = BytesIO()
    annotated_df.to_excel(towrite, index=False)
    towrite.seek(0)
    st.download_button("Download Excel", towrite, "annotated.xlsx")
