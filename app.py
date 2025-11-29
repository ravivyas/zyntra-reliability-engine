import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from zyntra_reliability_engine import ZyntraReliabilityEngine

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Zyntra AI Reliability Engine",
    page_icon="⚙️",
    layout="wide",
)

# ----------------- CUSTOM CSS FOR APPLE-STYLE LOOK -----------------
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top left, #f5f5f7 0, #ffffff 40%, #f5f5f7 100%);
        color: #111827;
        font-family: -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
    }

    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        margin-bottom: 0.25rem;
    }

    .sub-header {
        font-size: 1.0rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }

    .metric-card {
        padding: 1rem 1.25rem;
        border-radius: 1rem;
        background: rgba(255, 255, 255, 0.92);
        box-shadow: 0 18px 45px rgba(15,23,42,0.07);
        border: 1px solid rgba(148, 163, 184, 0.2);
    }

    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.4rem;
    }

    .footer-note {
        font-size: 0.75rem;
        color: #6b7280;
        margin-top: 2rem;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------- SIDEBAR -----------------
st.sidebar.markdown("### ⚙️ Zyntra AI Reliability Engine")
st.sidebar.write("Upload your ticket/incident data to analyze reliability health.")

client_name = st.sidebar.text_input("Client / Company Name", "")
environment_name = st.sidebar.text_input("Environment / System", "")
uploaded_file = st.sidebar.file_uploader("Upload Excel or CSV", type=["xlsx", "csv"])

engine = ZyntraReliabilityEngine()

# ----------------- HEADER -----------------
st.markdown(
    '<div class="main-header">Zyntra AI — Reliability & Issue Intelligence</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">AI-powered analysis of tickets and incidents to surface systemic issues, reliability risk, and the top fixes that move the needle.</div>',
    unsafe_allow_html=True,
)

# ----------------- MAIN LOGIC -----------------
if not uploaded_file:
    st.info("Upload a file in the left sidebar to get started.")
else:
    # Load data
    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    st.success("File uploaded successfully.")

    result = engine.run(df)
    kpis = result["kpis"]
    recs = result["recs"]
    annotated_df = result["annotated_df"]

    # ---------- EXEC SNAPSHOT + CORE KPIs ----------
    with st.container():
        col_left, col_right = st.columns([2, 3])

        with col_left:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="section-title">Executive Snapshot</div>
                    <p><b>Client:</b> {client_name if client_name else "N/A"}<br>
                    <b>Environment:</b> {environment_name if environment_name else "N/A"}<br>
                    <b>Tickets Analyzed:</b> {len(annotated_df)}</p>
                    <p>This snapshot summarizes the reliability health of your systems based on ticket and incident data.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_right:
            m1, m2 = st.columns(2)
            with m1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(
                    "Global MTTR (days)",
                    round(kpis.get("global_mttr_days", 0) or 0, 2),
                )
                st.markdown("</div>", unsafe_allow_html=True)
            with m2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(
                    "SLA Breach Rate (%)",
                    round(kpis.get("sla_breach_rate_pct", 0) or 0, 2),
                )
                st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ---------- OVERVIEW SECTION (ROOT CAUSES + RISK SERVICES) ----------
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            '<div class="section-title">🔥 Top Root Causes (Pareto Slice)</div>',
            unsafe_allow_html=True,
        )
        if recs.get("top_root_causes"):
            for rc in recs["top_root_causes"]:
                st.write(
                    f"**{rc['root_cause']}** — {int(rc['count'])} issues "
                    f"({rc['pct']:.1f}%)"
                )
        else:
            st.write("_No root cause data available._")

    with col2:
        st.markdown(
            '<div class="section-title">⚠️ Top High-Risk Services</div>',
            unsafe_allow_html=True,
        )
        if "per_service" in kpis and kpis["per_service"]:
            for svc in kpis["per_service"][:5]:
                st.write(
                    f"**{svc['service']}**  \n"
                    f"- MTTR: {round(svc.get('mttr_days', 0) or 0, 2)} days  \n"
                    f"- SLA Breach: {round(svc.get('sla_breach_rate_pct', 0) or 0, 2)}%  \n"
                    f"- Tickets: {svc.get('ticket_count', 0)}  \n"
                    f"- Risk Score: {round(svc.get('risk_score', 0) or 0, 2)}"
                )
        else:
            st.write("_No service-level data available._")

    # ----------------- TABS: PARETO + RECOMMENDATIONS -----------------
    pareto_tab, recs_tab = st.tabs(["Pareto Analysis (80/20)", "Recommendations"])

    # ---------- PARETO TAB ----------
    with pareto_tab:
        st.markdown(
            '<div class="section-title">Root Cause Pareto Chart</div>',
            unsafe_allow_html=True,
        )

        if "root_cause" in annotated_df.columns:
            rc_counts = annotated_df["root_cause"].value_counts(dropna=False)
            pareto_df = rc_counts.reset_index()
            pareto_df.columns = ["root_cause", "count"]
            total = pareto_df["count"].sum() or 1
            pareto_df["pct"] = pareto_df["count"] / total * 100.0
            pareto_df["cum_pct"] = pareto_df["pct"].cumsum()

            st.write("Root causes sorted by frequency with cumulative percentage:")
            st.dataframe(pareto_df, use_container_width=True)

            # Pareto chart: bars + cumulative line
            fig, ax1 = plt.subplots(figsize=(8, 4))
            ax1.bar(pareto_df["root_cause"], pareto_df["count"])
            ax1.set_xlabel("Root Cause")
            ax1.set_ylabel("Ticket Count")
            ax1.tick_params(axis="x", rotation=45)

            ax2 = ax1.twinx()
            ax2.plot(pareto_df["root_cause"], pareto_df["cum_pct"], marker="o")
            ax2.set_ylabel("Cumulative % of Tickets")

            fig.tight_layout()
            st.pyplot(fig)

            # Highlight the 80/20 zone
            top_80 = pareto_df[pareto_df["cum_pct"] <= 80]
            st.markdown("**Root causes contributing to ~80% of total issues:**")
            if not top_80.empty:
                for _, row in top_80.iterrows():
                    st.write(
                        f"- {row['root_cause']}: {int(row['count'])} issues "
                        f"({row['pct']:.1f}% of total, cumulative {row['cum_pct']:.1f}%)"
                    )
            else:
                st.write("_Data set too small to compute a meaningful Pareto split._")
        else:
            st.write("_No `root_cause` column found in the dataset._")

    # ---------- RECOMMENDATIONS TAB ----------
    with recs_tab:
        st.markdown(
            '<div class="section-title">Actionable Recommendations</div>',
            unsafe_allow_html=True,
        )

        mttr = kpis.get("global_mttr_days")
        sla_breach = kpis.get("sla_breach_rate_pct")

        backlog_count = None
        if "status" in annotated_df.columns:
            status_series = annotated_df["status"].astype(str).str.lower()
            backlog_count = status_series.isin(
                ["open", "new", "in progress", "wip", "pending"]
            ).sum()

        st.markdown("### 🎯 Where to focus")

        suggestions = []

        # MTTR-related suggestions
        if mttr is not None:
            suggestions.append(
                f"- **Reduce MTTR**: Current MTTR is around **{mttr:.1f} days**. "
                "Define MTTR targets per severity, enforce time-to-first-response SLAs, "
                "and ensure each cluster has a clear technical owner."
            )

        # SLA-related suggestions
        if sla_breach is not None:
            suggestions.append(
                f"- **Improve SLA adherence**: SLA breach rate is about **{sla_breach:.1f}%**. "
                "Introduce SLA-aware queues, pre-SLA reminders, and escalation rules for high-impact tickets."
            )

        # Backlog suggestions
        if backlog_count is not None and backlog_count > 0:
            suggestions.append(
                f"- **Clear the backlog**: There are approximately **{backlog_count} open/pending tickets**. "
                "Run a focused backlog burn-down starting with the largest root-cause clusters and oldest tickets; "
                "close duplicates and low-value items aggressively."
            )

        # Root cause / Pareto suggestions
        if recs.get("top_root_causes"):
            top_rc_names = [rc["root_cause"] for rc in recs["top_root_causes"]]
            suggestions.append(
                "- **Fix the big recurring problems first**: The top recurring root causes are: "
                + ", ".join(top_rc_names)
                + ". For each, assign an owner, document a short RCA, and implement 2–3 concrete mitigations."
            )

        # Systemic efficiency suggestions
        suggestions.append(
            "- **Make support more efficient**: Standardize ticket templates, enforce categorization and root-cause "
            "fields, and use clusters to drive knowledge base articles and runbooks. This cuts handling time and "
            "reduces repeat incidents."
        )
        suggestions.append(
            "- **Operationalize learnings**: Convert recurring issues into monitoring, pre-deployment checks, "
            "and guardrails (e.g., automated rollbacks, feature flags) so the same problems don’t come back."
        )

        if suggestions:
            for s in suggestions:
                st.markdown(s)
        else:
            st.write("No specific recommendations could be derived from the current dataset.")

    # ---------- DOWNLOAD ANNOTATED FILE ----------
    st.markdown(
        '<div class="section-title">⬇️ Download Annotated File</div>',
        unsafe_allow_html=True,
    )
    towrite = BytesIO()
    annotated_df.to_excel(towrite, index=False)
    towrite.seek(0)

    st.download_button(
        label="Download Annotated Excel",
        data=towrite,
        file_name="annotated_issues.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.markdown(
        '<div class="footer-note">Generated by Zyntra AI Reliability Engine • Draft insights only • Not a substitute for engineering judgment.</div>',
        unsafe_allow_html=True,
    )
