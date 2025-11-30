import textwrap
from io import BytesIO
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from zyntra_reliability_engine import ZyntraReliabilityEngine


# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="Zyntra Reliability Engine – Diagnostic",
    page_icon="⚙️",
    layout="wide",
)


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
REQUIRED_COLUMNS: List[str] = [
    "service",
    "date_created",
    "date_updated",
    "SLA_status",
    "root_cause",
]


def render_sidebar():
    st.sidebar.markdown("### ⚙️ Zyntra Reliability Engine")
    st.sidebar.write(
        "Upload your ticket / incident data to generate an audit-ready reliability snapshot "
        "across incidents, MTTR, SLAs, and root causes."
    )

    st.sidebar.markdown("#### Engagement context")
    client_name = st.sidebar.text_input("Client / Company Name")
    environment_name = st.sidebar.text_input(
        "Environment / System (e.g., Core API, Robotics Fleet)"
    )

    st.sidebar.markdown("#### Upload data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload ticket dataset (CSV or Excel)",
        type=["csv", "xlsx"],
        help="Include columns like service, dates, SLA_status, root_cause, description. Additional fields are OK.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Data handling")
    st.sidebar.caption(
        "Use scrubbed data with no PII or secrets. Uploaded data is processed in-session for diagnostic purposes "
        "during the audit."
    )

    return client_name, environment_name, uploaded_file


def read_uploaded_file(uploaded_file: BytesIO) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame()

    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        if name.endswith(".xlsx"):
            return pd.read_excel(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read file: {exc}")
        return pd.DataFrame()

    st.error("Unsupported file type. Please upload a CSV or Excel file.")
    return pd.DataFrame()


def validate_columns(df: pd.DataFrame) -> bool:
    if df.empty:
        return False

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        with st.expander("Missing recommended columns", expanded=True):
            st.warning(
                "The dataset is missing one or more recommended columns for a full diagnostic:\n\n"
                + ", ".join(missing)
            )
            st.caption(
                "The engine will still run where possible, but some MTTR, SLA, or root-cause views may be incomplete. "
                "During an engagement, Zyntra will work with you to map fields from your ticketing system."
            )
    return True


def metric_block(label: str, value, suffix: str = ""):
    if value is None:
        st.metric(label, "N/A")
        return
    if isinstance(value, float):
        if abs(value) >= 10:
            display = f"{value:0.1f}{suffix}"
        else:
            display = f"{value:0.2f}{suffix}"
    else:
        display = f"{value}{suffix}"
    st.metric(label, display)


def plot_pareto(df: pd.DataFrame, column: str, title: str, top_n: int = 10):
    if df.empty or column not in df.columns:
        st.info("Not enough data to build Pareto view.")
        return

    vc = df[column].value_counts(dropna=False)

    counts = vc.values.astype(float)
    labels = vc.index.astype(str).tolist()

    # Aggregate into Top N + Other
    if len(labels) > top_n:
        top_counts = counts[:top_n]
        top_labels = labels[:top_n]
        other_count = counts[top_n:].sum()
        top_counts = np.append(top_counts, other_count)
        top_labels = top_labels + ["Other (long tail)"]
    else:
        top_counts = counts
        top_labels = labels

    cumulative = np.cumsum(top_counts) / top_counts.sum() * 100.0

    # Shorten very long labels for readability
    def shorten(label: str, max_len: int = 18) -> str:
        return label if len(label) <= max_len else label[: max_len - 1] + "…"

    display_labels = [shorten(l) for l in top_labels]

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.bar(display_labels, top_counts, color="#1d4ed8", alpha=0.85)
    ax1.set_xlabel(column.replace("_", " ").title())
    ax1.set_ylabel("Ticket count", color="#1f2933")
    ax1.tick_params(axis="y", labelcolor="#1f2933")
    ax1.set_xticklabels(display_labels, rotation=25, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(display_labels, cumulative, color="#f97316", marker="o")
    ax2.set_ylabel("Cumulative % of tickets", color="#f97316")
    ax2.tick_params(axis="y", labelcolor="#f97316")
    ax2.set_ylim(0, 110)

    plt.title(title)
    plt.tight_layout()
    st.pyplot(fig)

    st.caption(
        f"Showing top {min(top_n, len(vc))} root causes by ticket volume; "
        "remaining categories are grouped into 'Other (long tail)'."
    )


def plot_risk_bars(per_service: List[dict], top_n: int = 8):
    svc_df = pd.DataFrame(per_service)
    if svc_df.empty:
        st.info("Service-level metrics are not available yet; check that the 'service' column is present.")
        return

    required_cols = ["service", "ticket_count", "mttr_days", "sla_breach_rate_pct", "risk_score"]
    if not all(c in svc_df.columns for c in required_cols):
        st.info("Risk view could not be rendered because some metrics are missing.")
        return

    # Sort by risk_score descending
    svc_df = svc_df.sort_values("risk_score", ascending=False).reset_index(drop=True)

    top = svc_df.head(top_n).copy()
    rest = svc_df.iloc[top_n:].copy()

    # Color by SLA breach rate
    def color_for_breach(b: float) -> str:
        if b <= 5:
            return "#16a34a"  # green
        if b <= 15:
            return "#f97316"  # amber
        return "#dc2626"     # red

    top["color"] = top["sla_breach_rate_pct"].fillna(0).apply(color_for_breach)

    # Horizontal bar chart of top N
    fig, ax = plt.subplots(figsize=(8, 4))
    y_pos = np.arange(len(top))
    ax.barh(y_pos, top["risk_score"], color=top["color"])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top["service"])
    ax.invert_yaxis()  # highest risk at top
    ax.set_xlabel("Composite risk score (higher = more risk)")
    ax.set_title(f"Top {len(top)} services by risk score")

    # Annotate bars with ticket volume and SLA breach %
    for i, (_, row) in enumerate(top.iterrows()):
        label = f"{int(row['ticket_count'])} tickets, {row['sla_breach_rate_pct']:.1f}% breaches"
        ax.text(
            row["risk_score"] + 0.5,
            i,
            label,
            va="center",
            fontsize=8,
            color="#111827",
        )

    plt.tight_layout()
    st.pyplot(fig)

    st.caption(
        "Bars show the highest-risk services. Each label includes ticket volume and SLA breach rate. "
        "Red = high breach rate, amber = moderate, green = low."
    )

    # Table of remaining services (lower priority, but visible)
    if not rest.empty:
        st.markdown("##### Remaining services (tracked, lower priority)")
        st.dataframe(
            rest[["service", "ticket_count", "mttr_days", "sla_breach_rate_pct", "risk_score"]],
            use_container_width=True,
            hide_index=True,
        )


# ------------------------------------------------------------
# Layout – top hero and guidance
# ------------------------------------------------------------
client_name, environment_name, uploaded_file = render_sidebar()

st.markdown(
    """
    <style>
    .zentra-hero {
        padding: 0.5rem 0 0.75rem 0;
        border-bottom: 1px solid rgba(148,163,184,0.35);
        margin-bottom: 0.6rem;
    }
    .zentra-hero-title {
        font-size: 1.6rem;
        font-weight: 600;
        margin-bottom: 0.1rem;
    }
    .zentra-hero-subtitle {
        color: #9ca3af;
        font-size: 0.95rem;
        margin-bottom: 0.3rem;
    }
    .zentra-pill {
        display: inline-flex;
        align-items: center;
        font-size: 0.8rem;
        padding: 0.1rem 0.45rem;
        border-radius: 999px;
        border: 1px solid rgba(148,163,184,0.4);
        color: #9ca3af;
        margin-right: 0.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="zentra-hero">
      <div class="zentra-hero-title">Zyntra Reliability Engine</div>
      <div class="zentra-hero-subtitle">
        Diagnostic tool used during reliability audits to baseline incident health, MTTR, SLAs, and systemic root causes.
      </div>
      <div>
        <span class="zentra-pill">Reliability audit</span>
        <span class="zentra-pill">Incident analytics</span>
        <span class="zentra-pill">MTTR & SLA insights</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col_a, col_b = st.columns([2, 3])
with col_a:
    st.markdown("#### How this tool is used")
    st.write(
        textwrap.dedent(
            """
            - As part of a Zyntra reliability audit, this engine ingests historical ticket / incident data.
            - It identifies the small number of categories and root causes that drive the majority of impact.
            - It quantifies MTTR and SLA risk by service, and suggests where to focus engineering and process changes.
            """
        )
    )
with col_b:
    st.markdown("#### Expected input (typical schema)")
    st.caption(
        "Column names can be different in your systems – they will be mapped during the engagement – "
        "but this sample shows the shape of the data."
    )
    sample = pd.DataFrame(
        {
            "service": ["checkout-api", "robotics-fleet"],
            "date_created": ["2025-01-01T08:15:00Z", "2025-01-02T10:03:00Z"],
            "date_updated": ["2025-01-01T10:45:00Z", "2025-01-02T16:18:00Z"],
            "SLA_status": ["Met", "Breached"],
            "root_cause": ["Config drift", "WiFi coverage"],
            "severity": ["P1", "P2"],
            "description": [
                "Checkout errors for 3% of sessions",
                "Robots dropping from network intermittently",
            ],
        }
    )
    st.dataframe(sample, use_container_width=True, hide_index=True)

st.markdown("---")

# ------------------------------------------------------------
# Data ingest and engine run
# ------------------------------------------------------------
df = read_uploaded_file(uploaded_file)

if uploaded_file is None or df.empty:
    st.info(
        "Upload a CSV or Excel file in the sidebar to generate an audit snapshot. "
        "During a live engagement, Zyntra will work with your team to map fields from your ticketing system "
        "into this engine."
    )
else:
    st.success(f"Dataset loaded with {len(df):,} rows and {len(df.columns)} columns.")
    validate_columns(df)

    engine = ZyntraReliabilityEngine()
    results = engine.run(df)
    kpis = results.get("kpis", {})
    recs = results.get("recs", {})
    annotated_df = results.get("annotated_df", df)

    # Summary header
    st.markdown("### Engagement snapshot")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_block("Client", client_name or "N/A")
    with c2:
        metric_block("Environment", environment_name or "N/A")
    with c3:
        metric_block("Tickets analyzed", len(annotated_df))
    with c4:
        metric_block(
            "Services detected",
            len(annotated_df["service"].unique()) if "service" in annotated_df.columns else "N/A",
        )

    st.caption(
        "This snapshot summarizes current reliability health based on the uploaded ticket / incident history. "
        "All figures are directional and will be refined during the full audit."
    )

    # Tabs for different views
    tab_overview, tab_pareto, tab_services, tab_recs, tab_data = st.tabs(
        ["Overview", "Pareto & Root Causes", "Services & Risk", "Recommendations", "Data view"]
    )

    # --------------------------------------------------------
    # Overview tab
    # --------------------------------------------------------
    with tab_overview:
        st.subheader("Core reliability metrics")

        col1, col2, col3 = st.columns(3)
        with col1:
            metric_block("Global MTTR (days)", kpis.get("global_mttr_days"))
        with col2:
            metric_block("SLA breach rate (%)", kpis.get("sla_breach_rate_pct"), suffix="%")
        with col3:
            metric_block(
                "Services with elevated risk",
                len(kpis.get("per_service") or []),
            )

        st.markdown("#### Observations")
        bullets = []

        mttr = kpis.get("global_mttr_days")
        if mttr is not None:
            if mttr > 5:
                bullets.append(
                    f"- Mean time to resolve is {mttr:.1f} days, indicating long-running tickets that likely drive operational drag."
                )
            else:
                bullets.append(
                    f"- Global MTTR is {mttr:.1f} days, suggesting resolution cycles are relatively contained."
                )

        breach = kpis.get("sla_breach_rate_pct")
        if breach is not None:
            if breach > 10:
                bullets.append(
                    f"- SLA breach rate of {breach:.1f}% highlights recurring risk to customer commitments and contractual SLAs."
                )
            else:
                bullets.append(
                    f"- SLA breach rate of {breach:.1f}% indicates breaches occur but are not yet systemic."
                )

        if not bullets:
            bullets.append(
                "- Metrics will update once required fields (dates and SLA status) are available in the dataset."
            )

        st.write("\n".join(bullets))

        st.markdown("#### MTTR & SLA distribution (by service, where available)")
        if kpis.get("per_service"):
            svc_df = pd.DataFrame(kpis["per_service"])
            svc_df_display = svc_df[
                ["service", "ticket_count", "mttr_days", "sla_breach_rate_pct", "risk_score"]
            ]
            st.dataframe(svc_df_display, use_container_width=True, hide_index=True)
        else:
            st.info(
                "Service-level breakdowns will appear here when service, ticket_age_days, and SLA_status are present."
            )

    # --------------------------------------------------------
    # Pareto & Root Causes tab
    # --------------------------------------------------------
    with tab_pareto:
        st.subheader("Where incidents concentrate")

        st.markdown("#### Pareto of ticket volume by root cause")
        plot_pareto(annotated_df, "root_cause", "Pareto: Ticket volume by root cause")

        st.markdown("#### Top root causes (by frequency)")
        top_rc = (recs.get("top_root_causes") or [])[:10]
        if top_rc:
            top_rc_df = pd.DataFrame(top_rc)
            st.dataframe(top_rc_df, use_container_width=True, hide_index=True)
        else:
            st.info(
                "Root cause information is not present or not populated. "
                "To unlock this view, ensure a 'root_cause' or equivalent column is available."
            )

    # --------------------------------------------------------
    # Services & Risk tab
    # --------------------------------------------------------
    with tab_services:
        st.subheader("Service-level risk and reliability")

        per_service = kpis.get("per_service") or []
        if not per_service:
            st.info("Service-level metrics are not available yet; check that the 'service' column is present.")
        else:
            svc_df = pd.DataFrame(per_service)
            st.markdown("#### Ranked list of services by composite risk score")
            st.dataframe(
                svc_df[
                    ["service", "ticket_count", "mttr_days", "sla_breach_rate_pct", "risk_score"]
                ],
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("#### Where to focus first (ranked by risk)")
            plot_risk_bars(per_service)

    # --------------------------------------------------------
    # Recommendations tab
    # --------------------------------------------------------
    with tab_recs:
        st.subheader("Targeted reliability recommendations")

        est_sla_improvement = recs.get("estimated_sla_improvement_pct_points")
        est_mttr_reduction = recs.get("estimated_mttr_reduction_pct")

        c1, c2 = st.columns(2)
        with c1:
            metric_block(
                "Potential SLA improvement (percentage points)",
                est_sla_improvement,
                suffix=" pp" if est_sla_improvement is not None else "",
            )
        with c2:
            metric_block(
                "Potential MTTR reduction (%)",
                est_mttr_reduction,
                suffix="%" if est_mttr_reduction is not None else "",
            )

        st.markdown("#### High-priority focus areas")
        top_services = recs.get("top_risk_services") or []
        if top_services:
            svc_df_top = pd.DataFrame(top_services)
            st.dataframe(
                svc_df_top[
                    ["service", "ticket_count", "mttr_days", "sla_breach_rate_pct", "risk_score"]
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("High-priority services will be highlighted here once per-service metrics are available.")

        st.markdown("#### Suggested next steps (for discussion during audit)")
        st.write(
            textwrap.dedent(
                """
                1. Standardize incident fields and taxonomy across teams so that root causes and services are consistently labeled.
                2. For the top root-cause categories, implement targeted runbooks and guardrails (alerts, automation, playbooks).
                3. Review on-call design and escalation paths for the highest-risk services to reduce time-to-detection and time-to-mitigation.
                4. Introduce or refine SLOs and error budgets for customer-facing services that show recurring SLA breaches.
                5. Establish a regular reliability review cadence using this engine’s outputs to track MTTR and SLA improvements over time.
                """
            )
        )

    # --------------------------------------------------------
    # Data view tab
    # --------------------------------------------------------
    with tab_data:
        st.subheader("Annotated dataset (for export and offline analysis)")
        st.caption(
            "This view shows the dataset as interpreted by the engine, including derived fields such as ticket_age_days "
            "and cluster labels (where applicable)."
        )
        st.dataframe(annotated_df, use_container_width=True)

        buffer = BytesIO()
        annotated_df.to_csv(buffer, index=False)
        buffer.seek(0)
        st.download_button(
            label="Download annotated dataset as CSV",
            data=buffer,
            file_name="zyntra_reliability_engine_annotated.csv",
            mime="text/csv",
        )
