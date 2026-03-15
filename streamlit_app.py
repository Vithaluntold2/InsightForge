"""
InsightForge — Premium Streamlit Dashboard
Light theme · Lucide icons · Polished tables · Visible numbers
"""

import os
import pickle
import uuid
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from dotenv import load_dotenv

# ── Page config (MUST be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="InsightForge — Business Intelligence",
    page_icon="https://api.iconify.design/lucide/bar-chart-3.svg",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Lucide SVG Icons (inline, no CDN needed) ────────────────────────────────
# Each icon is a minimal Lucide-style SVG that works inside st.markdown.
LUCIDE_SVGS: dict[str, str] = {
    "bar-chart-3": '<path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/>',
    "layout-dashboard": '<rect width="7" height="9" x="3" y="3" rx="1"/><rect width="7" height="5" x="14" y="3" rx="1"/><rect width="7" height="5" x="14" y="12" rx="1"/><rect width="7" height="9" x="3" y="16" rx="1"/>',
    "search": '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
    "line-chart": '<path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/>',
    "bot-message-square": '<path d="M12 6V2H8"/><path d="m8 18-4 4V8a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2Z"/><path d="M2 12h2"/><path d="M9 11v2"/><path d="M15 11v2"/><path d="M20 12h2"/>',
    "message-square": '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
    "message-square-plus": '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/><path d="M12 7v6"/><path d="M9 10h6"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/>',
    "dollar-sign": '<line x1="12" x2="12" y1="2" y2="22"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>',
    "trending-up": '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>',
    "bar-chart-2": '<line x1="18" x2="18" y1="20" y2="10"/><line x1="12" x2="12" y1="20" y2="4"/><line x1="6" x2="6" y1="20" y2="14"/>',
    "sigma": '<path d="M18 6H5l5.5 6L5 18h13"/>',
    "arrow-up-down": '<path d="m21 16-4 4-4-4"/><path d="M17 20V4"/><path d="m3 8 4-4 4 4"/><path d="M7 4v16"/>',
    "star": '<polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>',
    "users": '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "calendar": '<path d="M8 2v4"/><path d="M16 2v4"/><rect width="18" height="18" x="3" y="4" rx="2"/><path d="M3 10h18"/>',
    "package": '<path d="m7.5 4.27 9 5.15"/><path d="M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z"/><path d="m3.3 7 8.7 5 8.7-5"/><path d="M12 22V12"/>',
    "globe": '<circle cx="12" cy="12" r="10"/><path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20"/><path d="M2 12h20"/>',
    "user": '<path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
    "cake": '<path d="M20 21v-8a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8"/><path d="M4 16s.5-1 2-1 2.5 2 4 2 2.5-2 4-2 2.5 2 4 2 2-1 2-1"/><path d="M2 21h20"/><path d="M7 8v3"/><path d="M12 8v3"/><path d="M17 8v3"/><path d="M7 4h.01"/><path d="M12 4h.01"/><path d="M17 4h.01"/>',
    "filter": '<polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/>',
    "hash": '<line x1="4" x2="20" y1="9" y2="9"/><line x1="4" x2="20" y1="15" y2="15"/><line x1="10" x2="8" y1="3" y2="21"/><line x1="16" x2="14" y1="3" y2="21"/>',
    "table": '<path d="M12 3v18"/><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M3 9h18"/><path d="M3 15h18"/>',
    "lightbulb": '<path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/>',
    "award": '<circle cx="12" cy="8" r="6"/><path d="M15.477 12.89 17 22l-5-3-5 3 1.523-9.11"/>',
    "copyright": '<circle cx="12" cy="12" r="10"/><path d="M14.83 14.83a4 4 0 1 1 0-5.66"/>',
    "git-branch": '<line x1="6" x2="6" y1="3" y2="15"/><circle cx="18" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><path d="M18 9a9 9 0 0 1-9 9"/>',
}


def lucide(name: str, size: int = 18, color: str = "#4f46e5") -> str:
    """Return an inline Lucide SVG icon."""
    paths = LUCIDE_SVGS.get(name, LUCIDE_SVGS["star"])
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round" '
        f'style="vertical-align:middle;margin-right:6px;display:inline-block;">'
        f'{paths}</svg>'
    )


def metric_card(icon_name: str, label: str, value: str,
                accent: str = "#4f46e5") -> str:
    """Return styled HTML for a premium metric card with Lucide icon."""
    return f"""
    <div style="background:#ffffff;border:1px solid #e5e7eb;border-radius:12px;
                padding:20px 18px;text-align:center;
                box-shadow:0 1px 3px rgba(0,0,0,0.06);
                transition:box-shadow 0.2s ease;">
        <div style="margin-bottom:8px;">
            {lucide(icon_name, 22, accent)}
            <span style="font-size:0.82rem;font-weight:600;color:#6b7280;
                         letter-spacing:0.03em;text-transform:uppercase;">
                {label}
            </span>
        </div>
        <div style="font-size:1.6rem;font-weight:700;color:#111827;
                    font-variant-numeric:tabular-nums;">
            {value}
        </div>
    </div>
    """


def render_premium_table(dataframe: pd.DataFrame, title: str = "",
                         icon: str = "") -> None:
    """Render a clean, light-themed HTML table with visible numbers."""
    header_icon = lucide(icon, 18, "#4f46e5") if icon else ""

    header_cells = "".join(
        f'<th style="padding:10px 14px;text-align:right;font-weight:600;'
        f'color:#374151;font-size:0.82rem;text-transform:uppercase;'
        f'letter-spacing:0.04em;border-bottom:2px solid #e5e7eb;'
        f'background:#f9fafb;">{col}</th>'
        for col in dataframe.columns
    )
    # First column left-aligned
    idx_header = (
        f'<th style="padding:10px 14px;text-align:left;font-weight:600;'
        f'color:#374151;font-size:0.82rem;text-transform:uppercase;'
        f'letter-spacing:0.04em;border-bottom:2px solid #e5e7eb;'
        f'background:#f9fafb;">{dataframe.index.name or ""}</th>'
    )

    rows_html = ""
    for i, (idx, row) in enumerate(dataframe.iterrows()):
        bg = "#ffffff" if i % 2 == 0 else "#f9fafb"
        cells = (
            f'<td style="padding:10px 14px;text-align:left;color:#111827;'
            f'font-weight:600;font-size:0.88rem;border-bottom:1px solid #f3f4f6;'
            f'background:{bg};">{idx}</td>'
        )
        for val in row:
            cells += (
                f'<td style="padding:10px 14px;text-align:right;color:#1f2937;'
                f'font-size:0.88rem;font-variant-numeric:tabular-nums;'
                f'border-bottom:1px solid #f3f4f6;background:{bg};">{val}</td>'
            )
        rows_html += f"<tr>{cells}</tr>"

    title_html = ""
    if title:
        title_html = (
            f'<div style="margin-bottom:10px;font-size:1.05rem;font-weight:700;'
            f'color:#111827;">{header_icon}{title}</div>'
        )

    st.markdown(
        f"""
        {title_html}
        <div style="overflow-x:auto;border:1px solid #e5e7eb;border-radius:10px;
                    box-shadow:0 1px 2px rgba(0,0,0,0.04);">
        <table style="width:100%;border-collapse:collapse;font-family:
                      'Inter','Segoe UI',system-ui,sans-serif;">
            <thead><tr>{idx_header}{header_cells}</tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_summary_df(dataframe: pd.DataFrame,
                       fmt_map: dict) -> pd.DataFrame:  # type: ignore[type-arg]
    """Apply formatting to a summary dataframe for HTML rendering."""
    formatted = dataframe.copy()
    for col, fmt in fmt_map.items():
        if col in formatted.columns:
            formatted[col] = formatted[col].apply(lambda v, f=fmt: f.format(v))
    return formatted


# ── Inject custom CSS via st.html (properly supports <style> tags) ───────────
st.html(
    """
    <style>
    /* Global light theme */
    .stApp {
        background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
        font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f1f5f9 0%, #e2e8f0 100%);
        border-right: 1px solid #e2e8f0;
    }

    /* Typography */
    .stApp h1 { color: #0f172a !important; font-weight: 800 !important;
                 letter-spacing: -0.02em !important; }
    .stApp h2 { color: #1e293b !important; font-weight: 700 !important; }
    .stApp h3 { color: #334155 !important; font-weight: 600 !important; }
    .stApp p, .stApp span, .stApp label, .stApp li,
    .stApp .stMarkdown { color: #1e293b !important; }

    /* Sidebar typography */
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #1e293b !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        font-weight: 500 !important;
        padding: 6px 0 !important;
    }

    /* Metric containers */
    [data-testid="stMetricValue"] {
        color: #111827 !important;
        font-size: 1.65rem !important;
        font-weight: 700 !important;
        font-variant-numeric: tabular-nums !important;
    }
    [data-testid="stMetricLabel"] {
        color: #6b7280 !important;
        font-weight: 600 !important;
        font-size: 0.78rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: #f1f5f9;
        border-radius: 10px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        font-weight: 600;
        color: #64748b !important;
        padding: 8px 18px;
    }
    .stTabs [aria-selected="true"] {
        background: #ffffff !important;
        color: #4f46e5 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }

    /* Chat styling */
    [data-testid="stChatMessage"] {
        border-radius: 12px !important;
        border: 1px solid #e5e7eb !important;
        background: #ffffff !important;
        margin-bottom: 8px !important;
    }
    .stChatInput input {
        border-radius: 12px !important;
        border: 1.5px solid #d1d5db !important;
    }

    /* Dividers */
    hr { border-color: #e2e8f0 !important; opacity: 0.6; }

    /* Multiselect overrides */
    .stMultiSelect div[data-baseweb="select"] {
        background: #ffffff !important;
        border: 1.5px solid #e2e8f0 !important;
        border-radius: 10px !important;
    }
    .stMultiSelect span {
        color: #1e293b !important;
    }
    /* Multiselect chips (tags) — soft slate instead of bold purple */
    .stMultiSelect [data-baseweb="tag"] {
        background: #f1f5f9 !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 6px !important;
        color: #334155 !important;
    }
    .stMultiSelect [data-baseweb="tag"] span {
        color: #334155 !important;
        font-weight: 500 !important;
        font-size: 0.82rem !important;
    }
    .stMultiSelect [data-baseweb="tag"] svg {
        fill: #94a3b8 !important;
    }
    .stMultiSelect [data-baseweb="tag"]:hover {
        background: #e2e8f0 !important;
        border-color: #94a3b8 !important;
    }

    /* Select/dropdown labels */
    .stMultiSelect label, .stSelectbox label {
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        color: #374151 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.03em !important;
    }

    /* Radio buttons — softer accent */
    .stRadio [data-baseweb="radio"] div {
        border-color: #4f46e5 !important;
    }

    /* Buttons */
    .stButton button {
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    </style>
    """
)

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datasets"
OUTPUT_DIR = BASE_DIR / "output"
PLOTS_DIR = OUTPUT_DIR / "plots"
PICKLE_FILE = OUTPUT_DIR / "sales_summary.pkl"
VECTORDB_DIR = BASE_DIR / "vectorstore"

# ── Load .env for Azure creds ───────────────────────────────────────────────
load_dotenv(BASE_DIR / ".env")


# ═══════════════════════════════════════════════════════════════════════════
#  DATA LOADING HELPERS (cached)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_sales_data():
    """Load CSV and add derived columns."""
    df = pd.read_csv(DATA_DIR / "sales_data.csv", parse_dates=["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Quarter"] = df["Date"].dt.quarter
    df["YearMonth"] = df["Date"].dt.to_period("M").astype(str)
    df["AgeGroup"] = pd.cut(
        df["Customer_Age"],
        bins=[17, 25, 35, 45, 55, 70],
        labels=["18-25", "26-35", "36-45", "46-55", "56-69"],
    )
    return df


@st.cache_data(show_spinner=False)
def load_summary():
    """Load the pre-computed summary pickle, or compute it fresh from the CSV."""
    if PICKLE_FILE.exists():
        with open(PICKLE_FILE, "rb") as f:
            return pickle.load(f)
    # No pickle found — compute summary on the fly from the dataset
    try:
        df = load_sales_data()
        return _compute_summary(df)
    except Exception:
        return None


def _compute_summary(df):
    """Build the same summary dict that insightforge_solution.py produces."""
    summary = {}
    summary["overall"] = {
        "total_records": len(df),
        "date_range": f"{df['Date'].min().date()} to {df['Date'].max().date()}",
        "total_sales": int(df["Sales"].sum()),
        "mean_sales": round(df["Sales"].mean(), 2),
        "median_sales": round(df["Sales"].median(), 2),
        "std_sales": round(df["Sales"].std(), 2),
        "min_sales": int(df["Sales"].min()),
        "max_sales": int(df["Sales"].max()),
        "mean_satisfaction": round(df["Customer_Satisfaction"].mean(), 2),
        "median_satisfaction": round(df["Customer_Satisfaction"].median(), 2),
        "std_satisfaction": round(df["Customer_Satisfaction"].std(), 2),
        "mean_customer_age": round(df["Customer_Age"].mean(), 2),
        "median_customer_age": round(df["Customer_Age"].median(), 2),
    }
    yearly = df.groupby("Year").agg(
        total_sales=("Sales", "sum"), avg_sales=("Sales", "mean"),
        transaction_count=("Sales", "count"),
        avg_satisfaction=("Customer_Satisfaction", "mean"),
    ).round(2)
    summary["yearly"] = yearly.to_dict("index")
    products = df.groupby("Product").agg(
        total_sales=("Sales", "sum"), avg_sales=("Sales", "mean"),
        median_sales=("Sales", "median"), std_sales=("Sales", "std"),
        count=("Sales", "count"),
    ).round(2)
    summary["products"] = products.to_dict("index")
    regions = df.groupby("Region").agg(
        total_sales=("Sales", "sum"), avg_sales=("Sales", "mean"),
        median_sales=("Sales", "median"),
        avg_satisfaction=("Customer_Satisfaction", "mean"),
    ).round(2)
    summary["regions"] = regions.to_dict("index")
    gender = df.groupby("Customer_Gender").agg(
        total_sales=("Sales", "sum"), avg_sales=("Sales", "mean"),
        avg_satisfaction=("Customer_Satisfaction", "mean"),
        avg_age=("Customer_Age", "mean"), count=("Sales", "count"),
    ).round(2)
    summary["gender"] = gender.to_dict("index")
    age = df.groupby("AgeGroup", observed=True).agg(
        total_sales=("Sales", "sum"), avg_sales=("Sales", "mean"),
        avg_satisfaction=("Customer_Satisfaction", "mean"),
        count=("Sales", "count"),
    ).round(2)
    summary["age_groups"] = age.to_dict("index")
    numeric_cols = ["Sales", "Customer_Age", "Customer_Satisfaction"]
    summary["correlations"] = df[numeric_cols].corr().round(4).to_dict()
    return summary


def _get_secret(key: str, default: str | None = None) -> str:
    """Retrieve a secret from os.environ, then st.secrets, then default."""
    val = os.environ.get(key)
    if val:
        return val
    try:
        val = st.secrets.get(key)  # type: ignore[union-attr]
        if val:
            return str(val)
    except Exception:
        pass
    if default is not None:
        return default
    raise KeyError(f"Secret '{key}' not found in environment or Streamlit secrets")


@st.cache_resource(show_spinner="Loading RAG pipeline…")
def load_rag_chain():
    """Load vector store and build RetrievalQA chain (cached as resource)."""
    try:
        from langchain.chains import RetrievalQA
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain_core.prompts import PromptTemplate
        from langchain_openai import AzureChatOpenAI

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local(
            str(VECTORDB_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        llm = AzureChatOpenAI(
            azure_deployment=_get_secret("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
            azure_endpoint=_get_secret("AZURE_OPENAI_ENDPOINT"),
            api_key=_get_secret("AZURE_OPENAI_API_KEY"),  # type: ignore[arg-type]
            api_version=_get_secret("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            temperature=0.3,
        )
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are InsightForge, an AI Business Intelligence Assistant.\n"
                "You have access to the InsightForge sales dataset (sales_data.csv) which contains "
                "sales transactions with columns: Date, Product, Region, Sales, Customer_Age, "
                "Customer_Gender, Customer_Satisfaction.\n"
                "IMPORTANT: When answering questions about the dataset (regions, products, sales figures, "
                "customers, etc.), ALWAYS use the actual dataset summary facts from the context below. "
                "Do NOT use information from reference PDFs or external sources for dataset-specific questions.\n"
                "Provide actionable insights with specific numbers from the data.\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Answer:"
            ),
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 8}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )
        return qa_chain
    except Exception as e:
        st.warning(f"RAG chain unavailable: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  PLOTTING HELPERS (light-theme, visible numbers)
# ═══════════════════════════════════════════════════════════════════════════

# Vibrant color palette constants
_BRAND = "#6366f1"        # indigo-500
_BRAND_DARK = "#4338ca"   # indigo-700
_BRAND_LIGHT = "#a5b4fc"  # indigo-300
_ACCENT1 = "#f43f5e"      # rose-500
_ACCENT2 = "#10b981"      # emerald-500
_ACCENT3 = "#f59e0b"      # amber-500
_ACCENT4 = "#3b82f6"      # blue-500
_RICH_PALETTE = ["#6366f1", "#f43f5e", "#10b981", "#f59e0b", "#3b82f6", "#8b5cf6", "#ec4899"]
_BAR_GRADIENT = ["#818cf8", "#6366f1", "#4f46e5", "#4338ca"]


def _light_fig(figsize=(10, 5)):
    """Create a figure with guaranteed light background and dark text."""
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("#fafbff")
    ax.tick_params(colors="#374151", labelsize=9.5)
    ax.xaxis.label.set_color("#1e293b")
    ax.yaxis.label.set_color("#1e293b")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    return fig, ax


def plot_monthly_trend(df):
    fig, ax = _light_fig((12, 5))
    monthly = df.groupby("YearMonth")["Sales"].sum().reset_index().sort_values("YearMonth")
    x = range(len(monthly))
    y = monthly["Sales"].values
    # Main line — thick, vibrant
    ax.plot(x, y, color=_BRAND, linewidth=2.8, zorder=3,
            label=f"Monthly Total (n={len(df):,} transactions)", solid_capstyle="round")
    # Gradient fill
    ax.fill_between(x, y, alpha=0.15, color=_BRAND)
    ax.fill_between(x, y, alpha=0.06, color=_BRAND_LIGHT)
    # Data point markers on peaks/valleys
    peak_idx = np.argmax(y)
    valley_idx = np.argmin(y)
    ax.scatter([peak_idx], [y[peak_idx]], color=_ACCENT2, s=80, zorder=5, edgecolors="white", linewidth=2)
    ax.scatter([valley_idx], [y[valley_idx]], color=_ACCENT1, s=80, zorder=5, edgecolors="white", linewidth=2)
    ax.annotate(f"Peak: ${y[peak_idx]:,.0f}", xy=(peak_idx, y[peak_idx]),
                xytext=(10, 15), textcoords="offset points", fontsize=8.5,
                fontweight="bold", color=_ACCENT2,
                arrowprops=dict(arrowstyle="->", color=_ACCENT2, lw=1.2))
    ax.annotate(f"Low: ${y[valley_idx]:,.0f}", xy=(valley_idx, y[valley_idx]),
                xytext=(10, -20), textcoords="offset points", fontsize=8.5,
                fontweight="bold", color=_ACCENT1,
                arrowprops=dict(arrowstyle="->", color=_ACCENT1, lw=1.2))
    # Average line
    mean_val = y.mean()
    ax.axhline(mean_val, color=_ACCENT1, linestyle="--", linewidth=1.3, alpha=0.6,
               label=f"Average: ${mean_val:,.0f}/month")
    ax.set_title("Monthly Sales Trend", fontsize=16, fontweight="bold", color="#0f172a", pad=14)
    ax.set_ylabel("Total Sales ($)", fontsize=11, fontweight="600")
    ax.set_xlabel("Month", fontsize=11, fontweight="600")
    step = max(1, len(monthly) // 8)
    ax.set_xticks(range(0, len(monthly), step))
    ax.set_xticklabels(monthly["YearMonth"].values[::step], rotation=45, fontsize=8.5)
    ax.grid(axis="y", alpha=0.15, color="#94a3b8", linestyle="-")
    ax.legend(loc="upper left", fontsize=9, frameon=True, facecolor="white",
              edgecolor="#e5e7eb", framealpha=0.95, shadow=True)
    plt.tight_layout()
    return fig


def plot_sales_by_product(df):
    fig, ax = _light_fig((10, 5))
    product_sales = df.groupby("Product")["Sales"].sum().sort_values()
    product_counts = df.groupby("Product")["Sales"].count().reindex(product_sales.index)
    colors = [_ACCENT3, _ACCENT2, _ACCENT4, _BRAND]
    bars = ax.barh(product_sales.index, product_sales.values,
                   color=colors[:len(product_sales)], height=0.55,
                   edgecolor="white", linewidth=1.5)
    # Add subtle shadow bars
    for bar in bars:
        ax.barh(bar.get_y() + bar.get_height() / 2, bar.get_width(),
                height=bar.get_height(), color="#000000", alpha=0.03,
                left=bar.get_width() * 0.002)
    ax.set_title("Total Sales by Product", fontsize=16, fontweight="bold", color="#0f172a", pad=14)
    ax.set_xlabel("Total Sales ($)", fontsize=11, fontweight="600")
    for i, (v, c) in enumerate(zip(product_sales.values, product_counts.values)):
        ax.text(v + 2000, i, f"${v:,.0f}  ({c:,} txns)", va="center", fontsize=10.5,
                color="#1e293b", fontweight="bold")
    total = product_sales.sum()
    ax.text(0.98, 0.02, f"Grand Total: ${total:,.0f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9.5, color="#6b7280",
            style="italic", fontweight="500")
    ax.grid(axis="x", alpha=0.12, color="#94a3b8")
    plt.tight_layout()
    return fig


def plot_sales_by_region(df):
    fig, ax = _light_fig((8, 6))
    region_sales = df.groupby("Region")["Sales"].sum()
    colors = [_BRAND, _ACCENT1, _ACCENT2, _ACCENT3]
    explode = [0.03] * len(region_sales)
    wedges, texts, autotexts = ax.pie(
        region_sales, labels=region_sales.index, autopct="%1.1f%%",
        colors=colors, startangle=90, pctdistance=0.78, explode=explode,
        wedgeprops={"linewidth": 2.5, "edgecolor": "white"},
        textprops={"color": "#1e293b", "fontweight": "700", "fontsize": 12},
        shadow=True,
    )
    for t in autotexts:
        t.set_fontsize(11)
        t.set_fontweight("bold")
        t.set_color("#ffffff")
    ax.set_title("Sales Distribution by Region", fontsize=16, fontweight="bold",
                 color="#0f172a", pad=16)
    legend_labels = [f"{r}: ${v:,.0f}" for r, v in zip(region_sales.index, region_sales.values)]
    ax.legend(wedges, legend_labels, title="Region Sales", loc="lower left",
             fontsize=9.5, title_fontsize=10, frameon=True, facecolor="white",
             edgecolor="#e5e7eb", framealpha=0.95, shadow=True,
             bbox_to_anchor=(-0.15, -0.05))
    plt.tight_layout()
    return fig


def plot_heatmap(df):
    fig, ax = _light_fig((10, 5.5))
    pivot = df.pivot_table(values="Sales", index="Product", columns="Region", aggfunc="sum")
    # Custom colormap: indigo gradient
    from matplotlib.colors import LinearSegmentedColormap
    indigo_cmap = LinearSegmentedColormap.from_list(
        "indigo", ["#eef2ff", "#c7d2fe", "#818cf8", "#4f46e5", "#3730a3"])
    sns.heatmap(pivot, annot=True, fmt=",.0f", cmap=indigo_cmap, ax=ax,
                linewidths=3, linecolor="white",  # type: ignore[arg-type]
                annot_kws={"fontsize": 12, "fontweight": "bold", "color": "#1e293b"},
                cbar_kws={"label": "Total Sales ($)", "shrink": 0.75})
    ax.set_title("Product × Region Sales Heatmap", fontsize=16, fontweight="bold",
                 color="#0f172a", pad=14)
    ax.set_xlabel("Region", fontsize=11, fontweight="600")
    ax.set_ylabel("Product", fontsize=11, fontweight="600")
    ax.tick_params(labelsize=10.5)
    plt.tight_layout()
    return fig


def plot_age_distribution(df):
    fig, ax = _light_fig((10, 5))
    sns.histplot(df["Customer_Age"], bins="auto", kde=True,  # type: ignore[arg-type]
                 color=_BRAND, ax=ax, edgecolor="white", linewidth=0.8,
                 alpha=0.7, label="Age frequency")
    # KDE line enhancement
    for line in ax.get_lines():
        line.set_color(_BRAND_DARK)
        line.set_linewidth(2.5)
    mean_age = df["Customer_Age"].mean()
    median_age = df["Customer_Age"].median()
    ax.axvline(mean_age, color=_ACCENT1, linestyle="--", linewidth=2,
               label=f"Mean: {mean_age:.1f} yrs", zorder=4)
    ax.axvline(median_age, color=_ACCENT2, linestyle="-.", linewidth=2,
               label=f"Median: {median_age:.0f} yrs", zorder=4)
    ax.set_title("Customer Age Distribution", fontsize=16, fontweight="bold",
                 color="#0f172a", pad=14)
    ax.set_xlabel("Customer Age", fontsize=11, fontweight="600")
    ax.set_ylabel("Count", fontsize=11, fontweight="600")
    ax.legend(fontsize=9.5, frameon=True, facecolor="white", edgecolor="#e5e7eb",
              framealpha=0.95, shadow=True)
    ax.grid(axis="y", alpha=0.12, color="#94a3b8")
    plt.tight_layout()
    return fig


def plot_sales_by_age_group(df):
    fig, ax = _light_fig((10, 5))
    age_sales = df.groupby("AgeGroup", observed=True)["Sales"].mean()
    age_counts = df.groupby("AgeGroup", observed=True)["Sales"].count()
    colors = [_ACCENT4, _BRAND, _ACCENT2, _ACCENT3, _ACCENT1]
    bars = ax.bar(range(len(age_sales)), age_sales.values, color=colors[:len(age_sales)],
                  width=0.6, edgecolor="white", linewidth=2, zorder=3)
    ax.set_xticks(range(len(age_sales)))
    ax.set_xticklabels([f"{g}\n(n={age_counts[g]:,})" for g in age_sales.index], rotation=0,
                       fontsize=9.5)
    ax.set_title("Average Sales by Age Group", fontsize=16, fontweight="bold",
                 color="#0f172a", pad=14)
    ax.set_xlabel("Age Group (sample size)", fontsize=11, fontweight="600")
    ax.set_ylabel("Average Sales ($)", fontsize=11, fontweight="600")
    overall_mean = df["Sales"].mean()
    ax.axhline(overall_mean, color=_ACCENT1, linestyle="--", linewidth=1.5,
               label=f"Overall Avg: ${overall_mean:,.0f}", zorder=2)
    for i, v in enumerate(age_sales.values):
        ax.text(i, v + 5, f"${v:,.0f}", ha="center", fontsize=11,
                fontweight="bold", color="#0f172a")
    ax.legend(fontsize=9.5, frameon=True, facecolor="white", edgecolor="#e5e7eb",
              framealpha=0.95, loc="upper right", shadow=True)
    ax.grid(axis="y", alpha=0.12, color="#94a3b8")
    plt.tight_layout()
    return fig


def plot_gender_analysis(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor="white")
    for ax in axes:
        ax.set_facecolor("#fafbff")

    gender_sales = df.groupby("Customer_Gender")["Sales"].sum()
    gender_counts = df.groupby("Customer_Gender")["Sales"].count()
    g_colors = [_BRAND, _ACCENT1]
    wedges, texts, autotexts = axes[0].pie(
        gender_sales, labels=gender_sales.index, autopct="%1.1f%%",
        colors=g_colors, explode=[0.03, 0.03], shadow=True,
        wedgeprops={"linewidth": 2.5, "edgecolor": "white"},
        textprops={"color": "#1e293b", "fontweight": "700", "fontsize": 12})
    for t in autotexts:
        t.set_color("#ffffff")
        t.set_fontweight("bold")
        t.set_fontsize(11)
    axes[0].set_title("Sales by Gender", fontsize=15, fontweight="bold",
                      color="#0f172a", pad=14)
    legend_labels = [f"{g}: ${v:,.0f} ({gender_counts[g]:,} txns)"
                     for g, v in zip(gender_sales.index, gender_sales.values)]
    axes[0].legend(wedges, legend_labels, fontsize=9, frameon=True,
                   facecolor="white", edgecolor="#e5e7eb", loc="lower center",
                   bbox_to_anchor=(0.5, -0.08), shadow=True)

    gender_sat = df.groupby("Customer_Gender")["Customer_Satisfaction"].mean()
    axes[1].bar(gender_sat.index, gender_sat.values, color=g_colors,
                width=0.5, edgecolor="white", linewidth=2, zorder=3)
    axes[1].set_title("Avg Satisfaction by Gender", fontsize=15, fontweight="bold",
                      color="#0f172a", pad=14)
    axes[1].set_ylabel("Satisfaction Score (1–5)", fontsize=11, fontweight="600")
    overall_sat = df["Customer_Satisfaction"].mean()
    axes[1].axhline(overall_sat, color=_ACCENT3, linestyle="--", linewidth=1.5,
                    label=f"Overall Avg: {overall_sat:.2f}", zorder=2)
    for i, v in enumerate(gender_sat.values):
        axes[1].text(i, v + 0.04, f"{v:.2f}", ha="center", fontsize=13,
                     fontweight="bold", color="#0f172a")
    axes[1].set_xticklabels(gender_sat.index, rotation=0, fontsize=11)
    axes[1].legend(fontsize=9, frameon=True, facecolor="white", edgecolor="#e5e7eb",
                   framealpha=0.95, loc="upper right", shadow=True)
    axes[1].grid(axis="y", alpha=0.12, color="#94a3b8")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].spines["left"].set_color("#cbd5e1")
    axes[1].spines["bottom"].set_color("#cbd5e1")
    plt.tight_layout()
    return fig


def plot_satisfaction_box(df):
    fig, ax = _light_fig((10, 5.5))
    products = sorted(df["Product"].unique())
    palette = {"Widget A": _BRAND, "Widget B": _ACCENT1,
               "Widget C": _ACCENT2, "Widget D": _ACCENT3}
    bp = sns.boxplot(data=df, x="Product", y="Customer_Satisfaction",
                     order=products, palette=palette, ax=ax, linewidth=1.5,
                     flierprops={"marker": "o", "markerfacecolor": "#94a3b8",
                                 "markersize": 4, "alpha": 0.5},
                     medianprops={"color": "#0f172a", "linewidth": 2.5},
                     whiskerprops={"color": "#64748b", "linewidth": 1.2},
                     capprops={"color": "#64748b", "linewidth": 1.2})
    ax.set_title("Satisfaction Distribution by Product", fontsize=16,
                 fontweight="bold", color="#0f172a", pad=14)
    ax.set_xlabel("Product", fontsize=11, fontweight="600")
    ax.set_ylabel("Satisfaction Score (1–5)", fontsize=11, fontweight="600")
    medians = df.groupby("Product")["Customer_Satisfaction"].median().reindex(products)
    counts = df.groupby("Product")["Customer_Satisfaction"].count().reindex(products)
    new_labels = [f"{p}\nMedian: {medians[p]:.2f} | n={counts[p]:,}" for p in products]
    ax.set_xticklabels(new_labels, fontsize=9)
    overall_med = df["Customer_Satisfaction"].median()
    ax.axhline(overall_med, color=_ACCENT1, linestyle="--", linewidth=1.3,
               label=f"Overall Median: {overall_med:.2f}", zorder=1)
    ax.legend(fontsize=9.5, frameon=True, facecolor="white", edgecolor="#e5e7eb",
              framealpha=0.95, loc="upper right", shadow=True)
    ax.grid(axis="y", alpha=0.12, color="#94a3b8")
    plt.tight_layout()
    return fig


def plot_correlation_matrix(df):
    fig, ax = _light_fig((8.5, 6.5))
    corr = df[["Sales", "Customer_Age", "Customer_Satisfaction"]].corr()
    # Custom diverging colormap
    from matplotlib.colors import LinearSegmentedColormap
    corr_cmap = LinearSegmentedColormap.from_list(
        "corr_div", ["#f43f5e", "#fda4af", "#ffffff", "#a5b4fc", "#4f46e5"])
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, annot=True, cmap=corr_cmap, vmin=-1, vmax=1, ax=ax,
                linewidths=4, linecolor="white",  # type: ignore[arg-type]
                annot_kws={"fontsize": 15, "fontweight": "bold"},
                cbar_kws={"label": "Correlation Coefficient", "shrink": 0.75},
                mask=mask, square=True)
    ax.set_title("Correlation Matrix (Pearson)", fontsize=16, fontweight="bold",
                 color="#0f172a", pad=14)
    ax.tick_params(labelsize=11)
    ax.text(0.5, -0.06, "Values range from −1 (inverse) to +1 (direct correlation)",
            transform=ax.transAxes, ha="center", fontsize=9.5, color="#6b7280", style="italic")
    plt.tight_layout()
    return fig


def plot_yearly_sales(df):
    fig, ax = _light_fig((11, 5.5))
    yearly = df.groupby("Year")["Sales"].sum()
    yearly_counts = df.groupby("Year")["Sales"].count()
    # Vibrant gradient from blue → indigo → brand
    n = len(yearly)
    bar_colors = [plt.cm.viridis(i / max(n - 1, 1)) for i in range(n)]  # type: ignore[attr-defined]
    bars = ax.bar(range(n), yearly.values, color=bar_colors,
                  width=0.6, edgecolor="white", linewidth=2, zorder=3)
    # Highlight best/worst years
    best_idx = np.argmax(yearly.values)
    worst_idx = np.argmin(yearly.values)
    bars[best_idx].set_edgecolor(_ACCENT2)
    bars[best_idx].set_linewidth(3)
    bars[worst_idx].set_edgecolor(_ACCENT1)
    bars[worst_idx].set_linewidth(3)
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"{y}\n({yearly_counts[y]:,} txns)" for y in yearly.index],
                       rotation=0, fontsize=9.5)
    ax.set_title("Yearly Sales Comparison", fontsize=16, fontweight="bold",
                 color="#0f172a", pad=14)
    ax.set_xlabel("Year (transaction count)", fontsize=11, fontweight="600")
    ax.set_ylabel("Total Sales ($)", fontsize=11, fontweight="600")
    mean_yearly = yearly.mean()
    ax.axhline(mean_yearly, color=_ACCENT1, linestyle="--", linewidth=1.5,
               label=f"Yearly Avg: ${mean_yearly:,.0f}", zorder=2)
    for i, v in enumerate(yearly.values):
        color = _ACCENT2 if i == best_idx else (_ACCENT1 if i == worst_idx else "#0f172a")
        ax.text(i, v + 1500, f"${v:,.0f}", ha="center", fontsize=10,
                fontweight="bold", color=color)
    ax.legend(fontsize=9.5, frameon=True, facecolor="white", edgecolor="#e5e7eb",
              framealpha=0.95, loc="upper right", shadow=True)
    ax.grid(axis="y", alpha=0.12, color="#94a3b8")
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  SIDEBAR (with Lucide icons)
# ═══════════════════════════════════════════════════════════════════════════

st.sidebar.markdown(
    f"""
    <div style="padding:8px 0 4px 0;">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:2px;">
            {lucide("bar-chart-3", 26, "#4f46e5")}
            <span style="font-size:1.35rem;font-weight:800;color:#0f172a;
                         letter-spacing:-0.02em;">InsightForge</span>
        </div>
        <span style="font-size:0.78rem;color:#64748b;font-weight:500;">
            AI-Powered Business Intelligence
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.divider()

NAV_ITEMS = {
    "Dashboard": "layout-dashboard",
    "Data Explorer": "search",
    "Visualizations": "line-chart",
    "AI Assistant": "bot-message-square",
}

page = st.sidebar.radio(
    "Navigate",
    list(NAV_ITEMS.keys()),
    index=0,
)

st.sidebar.divider()
st.sidebar.markdown(
    f"""
    <div style="padding:4px 0;">
        {lucide("copyright", 13, "#94a3b8")}
        <span style="font-size:0.72rem;color:#94a3b8;">
            InsightForge — Capstone Project
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════

df = load_sales_data()
summary = load_summary()


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════

if page == "Dashboard":
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
            {lucide("layout-dashboard", 28, "#4f46e5")}
            <h1 style="margin:0;padding:0;">Dashboard</h1>
        </div>
        <p style="color:#64748b !important;font-size:0.9rem;margin-top:2px;">
            Key performance indicators and business metrics at a glance.
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    if summary:
        o = summary["overall"]

        # ── KPI Row 1 (HTML metric cards) ──
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(metric_card("database", "Total Records",
                                    f"{o['total_records']:,}", "#4f46e5"),
                        unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card("dollar-sign", "Total Sales",
                                    f"${o['total_sales']:,}", "#16a34a"),
                        unsafe_allow_html=True)
        with c3:
            st.markdown(metric_card("trending-up", "Mean Sales",
                                    f"${o['mean_sales']:,.2f}", "#2563eb"),
                        unsafe_allow_html=True)
        with c4:
            st.markdown(metric_card("bar-chart-2", "Median Sales",
                                    f"${o['median_sales']:,.2f}", "#7c3aed"),
                        unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # ── KPI Row 2 ──
        c5, c6, c7, c8 = st.columns(4)
        with c5:
            st.markdown(metric_card("sigma", "Std Dev Sales",
                                    f"${o['std_sales']:,.2f}", "#ea580c"),
                        unsafe_allow_html=True)
        with c6:
            st.markdown(metric_card("arrow-up-down", "Sales Range",
                                    f"${o['min_sales']:,} – ${o['max_sales']:,}",
                                    "#0891b2"),
                        unsafe_allow_html=True)
        with c7:
            st.markdown(metric_card("star", "Avg Satisfaction",
                                    f"{o['mean_satisfaction']:.2f}", "#eab308"),
                        unsafe_allow_html=True)
        with c8:
            st.markdown(metric_card("users", "Avg Customer Age",
                                    f"{o['mean_customer_age']:.1f}", "#ec4899"),
                        unsafe_allow_html=True)

        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

        # ── Yearly Breakdown (custom HTML table) ──
        yearly_df = pd.DataFrame(summary["yearly"]).T
        yearly_df.index.name = "Year"
        yearly_df = yearly_df.rename(columns={
            "total_sales": "Total Sales",
            "avg_sales": "Avg Sales",
            "transaction_count": "Transactions",
            "avg_satisfaction": "Avg Satisfaction",
        })
        yearly_fmt = format_summary_df(yearly_df, {
            "Total Sales": "${:,.0f}",
            "Avg Sales": "${:,.2f}",
            "Transactions": "{:,.0f}",
            "Avg Satisfaction": "{:.2f}",
        })
        render_premium_table(yearly_fmt, "Yearly Breakdown", "calendar")

        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

        # ── Product & Region side by side ──
        col_left, col_right = st.columns(2)

        with col_left:
            prod_df = pd.DataFrame(summary["products"]).T
            prod_df.index.name = "Product"
            prod_df = prod_df.rename(columns={
                "total_sales": "Total Sales",
                "avg_sales": "Avg Sales",
                "median_sales": "Median",
                "std_sales": "Std Dev",
                "count": "Count",
            })
            prod_fmt = format_summary_df(prod_df, {
                "Total Sales": "${:,.0f}",
                "Avg Sales": "${:,.2f}",
                "Median": "${:,.2f}",
                "Std Dev": "${:,.2f}",
                "Count": "{:,.0f}",
            })
            render_premium_table(prod_fmt, "Product Analysis", "package")

        with col_right:
            reg_df = pd.DataFrame(summary["regions"]).T
            reg_df.index.name = "Region"
            reg_df = reg_df.rename(columns={
                "total_sales": "Total Sales",
                "avg_sales": "Avg Sales",
                "median_sales": "Median",
                "avg_satisfaction": "Avg Satisfaction",
            })
            reg_fmt = format_summary_df(reg_df, {
                "Total Sales": "${:,.0f}",
                "Avg Sales": "${:,.2f}",
                "Median": "${:,.2f}",
                "Avg Satisfaction": "{:.2f}",
            })
            render_premium_table(reg_fmt, "Regional Analysis", "globe")

        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

        # ── Demographics ──
        col_a, col_b = st.columns(2)

        with col_a:
            gen_df = pd.DataFrame(summary["gender"]).T
            gen_df.index.name = "Gender"
            gen_df = gen_df.rename(columns={
                "total_sales": "Total Sales",
                "avg_sales": "Avg Sales",
                "avg_satisfaction": "Avg Satisfaction",
                "avg_age": "Avg Age",
                "count": "Count",
            })
            gen_fmt = format_summary_df(gen_df, {
                "Total Sales": "${:,.0f}",
                "Avg Sales": "${:,.2f}",
                "Avg Satisfaction": "{:.2f}",
                "Avg Age": "{:.1f}",
                "Count": "{:,.0f}",
            })
            render_premium_table(gen_fmt, "Gender Segmentation", "user")

        with col_b:
            age_df = pd.DataFrame(summary["age_groups"]).T
            age_df.index.name = "Age Group"
            age_df = age_df.rename(columns={
                "total_sales": "Total Sales",
                "avg_sales": "Avg Sales",
                "avg_satisfaction": "Avg Satisfaction",
                "count": "Count",
            })
            age_fmt = format_summary_df(age_df, {
                "Total Sales": "${:,.0f}",
                "Avg Sales": "${:,.2f}",
                "Avg Satisfaction": "{:.2f}",
                "Count": "{:,.0f}",
            })
            render_premium_table(age_fmt, "Age Group Segmentation", "cake")
    else:
        st.warning("No summary data found. Run `insightforge_solution.py` first.")


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE: DATA EXPLORER
# ═══════════════════════════════════════════════════════════════════════════

elif page == "Data Explorer":
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
            {lucide("search", 28, "#4f46e5")}
            <h1 style="margin:0;padding:0;">Data Explorer</h1>
        </div>
        <p style="color:#64748b !important;font-size:0.9rem;margin-top:2px;">
            Filter, search and explore the raw sales dataset.
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Filters in styled columns
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        products = st.multiselect(
            "Product", df["Product"].unique().tolist(),
            default=df["Product"].unique().tolist())
    with col_f2:
        regions = st.multiselect(
            "Region", df["Region"].unique().tolist(),
            default=df["Region"].unique().tolist())
    with col_f3:
        years = st.multiselect(
            "Year", sorted(df["Year"].unique().tolist()),
            default=sorted(df["Year"].unique().tolist()))

    filtered = df[
        (df["Product"].isin(products)) &
        (df["Region"].isin(regions)) &
        (df["Year"].isin(years))
    ]

    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:6px;margin:8px 0 16px 0;">
            {lucide("filter", 15, "#64748b")}
            <span style="font-size:0.85rem;color:#64748b;font-weight:500;">
                Showing <strong style="color:#1e293b;">{len(filtered):,}</strong>
                of {len(df):,} records
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Quick stats
    if len(filtered) > 0:
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown(metric_card("hash", "Records",
                                    f"{len(filtered):,}", "#4f46e5"),
                        unsafe_allow_html=True)
        with s2:
            st.markdown(metric_card("dollar-sign", "Total Sales",
                                    f"${filtered['Sales'].sum():,.0f}", "#16a34a"),
                        unsafe_allow_html=True)
        with s3:
            st.markdown(metric_card("trending-up", "Avg Sales",
                                    f"${filtered['Sales'].mean():,.2f}", "#2563eb"),
                        unsafe_allow_html=True)
        with s4:
            st.markdown(metric_card("star", "Avg Satisfaction",
                                    f"{filtered['Customer_Satisfaction'].mean():.2f}",
                                    "#eab308"),
                        unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # Render data as premium HTML table (first 200 rows)
    display_df = (
        filtered[["Date", "Product", "Region", "Sales", "Customer_Age",
                  "Customer_Gender", "Customer_Satisfaction"]]
        .sort_values("Date", ascending=False)
        .head(200)
        .copy()
    )
    display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")
    display_df["Sales"] = display_df["Sales"].apply(lambda v: f"${v:,.0f}")
    display_df["Customer_Satisfaction"] = display_df["Customer_Satisfaction"].apply(
        lambda v: f"{v:.2f}")
    display_df = display_df.rename(columns={
        "Customer_Age": "Age",
        "Customer_Gender": "Gender",
        "Customer_Satisfaction": "Satisfaction",
    })
    display_df.index = range(1, len(display_df) + 1)
    display_df.index.name = "#"
    render_premium_table(display_df, "Sales Records", "table")


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE: VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════

elif page == "Visualizations":
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
            {lucide("line-chart", 28, "#4f46e5")}
            <h1 style="margin:0;padding:0;">Visualizations</h1>
        </div>
        <p style="color:#64748b !important;font-size:0.9rem;margin-top:2px;">
            Interactive charts and visual analytics.
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "  Trends  ", "  Products & Regions  ", "  Demographics  ",
        "  Satisfaction  ", "  Correlations  ",
    ])

    with tab1:
        st.markdown(
            f"<div style='margin:12px 0 8px 0;'>{lucide('trending-up', 18, '#4f46e5')}"
            f"<strong style='color:#1e293b;'>Sales Trends Over Time</strong></div>",
            unsafe_allow_html=True,
        )
        st.pyplot(plot_monthly_trend(df))
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        st.pyplot(plot_yearly_sales(df))

    with tab2:
        st.markdown(
            f"<div style='margin:12px 0 8px 0;'>{lucide('package', 18, '#4f46e5')}"
            f"<strong style='color:#1e293b;'>Product & Regional Performance</strong></div>",
            unsafe_allow_html=True,
        )
        st.pyplot(plot_sales_by_product(df))
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.pyplot(plot_sales_by_region(df))
        with col_v2:
            st.pyplot(plot_heatmap(df))

    with tab3:
        st.markdown(
            f"<div style='margin:12px 0 8px 0;'>{lucide('users', 18, '#4f46e5')}"
            f"<strong style='color:#1e293b;'>Customer Demographics</strong></div>",
            unsafe_allow_html=True,
        )
        st.pyplot(plot_age_distribution(df))
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.pyplot(plot_sales_by_age_group(df))
        with col_d2:
            st.pyplot(plot_gender_analysis(df))

    with tab4:
        st.markdown(
            f"<div style='margin:12px 0 8px 0;'>{lucide('star', 18, '#4f46e5')}"
            f"<strong style='color:#1e293b;'>Satisfaction Analysis</strong></div>",
            unsafe_allow_html=True,
        )
        st.pyplot(plot_satisfaction_box(df))

    with tab5:
        st.markdown(
            f"<div style='margin:12px 0 8px 0;'>{lucide('git-branch', 18, '#4f46e5')}"
            f"<strong style='color:#1e293b;'>Feature Correlations</strong></div>",
            unsafe_allow_html=True,
        )
        st.pyplot(plot_correlation_matrix(df))


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE: AI ASSISTANT (RAG Chat)
# ═══════════════════════════════════════════════════════════════════════════

elif page == "AI Assistant":

    # ── Session management helpers ──────────────────────────────────────
    def _init_sessions():
        """Ensure session store exists with at least one session."""
        if "chat_sessions" not in st.session_state:
            first_id = str(uuid.uuid4())[:8]
            st.session_state.chat_sessions = {
                first_id: {
                    "title": "New Chat",
                    "messages": [],
                    "created": datetime.now().strftime("%b %d, %I:%M %p"),
                }
            }
            st.session_state.active_session = first_id
        if "active_session" not in st.session_state:
            st.session_state.active_session = next(iter(st.session_state.chat_sessions))

    def _get_active_messages() -> list:
        sid = st.session_state.active_session
        return st.session_state.chat_sessions[sid]["messages"]

    def _auto_title(question: str) -> str:
        """Generate a short title from the first question."""
        text = question.strip()
        return (text[:38] + "…") if len(text) > 40 else text

    _init_sessions()

    # ── Sidebar: session controls ───────────────────────────────────────
    with st.sidebar:
        st.markdown("---")
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:6px;margin-bottom:10px;'>"
            f"{lucide('message-square-plus', 16, '#4f46e5')}"
            f"<strong style='font-size:0.85rem;color:#1e293b;'>Chat Sessions</strong></div>",
            unsafe_allow_html=True,
        )

        # New chat button
        if st.button("＋  New Chat", key="new_chat_btn", use_container_width=True):
            new_id = str(uuid.uuid4())[:8]
            st.session_state.chat_sessions[new_id] = {
                "title": "New Chat",
                "messages": [],
                "created": datetime.now().strftime("%b %d, %I:%M %p"),
            }
            st.session_state.active_session = new_id
            st.rerun()

        # Session list
        sessions = st.session_state.chat_sessions
        session_ids = list(sessions.keys())
        session_labels = []
        for sid in session_ids:
            s = sessions[sid]
            msg_count = len([m for m in s["messages"] if m["role"] == "user"])
            label = f"{s['title']}  ({msg_count} Q" + ("s" if msg_count != 1 else "") + ")"
            session_labels.append(label)

        current_idx = session_ids.index(st.session_state.active_session) if st.session_state.active_session in session_ids else 0
        selected_idx = st.radio(
            "Sessions", range(len(session_ids)),
            format_func=lambda i: session_labels[i],
            index=current_idx, key="session_radio", label_visibility="collapsed",
        )
        if session_ids[selected_idx] != st.session_state.active_session:
            st.session_state.active_session = session_ids[selected_idx]
            st.rerun()

        # Clear / Delete controls
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🗑 Clear", key="clear_chat_btn", use_container_width=True,
                         help="Clear messages in this session"):
                st.session_state.chat_sessions[st.session_state.active_session]["messages"] = []
                st.session_state.chat_sessions[st.session_state.active_session]["title"] = "New Chat"
                st.rerun()
        with col_b:
            if len(sessions) > 1:
                if st.button("✕ Delete", key="delete_chat_btn", use_container_width=True,
                             help="Delete this session"):
                    del st.session_state.chat_sessions[st.session_state.active_session]
                    st.session_state.active_session = next(iter(st.session_state.chat_sessions))
                    st.rerun()

    # ── Main area ───────────────────────────────────────────────────────
    active = st.session_state.chat_sessions[st.session_state.active_session]

    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
            {lucide("bot-message-square", 28, "#4f46e5")}
            <h1 style="margin:0;padding:0;">AI Assistant</h1>
        </div>
        <p style="color:#64748b !important;font-size:0.9rem;margin-top:2px;">
            Ask questions about the sales data or uploaded PDF knowledge base.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # Session info bar
    st.markdown(
        f"<div style='background:#f1f5f9;border-radius:8px;padding:8px 14px;"
        f"margin-bottom:12px;display:flex;align-items:center;gap:8px;'>"
        f"{lucide('message-square', 14, '#6366f1')}"
        f"<span style='font-size:0.82rem;color:#475569;'>"
        f"<strong>{active['title']}</strong> · Started {active['created']}"
        f" · {len([m for m in active['messages'] if m['role']=='user'])} questions</span></div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Display chat history for active session
    for msg in active["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Suggested questions (when empty, shown BEFORE chat_input)
    if not active["messages"]:
        st.markdown(
            f"""
            <div style="margin-top:24px;">
                <div style="display:flex;align-items:center;gap:6px;margin-bottom:14px;">
                    {lucide("lightbulb", 18, "#eab308")}
                    <strong style="color:#1e293b;font-size:0.95rem;">
                        Try asking
                    </strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        suggestions = [
            ("trending-up", "What are the overall sales trends in the dataset?"),
            ("award", "Which product has the highest total sales?"),
            ("star", "How does customer satisfaction vary across regions?"),
            ("users", "What insights can you provide about customer demographics?"),
        ]
        st.html("""<style>
            div[data-testid="stColumns"] button[kind="secondary"] {
                background: #f8fafc !important;
                border: 1px solid #e2e8f0 !important;
                border-radius: 10px !important;
                padding: 14px 16px !important;
                text-align: left !important;
                color: #475569 !important;
                font-size: 0.85rem !important;
                transition: all 0.2s ease !important;
                width: 100% !important;
            }
            div[data-testid="stColumns"] button[kind="secondary"]:hover {
                background: #eef2ff !important;
                border-color: #6366f1 !important;
                color: #4338ca !important;
                cursor: pointer !important;
            }
        </style>""")
        cols = st.columns(2)
        for i, (icon, text) in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(text, key=f"suggest_{i}", use_container_width=True):
                    st.session_state["_pending_question"] = text
                    st.rerun()

    # Input
    user_input = st.chat_input("Ask InsightForge a question…")

    # Check if a suggestion button was clicked
    if "_pending_question" in st.session_state:
        user_input = st.session_state.pop("_pending_question")

    if user_input:
        active["messages"].append({"role": "user", "content": user_input})
        # Auto-title from first question
        if active["title"] == "New Chat":
            active["title"] = _auto_title(user_input)

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            qa_chain = load_rag_chain()
            if qa_chain is None:
                answer = ("RAG pipeline is not available. Please check your "
                          "Azure OpenAI credentials and vector store.")
            else:
                with st.spinner("Analyzing…"):
                    try:
                        result = qa_chain.invoke({"query": user_input})
                        answer = result["result"]
                    except Exception as e:
                        answer = f"Error: {e}"
            st.markdown(answer)

        active["messages"].append({"role": "assistant", "content": answer})
        st.rerun()  # refresh sidebar session counter
