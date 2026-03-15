"""
InsightForge — AI-Powered Business Intelligence Assistant
Capstone Project

Part 1: Data Preparation, Knowledge Base, RAG, Memory
Part 2: Evaluation with QAEvalChain

Uses RetrievalQA, ConversationalRetrievalChain, ConversationBufferMemory,
QAEvalChain, FAISS, HuggingFace embeddings, Azure OpenAI.
"""

import os
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore
from dotenv import load_dotenv  # type: ignore

from langchain.chains import RetrievalQA  # type: ignore
from langchain.chains import ConversationalRetrievalChain  # type: ignore
from langchain.evaluation.qa import QAEvalChain  # type: ignore
from langchain.memory import ConversationBufferMemory  # type: ignore
from langchain_community.document_loaders import PyPDFLoader  # type: ignore
from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
from langchain_community.vectorstores import FAISS  # type: ignore
from langchain_core.documents import Document  # type: ignore
from langchain_core.prompts import PromptTemplate  # type: ignore
from langchain_openai import AzureChatOpenAI  # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Configuration (loaded from .env, never hardcoded) ---
load_dotenv(Path(__file__).resolve().parent / ".env")

AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

# --- File paths ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datasets"
PDF_DIR = DATA_DIR / "PDF Folder"
OUTPUT_DIR = BASE_DIR / "output"
VECTORDB_DIR = BASE_DIR / "vectorstore"
PICKLE_FILE = OUTPUT_DIR / "sales_summary.pkl"
PLOTS_DIR = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)


# ============================================================================
# STEP 1 — DATA PREPARATION & ADVANCED SUMMARY
# ============================================================================

def load_sales_data():
    """Load the sales CSV and add derived columns."""
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


def compute_advanced_summary(df):
    """Compute key metrics: overall stats, yearly trends, product/region/demographic breakdowns."""
    summary = {}

    # Overall stats
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

    # Yearly breakdown
    yearly = df.groupby("Year").agg(
        total_sales=("Sales", "sum"),
        avg_sales=("Sales", "mean"),
        transaction_count=("Sales", "count"),
        avg_satisfaction=("Customer_Satisfaction", "mean"),
    ).round(2)
    summary["yearly"] = yearly.to_dict("index")

    # Product breakdown
    products = df.groupby("Product").agg(
        total_sales=("Sales", "sum"),
        avg_sales=("Sales", "mean"),
        median_sales=("Sales", "median"),
        std_sales=("Sales", "std"),
        count=("Sales", "count"),
    ).round(2)
    summary["products"] = products.to_dict("index")

    # Regional breakdown
    regions = df.groupby("Region").agg(
        total_sales=("Sales", "sum"),
        avg_sales=("Sales", "mean"),
        median_sales=("Sales", "median"),
        avg_satisfaction=("Customer_Satisfaction", "mean"),
    ).round(2)
    summary["regions"] = regions.to_dict("index")

    # Gender breakdown
    gender = df.groupby("Customer_Gender").agg(
        total_sales=("Sales", "sum"),
        avg_sales=("Sales", "mean"),
        avg_satisfaction=("Customer_Satisfaction", "mean"),
        avg_age=("Customer_Age", "mean"),
        count=("Sales", "count"),
    ).round(2)
    summary["gender"] = gender.to_dict("index")

    # Age group breakdown
    age = df.groupby("AgeGroup", observed=True).agg(
        total_sales=("Sales", "sum"),
        avg_sales=("Sales", "mean"),
        avg_satisfaction=("Customer_Satisfaction", "mean"),
        count=("Sales", "count"),
    ).round(2)
    summary["age_groups"] = age.to_dict("index")

    # Correlations
    numeric_cols = ["Sales", "Customer_Age", "Customer_Satisfaction"]
    summary["correlations"] = df[numeric_cols].corr().round(4).to_dict()

    return summary


def print_summary(summary):
    """Print a clean, human-readable plain-text summary to console."""
    o = summary["overall"]

    print()
    print("=" * 70)
    print("  INSIGHTFORGE — ADVANCED DATA SUMMARY")
    print("=" * 70)
    print(f"  Dataset: {o['total_records']} records | {o['date_range']}")
    print(f"  Total Sales: {o['total_sales']:,}")
    print(f"  Mean Sales: {o['mean_sales']} | Median: {o['median_sales']} | Std: {o['std_sales']}")
    print(f"  Sales Range: {o['min_sales']} - {o['max_sales']}")
    print(f"  Mean Satisfaction: {o['mean_satisfaction']} | Median: {o['median_satisfaction']}")
    print(f"  Mean Customer Age: {o['mean_customer_age']} | Median: {o['median_customer_age']}")

    print("\n  --- Sales by Year ---")
    for year, d in sorted(summary["yearly"].items()):
        print(f"  {year}: Total={d['total_sales']:,}  Avg={d['avg_sales']}  "
              f"Txns={d['transaction_count']}  AvgSat={d['avg_satisfaction']}")

    print("\n  --- Product Analysis ---")
    for product, d in sorted(summary["products"].items()):
        print(f"  {product}: Total={d['total_sales']:,}  Avg={d['avg_sales']}  "
              f"Median={d['median_sales']}  Std={d['std_sales']}  Count={d['count']}")

    print("\n  --- Regional Analysis ---")
    for region, d in sorted(summary["regions"].items()):
        print(f"  {region}: Total={d['total_sales']:,}  Avg={d['avg_sales']}  "
              f"Median={d['median_sales']}  AvgSat={d['avg_satisfaction']}")

    print("\n  --- Gender Segmentation ---")
    for gender, d in sorted(summary["gender"].items()):
        print(f"  {gender}: Total={d['total_sales']:,}  Avg={d['avg_sales']}  "
              f"AvgSat={d['avg_satisfaction']}  Count={d['count']}")

    print("\n  --- Age Group Segmentation ---")
    for age, d in summary["age_groups"].items():
        print(f"  {age}: Total={d['total_sales']:,}  Avg={d['avg_sales']}  "
              f"AvgSat={d['avg_satisfaction']}  Count={d['count']}")

    print("\n  --- Correlations ---")
    for col, correlations in summary["correlations"].items():
        others = ", ".join(f"{k}={v}" for k, v in correlations.items() if k != col)
        print(f"  {col}: {others}")
    print("=" * 70)


def generate_plots(df):
    """Generate 10 visualization plots and save them as PNGs."""
    sns.set_theme(style="whitegrid")
    count = 0

    # 1. Monthly sales trend
    fig, ax = plt.subplots(figsize=(12, 5))
    monthly = df.groupby("YearMonth")["Sales"].sum().reset_index().sort_values("YearMonth")
    ax.plot(monthly["YearMonth"], monthly["Sales"], color="#2563eb", linewidth=1.5)
    ax.fill_between(range(len(monthly)), monthly["Sales"].values, alpha=0.1, color="#2563eb")
    ax.set_title("Monthly Sales Trend", fontsize=14, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Sales ($)")
    step = max(1, len(monthly) // 8)
    ax.set_xticks(range(0, len(monthly), step))
    ax.set_xticklabels(monthly["YearMonth"].values[::step], rotation=45, fontsize=7)
    plt.tight_layout()
    fig.savefig(str(PLOTS_DIR / "01_sales_trend.png"), dpi=150)
    plt.close(fig)
    count += 1

    # 2. Sales by product (horizontal bar)
    fig, ax = plt.subplots(figsize=(8, 5))
    product_sales = df.groupby("Product")["Sales"].sum().sort_values()
    product_sales.plot(kind="barh", ax=ax, color=sns.color_palette("viridis", len(product_sales)))
    ax.set_title("Total Sales by Product", fontsize=14, fontweight="bold")
    ax.set_xlabel("Total Sales ($)")
    for i, v in enumerate(product_sales.values):
        ax.text(v + 1000, i, f"${v:,.0f}", va="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(str(PLOTS_DIR / "02_sales_by_product.png"), dpi=150)
    plt.close(fig)
    count += 1

    # 3. Sales by region (pie chart)
    fig, ax = plt.subplots(figsize=(8, 5))
    region_sales = df.groupby("Region")["Sales"].sum()
    ax.pie(region_sales, labels=region_sales.index, autopct="%1.1f%%",
           colors=["#3b82f6", "#ef4444", "#22c55e", "#f59e0b"], startangle=90)
    ax.set_title("Sales Distribution by Region", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(str(PLOTS_DIR / "03_sales_by_region.png"), dpi=150)
    plt.close(fig)
    count += 1

    # 4. Product x Region heatmap
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot = df.pivot_table(values="Sales", index="Product", columns="Region", aggfunc="sum")
    sns.heatmap(pivot, annot=True, fmt=",.0f", cmap="YlGnBu", ax=ax, linewidths=0.5)  # type: ignore[arg-type]
    ax.set_title("Product x Region Sales Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(str(PLOTS_DIR / "04_product_region_heatmap.png"), dpi=150)
    plt.close(fig)
    count += 1

    # 5. Customer age distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["Customer_Age"], bins=20, kde=True, color="steelblue", ax=ax)  # type: ignore[arg-type]
    ax.set_title("Customer Age Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(str(PLOTS_DIR / "05_age_distribution.png"), dpi=150)
    plt.close(fig)
    count += 1

    # 6. Average sales by age group
    fig, ax = plt.subplots(figsize=(8, 5))
    age_sales = df.groupby("AgeGroup", observed=True)["Sales"].mean()
    age_sales.plot(kind="bar", ax=ax, color=sns.color_palette("rocket", 5))
    ax.set_title("Average Sales by Age Group", fontsize=14, fontweight="bold")
    ax.set_ylabel("Average Sales ($)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    fig.savefig(str(PLOTS_DIR / "06_sales_by_age_group.png"), dpi=150)
    plt.close(fig)
    count += 1

    # 7. Gender analysis (two subplots)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    gender_sales = df.groupby("Customer_Gender")["Sales"].sum()
    gender_sales.plot(kind="pie", autopct="%1.1f%%", ax=axes[0], colors=["#66b3ff", "#ff9999"])
    axes[0].set_title("Sales by Gender", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("")
    gender_sat = df.groupby("Customer_Gender")["Customer_Satisfaction"].mean()
    gender_sat.plot(kind="bar", ax=axes[1], color=["#66b3ff", "#ff9999"])
    axes[1].set_title("Avg Satisfaction by Gender", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Score")
    axes[1].set_xticklabels(gender_sat.index, rotation=0)
    plt.tight_layout()
    fig.savefig(str(PLOTS_DIR / "07_gender_analysis.png"), dpi=150)
    plt.close(fig)
    count += 1

    # 8. Satisfaction distribution by product (box plot)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="Product", y="Customer_Satisfaction", palette="Set2", ax=ax)
    ax.set_title("Satisfaction Distribution by Product", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(str(PLOTS_DIR / "08_satisfaction_distribution.png"), dpi=150)
    plt.close(fig)
    count += 1

    # 9. Correlation matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df[["Sales", "Customer_Age", "Customer_Satisfaction"]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax, linewidths=0.5)  # type: ignore[arg-type]
    ax.set_title("Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(str(PLOTS_DIR / "09_correlation_matrix.png"), dpi=150)
    plt.close(fig)
    count += 1

    # 10. Yearly sales comparison (bar)
    fig, ax = plt.subplots(figsize=(10, 5))
    yearly = df.groupby("Year")["Sales"].sum()
    yearly.plot(kind="bar", ax=ax, color=sns.color_palette("crest", len(yearly)))
    ax.set_title("Yearly Sales Comparison", fontsize=14, fontweight="bold")
    ax.set_ylabel("Total Sales ($)")
    ax.set_xticklabels(yearly.index, rotation=0)
    for i, v in enumerate(yearly.values):
        ax.text(i, v + 1000, f"${v:,.0f}", ha="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(str(PLOTS_DIR / "10_yearly_sales.png"), dpi=150)
    plt.close(fig)
    count += 1

    print(f"  Saved {count} plots to {PLOTS_DIR}")


def save_summary_pickle(summary):
    """Save the summary dict to a pickle file for the Streamlit app."""
    with open(PICKLE_FILE, "wb") as f:
        pickle.dump(summary, f)
    print(f"  Summary saved to {PICKLE_FILE}")


def prepare_summary_text(summary):
    """Convert summary dict to natural language for the RAG knowledge base."""
    o = summary["overall"]
    lines = [
        "InsightForge Sales Data Advanced Summary",
        f"Dataset contains {o['total_records']} records from {o['date_range']}.",
        f"Total sales revenue: {o['total_sales']:,}.",
        f"Mean sales: {o['mean_sales']}, Median: {o['median_sales']}, Std: {o['std_sales']}.",
        f"Sales range: {o['min_sales']} to {o['max_sales']}.",
        f"Average customer satisfaction: {o['mean_satisfaction']} (median {o['median_satisfaction']}).",
        f"Average customer age: {o['mean_customer_age']} (median {o['median_customer_age']}).",
        "",
        "Sales Performance by Year:",
    ]
    for year, d in sorted(summary["yearly"].items()):
        lines.append(f"  Year {year}: total sales {d['total_sales']:,}, average {d['avg_sales']}, "
                     f"transactions {d['transaction_count']}, avg satisfaction {d['avg_satisfaction']}")

    lines.append("\nProduct Analysis:")
    for product, d in sorted(summary["products"].items()):
        lines.append(f"  {product}: total sales {d['total_sales']:,}, average {d['avg_sales']}, "
                     f"median {d['median_sales']}, std {d['std_sales']}, count {d['count']}")

    lines.append("\nRegional Analysis:")
    for region, d in sorted(summary["regions"].items()):
        lines.append(f"  {region}: total sales {d['total_sales']:,}, average {d['avg_sales']}, "
                     f"median {d['median_sales']}, avg satisfaction {d['avg_satisfaction']}")

    lines.append("\nCustomer Segmentation by Gender:")
    for gender, d in sorted(summary["gender"].items()):
        lines.append(f"  {gender}: total {d['total_sales']:,}, avg {d['avg_sales']}, "
                     f"avg satisfaction {d['avg_satisfaction']}, count {d['count']}")

    lines.append("\nCustomer Segmentation by Age Group:")
    for age, d in summary["age_groups"].items():
        lines.append(f"  {age}: total {d['total_sales']:,}, avg {d['avg_sales']}, "
                     f"avg satisfaction {d['avg_satisfaction']}, count {d['count']}")

    lines.append("\nCorrelations:")
    for col, correlations in summary["correlations"].items():
        others = ", ".join(f"{k}={v}" for k, v in correlations.items() if k != col)
        lines.append(f"  {col}: {others}")

    return "\n".join(lines)


# ============================================================================
# STEP 2 — KNOWLEDGE BASE CREATION
# ============================================================================

def load_pdf_documents():
    """Load all PDFs from the dataset folder using PyPDFLoader."""
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    all_docs = []
    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        print(f"    {pdf_path.name} — {len(pages)} pages")
        all_docs.extend(pages)
    print(f"  Total pages loaded: {len(all_docs)}")
    return all_docs


def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    print(f"  Split into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks


def get_embeddings():
    """Return HuggingFace local embeddings (all-MiniLM-L6-v2, 384 dimensions, free)."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def create_vector_store(chunks, summary_text):
    """Build a FAISS vector store from PDF chunks plus the sales summary."""
    # Turn the summary into document chunks too
    summary_doc = Document(
        page_content=summary_text,
        metadata={"source": "sales_data_summary", "type": "structured_data"},
    )
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    summary_chunks = splitter.split_documents([summary_doc])

    all_chunks = chunks + summary_chunks
    print(f"  Total chunks for vector store: {len(all_chunks)} "
          f"({len(chunks)} PDF + {len(summary_chunks)} summary)")

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(
        documents=all_chunks,
        embedding=embeddings,
    )
    vectorstore.save_local(str(VECTORDB_DIR))
    print(f"  FAISS vector store saved to {VECTORDB_DIR}")
    return vectorstore


def load_vector_store():
    """Load an existing FAISS vector store from disk."""
    embeddings = get_embeddings()
    return FAISS.load_local(
        str(VECTORDB_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def get_llm():
    """Create an Azure OpenAI chat model instance."""
    return AzureChatOpenAI(
        azure_deployment=AZURE_OPENAI_DEPLOYMENT,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,  # type: ignore[arg-type]
        api_version=AZURE_OPENAI_API_VERSION,
        temperature=0.3,
    )


# ============================================================================
# STEP 3 — RAG SYSTEM (RetrievalQA + Prompt Engineering)
# ============================================================================

def build_retrieval_qa(vectorstore):
    """Build a RetrievalQA chain with a custom prompt template."""
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are InsightForge, an AI Business Intelligence Assistant.\n"
            "Analyze business data and provide actionable insights with specific numbers.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        ),
    )

    llm = get_llm()
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 8})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain


# ============================================================================
# STEP 4 — MEMORY INTEGRATION (ConversationalRetrievalChain)
# ============================================================================

def build_conversational_chain(vectorstore):
    """Build a ConversationalRetrievalChain with ConversationBufferMemory."""
    llm = get_llm()
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 8})

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
    )
    return chain, memory


# ============================================================================
# STEP 5 — MODEL EVALUATION (QAEvalChain)
# ============================================================================

def evaluate_model(qa_chain):
    """Evaluate the RAG model using LangChain's QAEvalChain."""
    eval_llm = get_llm()

    qa_pairs = [
        {"query": "What is the total number of sales records in the dataset?",
         "answer": "The dataset contains 2500 records."},
        {"query": "Which products are available in the dataset?",
         "answer": "Widget A, Widget B, Widget C, and Widget D."},
        {"query": "What are the four regions in the data?",
         "answer": "North, South, East, and West."},
        {"query": "What is the average customer satisfaction score?",
         "answer": "The average customer satisfaction is approximately 3.03."},
        {"query": "What is the date range of the sales data?",
         "answer": "The data ranges from 2022-01-01 to 2028-11-04."},
    ]

    # Get predictions from the QA chain
    predictions = []
    for pair in qa_pairs:
        result = qa_chain.invoke({"query": pair["query"]})
        predictions.append({"result": result["result"]})

    # Grade using QAEvalChain
    eval_chain = QAEvalChain.from_llm(eval_llm)
    graded = eval_chain.evaluate(examples=qa_pairs, predictions=predictions)

    correct = 0
    for pair, pred, grade in zip(qa_pairs, predictions, graded):
        grade_text = grade.get("results", grade.get("text", "")).strip()
        passed = "CORRECT" in grade_text.upper() and "INCORRECT" not in grade_text.upper()
        if passed:
            correct += 1
        status = "PASS" if passed else "FAIL"
        print(f"  {status}  Q: {pair['query']}")
        print(f"         Expected: {pair['answer']}")
        print(f"         Got:      {pred['result'][:80]}...")
        print(f"         Grade:    {grade_text}")
        print()

    total = len(qa_pairs)
    print(f"  Score: {correct}/{total} ({100 * correct / total:.0f}%)")
    return correct, total, graded


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_part1():
    """Execute the full pipeline: data prep, knowledge base, RAG, memory, evaluation."""
    print("\nInsightForge — AI-Powered Business Intelligence Assistant")
    print("=" * 58)

    # Step 1: Data Preparation
    print("\n" + "-" * 50)
    print("STEP 1: Data Preparation & Advanced Summary")
    print("-" * 50)
    df = load_sales_data()
    summary = compute_advanced_summary(df)
    print_summary(summary)
    generate_plots(df)
    save_summary_pickle(summary)

    # Step 2: Knowledge Base
    print("\n" + "-" * 50)
    print("STEP 2: Knowledge Base Creation")
    print("-" * 50)
    pdf_docs = load_pdf_documents()
    chunks = chunk_documents(pdf_docs)
    summary_text = prepare_summary_text(summary)
    vectorstore = create_vector_store(chunks, summary_text)

    # Step 3: RAG System
    print("\n" + "-" * 50)
    print("STEP 3: RAG System with RetrievalQA Chain")
    print("-" * 50)
    qa_chain = build_retrieval_qa(vectorstore)

    sample_questions = [
        "What are the overall sales trends in the dataset?",
        "Which product has the highest total sales?",
        "How does customer satisfaction vary across regions?",
        "What insights can you provide about customer demographics?",
    ]
    print("\n  --- Sample RAG Responses ---\n")
    for q in sample_questions:
        result = qa_chain.invoke({"query": q})
        print(f"  Q: {q}")
        print(f"  A: {result['result']}\n")

    # Step 4: Conversational Memory
    print("\n" + "-" * 50)
    print("STEP 4: Memory Integration (ConversationalRetrievalChain)")
    print("-" * 50)
    conv_chain, memory = build_conversational_chain(vectorstore)

    conversation = [
        "What is the total sales revenue in the dataset?",
        "How does that break down by product?",
        "Which region performs best among those products?",
    ]
    print("\n  --- Conversation with Memory ---\n")
    for q in conversation:
        result = conv_chain.invoke({"question": q})
        print(f"  User: {q}")
        print(f"  InsightForge: {result['answer']}\n")

    # Step 5: Model Evaluation
    print("\n" + "-" * 50)
    print("STEP 5: Model Evaluation (QAEvalChain)")
    print("-" * 50)
    print("\n  --- Evaluation Results ---\n")
    evaluate_model(qa_chain)

    print("\nAll steps completed successfully.\n")


if __name__ == "__main__":
    run_part1()
