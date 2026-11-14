
import os
import re
import json
import base64
import hashlib
from pathlib import Path
from datetime import date, datetime, timedelta

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================

API_KEY = os.getenv("SARVAM_API_KEY", "").strip()
BASE_URL = "https://api.sarvam.ai"

STT_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v4")
STT_LANGUAGE = os.getenv("SARVAM_STT_LANGUAGE", "unknown")
STT_MODE = os.getenv("SARVAM_STT_MODE", "codemix")
TRANSLATION_MODEL = os.getenv(
    "SARVAM_TRANSLATION_MODEL", "sarvam-translate:v1"
)

TTS_MODEL = os.getenv("SARVAM_TTS_MODEL", "bulbul:v4-flash")
TTS_SPEAKER = os.getenv("SARVAM_TTS_SPEAKER", "simran_en_customer")
TTS_LANGUAGE = os.getenv("SARVAM_TTS_LANGUAGE", "en-IN")
TTS_SUPPORTED_LANGUAGES = {
    "en-IN", "bn-IN", "gu-IN", "hi-IN", "kn-IN",
    "ml-IN", "mr-IN", "od-IN", "pa-IN", "ta-IN", "te-IN",
}

CHAT_MODEL = os.getenv("SARVAM_CHAT_MODEL", "sarvam-105b")

PAYMENT_COLUMNS = [
    "Transaction ID",
    "Customer ID",
    "Customer Name",
    "Amount (INR)",
    "Transaction Date & Time",
    "Payment Source",
    "Payment Status",
]

INVENTORY_COLUMNS = [
    "Product ID",
    "Product Name",
    "Category",
    "Unit",
    "Quantity in Stock",
    "Reorder Level",
    "Purchase Price (₹)",
    "Selling Price (₹)",
    "Supplier",
    "Expiry Date",
]

# Only these operations can be executed.
ALLOWED_DATASETS = {"payments", "inventory"}

ALLOWED_INTENTS = {
    "revenue",
    "transaction_count",
    "payment_status",
    "payment_sources",
    "top_customers",
    "recent_transactions",
    "product_stock",
    "low_stock",
    "expiry",
    "reorder",
    "inventory_value",
    "product_list",
    "unsupported",
}


# ============================================================
# DATA LOADING AND VALIDATION
# ============================================================

def load_payment_workbook(file):
    df = pd.read_excel(
        file,
        sheet_name="Transactions",
        engine="openpyxl",
    )

    missing = [c for c in PAYMENT_COLUMNS if c not in df.columns]

    if missing:
        raise ValueError(
            "Transactions sheet is missing columns: "
            + ", ".join(missing)
        )

    df["Amount (INR)"] = pd.to_numeric(
        df["Amount (INR)"], errors="coerce"
    )

    df["Transaction Date & Time"] = pd.to_datetime(
        df["Transaction Date & Time"], errors="coerce"
    )

    df["Payment Status"] = (
        df["Payment Status"].fillna("").astype(str).str.strip().str.title()
    )

    df["Payment Source"] = (
        df["Payment Source"].fillna("Unknown").astype(str).str.strip()
    )

    df["Customer Name"] = (
        df["Customer Name"].fillna("Unknown").astype(str).str.strip()
    )

    df = df.dropna(
        subset=["Amount (INR)", "Transaction Date & Time"]
    ).copy()

    if df.empty:
        raise ValueError("No valid payment transactions were found.")

    return df


def load_inventory_workbook(file):
    df = pd.read_excel(
        file,
        sheet_name="Inventory",
        engine="openpyxl",
    )

    missing = [c for c in INVENTORY_COLUMNS if c not in df.columns]

    if missing:
        raise ValueError(
            "Inventory sheet is missing columns: "
            + ", ".join(missing)
        )

    numeric_columns = [
        "Quantity in Stock",
        "Reorder Level",
        "Purchase Price (₹)",
        "Selling Price (₹)",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Expiry Date"] = pd.to_datetime(
        df["Expiry Date"], errors="coerce"
    ).dt.date

    for col in ["Product Name", "Product ID", "Category", "Supplier"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    return df


# ============================================================
# SARVAM AI: NATURAL LANGUAGE -> STRUCTURED QUERY
# ============================================================

def call_sarvam_chat(question):
    if not API_KEY:
        raise RuntimeError("SARVAM_API_KEY is missing.")

    system_prompt = """
You are the query understanding engine for Dukaan Saathi,
an AI assistant for small shop owners in India.

Convert the merchant's question into ONE JSON object.

Choose exactly one dataset:
payments or inventory.

Allowed intents:
revenue, transaction_count, payment_status, payment_sources,
top_customers, recent_transactions, product_stock, low_stock,
expiry, reorder, inventory_value, product_list, unsupported.

Return this JSON structure:

{
  "dataset": "payments",
  "intent": "revenue",
  "period": "today",
  "start_date": null,
  "end_date": null,
  "status": "Success",
  "product_name": null,
  "group_by": null,
  "top_n": 5
}

Rules:

1. Understand English, Hindi, Telugu and mixed-language questions.
2. Do not answer the merchant. Only return JSON.
3. Use dataset "payments" for payment and sales questions.
4. Use dataset "inventory" for stock, products and expiry questions.
5. "How much did I sell?" means revenue.
6. "How many transactions?" means transaction_count.
7. "Who are my best customers?" means top_customers.
8. "Which payment methods are popular?" means payment_sources.
9. "Show latest payments" means recent_transactions.
10. "How much stock of rice?" means product_stock.
11. "Which products are running low?" means low_stock.
12. "Which products expire tomorrow?" means expiry with period tomorrow.
13. "Which products expire in the next 7 days?" means expiry with period next_7_days.
14. "What should I reorder?" means reorder.
15. "What is my inventory value?" means inventory_value.
16. "Show all products" means product_list.
17. "Today" means today.
18. "Yesterday" means yesterday.
19. "Last hour" means last_hour.
20. "This week" means this_week.
21. "This month" means this_month.
22. "Tomorrow" means tomorrow, NOT next 30 days.
23. "Day after tomorrow" means day_after_tomorrow.
24. If the user gives a product name, preserve it in product_name.
25. If a date range is explicitly supplied, use start_date and end_date
    in YYYY-MM-DD format when possible.
26. top_n should be between 1 and 20.
27. Do not invent a product name or date.
28. If the question cannot be mapped to these operations, use unsupported.
"""

    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "temperature": 0.1,
        "max_tokens": 400,
        "response_format": {"type": "json_object"},
    }

    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        headers={
            "api-subscription-key": API_KEY,
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=45,
    )

    if not response.ok:
        raise RuntimeError(
            f"Sarvam AI query understanding failed "
            f"({response.status_code}): {response.text[:500]}"
        )

    result = response.json()
    content = result["choices"][0]["message"]["content"]

    plan = json.loads(content)

    return validate_plan(plan)


def validate_plan(plan):
    if not isinstance(plan, dict):
        raise ValueError("AI returned an invalid query plan.")

    dataset = str(plan.get("dataset", "")).lower()
    intent = str(plan.get("intent", "unsupported")).lower()

    if dataset not in ALLOWED_DATASETS:
        raise ValueError("AI selected an invalid dataset.")

    if intent not in ALLOWED_INTENTS:
        intent = "unsupported"

    try:
        top_n = int(plan.get("top_n") or 5)
    except (TypeError, ValueError):
        top_n = 5

    top_n = max(1, min(top_n, 20))

    start_date = plan.get("start_date")
    end_date = plan.get("end_date")

    # Accept dates only in ISO format.
    for value in [start_date, end_date]:
        if value is not None:
            try:
                datetime.strptime(str(value), "%Y-%m-%d")
            except ValueError:
                raise ValueError("AI returned an invalid date.")

    return {
        "dataset": dataset,
        "intent": intent,
        "period": str(plan.get("period") or "all").lower(),
        "start_date": start_date,
        "end_date": end_date,
        "status": str(plan.get("status") or "").title(),
        "product_name": str(plan.get("product_name") or "").strip(),
        "group_by": plan.get("group_by"),
        "top_n": top_n,
    }


# ============================================================
# FALLBACK QUERY UNDERSTANDING
# Used only if Sarvam chat is unavailable.
# ============================================================

def fallback_plan(question):
    q = question.casefold()

    plan = {
        "dataset": "payments",
        "intent": "unsupported",
        "period": "all",
        "start_date": None,
        "end_date": None,
        "status": "",
        "product_name": "",
        "group_by": None,
        "top_n": 5,
    }

    if any(x in q for x in ["expiry", "expire", "expiring", "best before"]):
        plan.update(dataset="inventory", intent="expiry")

    elif any(x in q for x in ["low stock", "running low", "stock alert"]):
        plan.update(dataset="inventory", intent="low_stock")

    elif any(x in q for x in ["reorder", "restock"]):
        plan.update(dataset="inventory", intent="reorder")

    elif any(x in q for x in ["inventory value", "stock value", "value of stock"]):
        plan.update(dataset="inventory", intent="inventory_value")

    elif any(x in q for x in ["list products", "all products", "show products"]):
        plan.update(dataset="inventory", intent="product_list")

    elif any(x in q for x in ["stock", "quantity", "how many", "remaining"]):
        plan.update(dataset="inventory", intent="product_stock")

    elif any(x in q for x in ["top customer", "best customer", "highest spending"]):
        plan.update(dataset="payments", intent="top_customers")

    elif any(x in q for x in ["payment source", "payment method", "upi", "card"]):
        plan.update(dataset="payments", intent="payment_sources")

    elif any(x in q for x in ["failed", "failure", "refunded", "refund"]):
        plan.update(dataset="payments", intent="payment_status")

    elif any(x in q for x in ["recent transaction", "latest payment", "recent payment"]):
        plan.update(dataset="payments", intent="recent_transactions")

    elif any(x in q for x in ["how many transactions", "transaction count"]):
        plan.update(dataset="payments", intent="transaction_count")

    elif any(x in q for x in ["sales", "revenue", "business", "earned", "income"]):
        plan.update(dataset="payments", intent="revenue")

    if "day after tomorrow" in q:
        plan["period"] = "day_after_tomorrow"
    elif "tomorrow" in q:
        plan["period"] = "tomorrow"
    elif "yesterday" in q:
        plan["period"] = "yesterday"
    elif "today" in q or "aaj" in q:
        plan["period"] = "today"
    elif "last hour" in q or "past hour" in q:
        plan["period"] = "last_hour"
    elif "this week" in q:
        plan["period"] = "this_week"
    elif "this month" in q:
        plan["period"] = "this_month"

    number_match = re.search(r"\btop\s+(\d+)\b", q)
    if number_match:
        plan["top_n"] = min(20, max(1, int(number_match.group(1))))

    if "failed" in q:
        plan["status"] = "Failed"
    elif "refunded" in q or "refund" in q:
        plan["status"] = "Refunded"

    return plan


def understand_question(question):
    try:
        return call_sarvam_chat(question), "Sarvam AI"
    except Exception as exc:
        st.warning(
            "AI query understanding was unavailable. "
            "Using basic fallback understanding instead."
        )
        st.caption(str(exc))
        return fallback_plan(question), "Fallback"


# ============================================================
# DATE FILTERING
# ============================================================

def get_date_range(plan):
    today = date.today()
    period = plan["period"]

    if plan["start_date"] and plan["end_date"]:
        return (
            date.fromisoformat(plan["start_date"]),
            date.fromisoformat(plan["end_date"]),
        )

    if plan["start_date"]:
        start = date.fromisoformat(plan["start_date"])
        end = date.fromisoformat(plan["end_date"]) if plan["end_date"] else start
        return start, end

    if period == "today":
        return today, today

    if period == "yesterday":
        yesterday = today - timedelta(days=1)
        return yesterday, yesterday

    if period == "tomorrow":
        tomorrow = today + timedelta(days=1)
        return tomorrow, tomorrow

    if period == "day_after_tomorrow":
        target = today + timedelta(days=2)
        return target, target

    if period.startswith("next_") and period.endswith("_days"):
        match = re.search(r"next_(\d+)_days", period)
        days = min(int(match.group(1)), 365) if match else 7
        return today, today + timedelta(days=days)

    if period == "this_week":
        start = today - timedelta(days=today.weekday())
        return start, today

    if period == "this_month":
        start = today.replace(day=1)
        return start, today

    return None, None


def filter_payments_by_period(df, plan):
    period = plan["period"]

    if period == "last_hour":
        now = datetime.now()
        start = now - timedelta(hours=1)

        return df[
            (df["Transaction Date & Time"] >= start)
            & (df["Transaction Date & Time"] <= now)
        ]

    start, end = get_date_range(plan)

    if start is None:
        return df

    return df[
        (df["Transaction Date & Time"].dt.date >= start)
        & (df["Transaction Date & Time"].dt.date <= end)
    ]


# ============================================================
# PAYMENT ANALYTICS
# ============================================================

def execute_payment_query(plan, payments):
    df = filter_payments_by_period(payments, plan)
    intent = plan["intent"]

    successful = df[
        df["Payment Status"].str.casefold() == "success"
    ]

    if intent == "revenue":
        amount = successful["Amount (INR)"].sum()

        return (
            f"Your successful payment revenue is ₹{amount:,.2f} "
            f"from {len(successful)} successful payments."
        )

    if intent == "transaction_count":
        return (
            f"There are {len(df)} transactions in the selected period, "
            f"including {len(successful)} successful payments."
        )

    if intent == "payment_status":
        status = plan["status"]

        if status:
            selected = df[
                df["Payment Status"].str.casefold() == status.casefold()
            ]
            return (
                f"You have {len(selected)} {status.lower()} transactions "
                f"totalling ₹{selected['Amount (INR)'].sum():,.2f}."
            )

        counts = df["Payment Status"].value_counts()

        return "Payment status summary: " + "; ".join(
            f"{name}: {count}"
            for name, count in counts.items()
        ) + "."

    if intent == "payment_sources":
        if successful.empty:
            return "There are no successful payments in this period."

        grouped = successful.groupby("Payment Source").agg(
            count=("Amount (INR)", "count"),
            total=("Amount (INR)", "sum"),
        ).sort_values("total", ascending=False)

        parts = [
            f"{name}: {int(row['count'])} payments, ₹{row['total']:,.2f}"
            for name, row in grouped.iterrows()
        ]

        return "Payment source summary: " + "; ".join(parts) + "."

    if intent == "top_customers":
        if successful.empty:
            return "There are no successful payments to analyse."

        top = (
            successful.groupby("Customer Name")["Amount (INR)"]
            .sum()
            .nlargest(plan["top_n"])
        )

        return "Your top customers are: " + "; ".join(
            f"{name}, ₹{amount:,.2f}"
            for name, amount in top.items()
        ) + "."

    if intent == "recent_transactions":
        recent = df.sort_values(
            "Transaction Date & Time", ascending=False
        ).head(plan["top_n"])

        if recent.empty:
            return "No transactions were found."

        items = []

        for _, row in recent.iterrows():
            items.append(
                f"{row['Customer Name']}, "
                f"₹{row['Amount (INR)']:,.2f}, "
                f"{row['Payment Status']}"
            )

        return "Your latest transactions are: " + "; ".join(items) + "."

    return None


# ============================================================
# INVENTORY ANALYTICS
# ============================================================

def match_product(inventory, product_name):
    if not product_name:
        return None, "Please tell me the product name."

    query = product_name.casefold().strip()

    names = inventory["Product Name"].str.casefold()
    ids = inventory["Product ID"].str.casefold()

    exact = inventory[
        (names == query) | (ids == query)
    ]

    if not exact.empty:
        return exact, None

    partial = inventory[
        names.str.contains(re.escape(query), na=False)
        | ids.str.contains(re.escape(query), na=False)
    ]

    if partial.empty:
        return None, f"I couldn't find {product_name} in your inventory."

    if len(partial) > 1:
        choices = ", ".join(partial["Product Name"].head(5).tolist())
        return None, f"I found multiple matches: {choices}. Please specify one."

    return partial, None


def execute_inventory_query(plan, inventory):
    intent = plan["intent"]

    if intent == "product_stock":
        matches, error = match_product(
            inventory, plan["product_name"]
        )

        if error:
            return error

        row = matches.iloc[0]
        qty = row["Quantity in Stock"]

        if pd.isna(qty):
            return f"Stock quantity for {row['Product Name']} is missing."

        return (
            f"You have {qty:g} {row['Unit']} "
            f"of {row['Product Name']} in stock."
        )

    if intent == "low_stock" or intent == "reorder":
        valid = inventory.dropna(
            subset=["Quantity in Stock", "Reorder Level"]
        )

        low = valid[
            valid["Quantity in Stock"] <= valid["Reorder Level"]
        ]

        if low.empty:
            return "Good news. No products are at or below their reorder level."

        items = [
            f"{r['Product Name']}: {r['Quantity in Stock']:g} "
            f"{r['Unit']} left, reorder level {r['Reorder Level']:g}"
            for _, r in low.iterrows()
        ]

        answer = (
            f"{len(low)} products need your attention: "
            + "; ".join(items[:10])
            + "."
        )

        if intent == "reorder":
            answer += (
                " I can identify which products need reordering, "
                "but your workbook does not contain target stock, "
                "sales demand, supplier lead time or pack size. "
                "So I cannot calculate a reliable order quantity."
            )

        return answer

    if intent == "expiry":
        valid = inventory.dropna(subset=["Expiry Date"]).copy()

        start, end = get_date_range(plan)

        if start is None:
            start = date.today()
            end = start + timedelta(days=30)

        expiring = valid[
            (valid["Expiry Date"] >= start)
            & (valid["Expiry Date"] <= end)
        ].sort_values("Expiry Date")

        if expiring.empty:
            return (
                f"No products with recorded expiry dates were found "
                f"between {start.strftime('%d %b %Y')} "
                f"and {end.strftime('%d %b %Y')}."
            )

        items = [
            f"{r['Product Name']} on "
            f"{r['Expiry Date'].strftime('%d %b %Y')}"
            for _, r in expiring.head(15).iterrows()
        ]

        return (
            f"{len(expiring)} products expire between "
            f"{start.strftime('%d %b')} and {end.strftime('%d %b')}: "
            + "; ".join(items)
            + "."
        )

    if intent == "inventory_value":
        valid = inventory.dropna(
            subset=["Quantity in Stock", "Purchase Price (₹)"]
        )

        value = (
            valid["Quantity in Stock"]
            * valid["Purchase Price (₹)"]
        ).sum()

        return (
            f"The estimated inventory purchase value is ₹{value:,.2f}. "
            "This is calculated using quantity in stock multiplied "
            "by purchase price."
        )

    if intent == "product_list":
        products = inventory["Product Name"].dropna().tolist()

        if not products:
            return "No products were found."

        return (
            f"You have {len(products)} products in your inventory: "
            + ", ".join(products[:20])
            + "."
        )

    return None


# ============================================================
# QUERY EXECUTION
# ============================================================

def answer_question(question, payments, inventory):
    if not question.strip():
        return "Please ask me a question."

    plan, engine = understand_question(question)

    st.session_state["last_query_plan"] = plan
    st.session_state["last_query_engine"] = engine

    intent = plan["intent"]
    dataset = plan["dataset"]

    if intent == "unsupported":
        return (
            "I couldn't understand that request. You can ask about "
            "sales, transactions, customers, payment methods, stock, "
            "low-stock products, expiry dates or inventory value."
        )

    if dataset == "payments":
        if payments is None:
            return "Please upload your payment transactions workbook first."

        result = execute_payment_query(plan, payments)

        if result:
            return result

    if dataset == "inventory":
        if inventory is None:
            return "Please upload your inventory workbook first."

        result = execute_inventory_query(plan, inventory)

        if result:
            return result

    return "I couldn't process that question. Please try asking it another way."


# ============================================================
# SARVAM SPEECH-TO-TEXT
# ============================================================

def transcribe_audio(audio_bytes, language_code, mode):
    if not API_KEY:
        raise RuntimeError("SARVAM_API_KEY is missing from .env.")

    response = requests.post(
        f"{BASE_URL}/speech-to-text",
        headers={"api-subscription-key": API_KEY},
        files={
            "file": (
                "merchant_question.wav",
                audio_bytes,
                "audio/wav",
            )
        },
        data={
            "model": STT_MODEL,
            "language_code": language_code,
            "mode": mode,
        },
        timeout=60,
    )

    if not response.ok:
        raise RuntimeError(
            f"Sarvam STT error ({response.status_code}): "
            f"{response.text[:500]}"
        )

    result = response.json()
    transcript = (result.get("transcript") or "").strip()

    if not transcript:
        raise ValueError("No speech was recognised. Please try again.")

    return transcript, result.get("language_code")


def translate_text(text, source_language_code, target_language_code):
    if not API_KEY:
        raise RuntimeError("SARVAM_API_KEY is missing from .env.")
    if source_language_code == target_language_code:
        return text, source_language_code

    max_chunk_length = 900 if source_language_code == "auto" else 1800
    remaining = text.strip()
    chunks = []
    while len(remaining) > max_chunk_length:
        split_at = max(
            remaining.rfind(" ", 0, max_chunk_length),
            remaining.rfind("\n", 0, max_chunk_length),
        )
        if split_at < max_chunk_length // 2:
            split_at = max_chunk_length
        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        chunks.append(remaining)

    translated_chunks = []
    resolved_source = source_language_code
    for chunk in chunks:
        model = (
            "mayura:v1"
            if resolved_source == "auto"
            else TRANSLATION_MODEL
        )
        response = requests.post(
            f"{BASE_URL}/translate",
            headers={
                "api-subscription-key": API_KEY,
                "Content-Type": "application/json",
            },
            json={
                "input": chunk,
                "source_language_code": resolved_source,
                "target_language_code": target_language_code,
                "model": model,
            },
            timeout=60,
        )

        if not response.ok:
            raise RuntimeError(
                f"Sarvam translation error ({response.status_code}): "
                f"{response.text[:500]}"
            )

        result = response.json()
        translated_chunk = (result.get("translated_text") or "").strip()
        if not translated_chunk:
            raise RuntimeError("Sarvam translation returned no text.")
        translated_chunks.append(translated_chunk)

        if resolved_source == "auto":
            resolved_source = result.get("source_language_code")
            if not resolved_source or resolved_source == "auto":
                raise RuntimeError("Sarvam could not detect the text language.")

    return " ".join(translated_chunks), resolved_source


# ============================================================
# SARVAM TEXT-TO-SPEECH
# ============================================================

def speak_answer(text, language_code="en-IN"):
    if not API_KEY:
        raise RuntimeError("SARVAM_API_KEY is missing.")

    tts_model = TTS_MODEL
    speaker = TTS_SPEAKER
    if TTS_MODEL == "bulbul:v4-flash" and language_code not in {
        "en-IN", "hi-IN"
    }:
        tts_model = "bulbul:v3"
        speaker = "shubh"
    elif TTS_MODEL == "bulbul:v3":
        speaker = "shubh"
    elif (
        TTS_MODEL == "bulbul:v4-flash"
        and language_code == "hi-IN"
        and "_hi_" not in speaker
        and "_enhi_" not in speaker
    ):
        speaker = "shubh_hi_customer"

    payload = {
        "text": text[:2400],
        "language_code": language_code,
        "speaker": speaker,
        "model": tts_model,
        "output_audio_codec": "mp3",
    }

    response = requests.post(
        f"{BASE_URL}/text-to-speech",
        headers={
            "api-subscription-key": API_KEY,
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )

    if not response.ok:
        raise RuntimeError(
            f"Sarvam TTS error ({response.status_code}): "
            f"{response.text[:500]}"
        )

    audios = response.json().get("audios") or []

    if not audios:
        raise RuntimeError("Sarvam TTS returned no audio.")

    return base64.b64decode("".join(audios))


# ============================================================
# STREAMLIT INTERFACE
# ============================================================

st.set_page_config(
    page_title="Dukaan Saathi",
    page_icon="🎙️",
    layout="centered",
)

st.markdown("""
<style>
.block-container {
    max-width: 780px;
    padding-top: 1.5rem;
}
.hero {
    background: linear-gradient(120deg, #102b46, #24577a);
    color: white;
    padding: 24px;
    border-radius: 18px;
    margin-bottom: 20px;
}
.hero h1 {
    margin: 0;
    font-size: 2rem;
}
.hero p {
    margin-bottom: 0;
    opacity: 0.85;
}
.stButton button {
    border-radius: 12px;
    min-height: 45px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🎙️ Dukaan Saathi</h1>
    <p>Your AI shop assistant. Speak naturally about your sales and stock.</p>
</div>
""", unsafe_allow_html=True)


# -------------------- SIDEBAR --------------------

with st.sidebar:
    st.header("Connect your shop")

    payment_file = st.file_uploader(
        "Replace payment transactions Excel (optional)",
        type=["xlsx"],
        key="payment_upload",
    )

    inventory_file = st.file_uploader(
        "Replace inventory Excel (optional)",
        type=["xlsx"],
        key="inventory_upload",
    )

    st.divider()

    language_choice = st.selectbox(
        "Your speaking language",
        [
            "Auto detect",
            "English",
            "Hindi",
            "Telugu",
            "Tamil",
            "Kannada",
        ],
    )

    language_codes = {
        "Auto detect": "unknown",
        "English": "en-IN",
        "Hindi": "hi-IN",
        "Telugu": "te-IN",
        "Tamil": "ta-IN",
        "Kannada": "kn-IN",
    }

    selected_language = language_codes[language_choice]

    mode_choice = st.selectbox(
        "Recognition mode",
        ["Transcribe", "Code-mixed"],
    )

    mode_codes = {
        "Transcribe": "transcribe",
        "Code-mixed": "codemix",
    }

    selected_mode = mode_codes[mode_choice]

    st.caption(
        "Bundled transaction and inventory workbooks load automatically. "
        "Upload a replacement only if needed."
    )


# -------------------- LOAD WORKBOOKS --------------------

backend_dir = Path(__file__).resolve().parent
payment_source = (
    payment_file
    if payment_file is not None
    else backend_dir / "Dukaan_Saathi_POS_Dummy_Transactions.xlsx"
)
inventory_source = (
    inventory_file
    if inventory_file is not None
    else backend_dir / "inventory.xlsx"
)

payments = None
inventory = None

try:
    payments = load_payment_workbook(payment_source)
    st.sidebar.success(
        f"Payments connected: {len(payments)} transactions"
    )
except Exception as exc:
    st.sidebar.error(f"Payment workbook error: {exc}")

try:
    inventory = load_inventory_workbook(inventory_source)
    st.sidebar.success(
        f"Inventory connected: {len(inventory)} products"
    )
except Exception as exc:
    st.sidebar.error(f"Inventory workbook error: {exc}")


# -------------------- CONNECTION STATUS --------------------

col1, col2 = st.columns(2)

with col1:
    if payments is not None:
        st.success("Payments connected")
    else:
        st.warning("Payments not connected")

with col2:
    if inventory is not None:
        st.success("Inventory connected")
    else:
        st.warning("Inventory not connected")


# -------------------- EXAMPLE QUESTIONS --------------------

with st.expander("Try asking Dukaan Saathi"):
    st.markdown("""
    **Sales and payments**
    - How much revenue did I make today?
    - Show my last hour sales.
    - Who are my top customers?
    - How many payments failed?
    - Which payment method is performing best?
    - Show my latest 5 transactions.

    **Inventory**
    - How much rice is left?
    - Which products are running low?
    - Which products expire tomorrow?
    - Which products expire in the next 7 days?
    - What should I reorder?
    - What is my total inventory value?
    """)


# -------------------- VOICE INPUT --------------------

st.markdown("### Ask your shop assistant")

st.write(
    "Tap the microphone, speak your question, and stop recording. "
    "Dukaan Saathi will understand it and respond."
)

audio = st.audio_input(
    "🎤 Tap to speak",
    sample_rate=16000,
)

if audio is not None:
    audio_bytes = audio.getvalue()
    audio_hash = hashlib.sha256(audio_bytes).hexdigest()

    if st.session_state.get("last_audio_hash") != audio_hash:
        st.session_state["last_audio_hash"] = audio_hash

        try:
            with st.spinner("Listening to your question..."):
                transcript, detected_language = transcribe_audio(
                    audio_bytes,
                    selected_language,
                    selected_mode,
                )

            st.markdown("#### You asked")
            st.info(transcript)

            if detected_language in (None, "", "unknown"):
                english_question, detected_language = translate_text(
                    transcript, "auto", "en-IN"
                )
            else:
                english_question, _ = translate_text(
                    transcript, detected_language, "en-IN"
                )

            with st.spinner("Dukaan Saathi is analysing your data..."):
                answer_english = answer_question(
                    english_question,
                    payments,
                    inventory,
                )

            answer = answer_english
            if detected_language != "en-IN":
                answer, _ = translate_text(
                    answer_english, "en-IN", detected_language
                )

            st.markdown("#### Dukaan Saathi")

            st.write(answer)

            with st.expander("See how your question was understood"):
                st.json(st.session_state.get("last_query_plan", {}))
                st.caption(
                    "Query engine: "
                    + st.session_state.get("last_query_engine", "Unknown")
                )

            if detected_language in TTS_SUPPORTED_LANGUAGES:
                with st.spinner("Preparing voice response..."):
                    speech = speak_answer(answer, detected_language)

                st.audio(
                    speech,
                    format="audio/mp3",
                    autoplay=True,
                )
            else:
                st.info(
                    "The answer is translated, but voice playback is not "
                    "available for this language."
                )

            st.session_state["last_answer"] = answer

        except Exception as exc:
            st.error(str(exc))

    else:
        st.caption(
            "This recording has already been processed. "
            "Record a new question to continue."
        )


# -------------------- TEXT QUESTION FALLBACK --------------------

st.divider()

st.markdown("### Or type your question")

typed_question = st.text_input(
    "Ask in English, Hindi, Telugu or mixed language",
    placeholder="Example: Which products expire tomorrow?",
    key="typed_question",
)

if st.button("Ask Dukaan Saathi", use_container_width=True):
    if typed_question.strip():
        with st.spinner("Understanding your question..."):
            english_question, detected_language = translate_text(
                typed_question, "auto", "en-IN"
            )
            answer_english = answer_question(
                english_question,
                payments,
                inventory,
            )

        answer = answer_english
        if detected_language != "en-IN":
            answer, _ = translate_text(
                answer_english, "en-IN", detected_language
            )

        st.markdown("#### Dukaan Saathi")
        st.write(answer)

        with st.expander("See query understanding"):
            st.json(st.session_state.get("last_query_plan", {}))

        if detected_language in TTS_SUPPORTED_LANGUAGES:
            try:
                with st.spinner("Preparing voice response..."):
                    speech = speak_answer(answer, detected_language)

                st.audio(speech, format="audio/mp3", autoplay=True)
            except Exception as exc:
                st.warning(
                    f"Text answer is ready, but voice output failed: {exc}"
                )
        else:
            st.info(
                "The answer is translated, but voice playback is not "
                "available for this language."
            )
    else:
        st.warning("Please type a question first.")


# -------------------- FOOTER --------------------

st.divider()

st.caption(
    "Dukaan Saathi · Powered by Sarvam AI · Excel data source"
)

st.caption(
    "Payment transactions do not contain product-level sales details. "
    "Inventory quantities are not automatically reduced by payment records."
)
