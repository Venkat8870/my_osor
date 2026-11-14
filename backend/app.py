
import os
import re
import json
import base64
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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
    "Transaction ID", "Customer ID", "Customer Name",
    "Amount (INR)", "Transaction Date & Time",
    "Payment Source", "Payment Status",
]

INVENTORY_COLUMNS = [
    "Product ID", "Product Name", "Category", "Unit",
    "Quantity in Stock", "Reorder Level",
    "Purchase Price (₹)", "Selling Price (₹)",
    "Supplier", "Expiry Date",
]

ALLOWED_DATASETS = {"payments", "inventory"}

ALLOWED_INTENTS = {
    "revenue", "transaction_count", "payment_status",
    "payment_sources", "top_customers", "recent_transactions",
    "product_stock", "low_stock", "expiry", "reorder",
    "inventory_value", "product_list", "unsupported",
}

# In-memory data for this demo.
# Uploaded files remain available while this backend process is running.
payments_df = None
inventory_df = None


# ============================================================
# FASTAPI APPLICATION
# ============================================================

@asynccontextmanager
async def lifespan(_app):
    initialize_sample_workbooks()
    yield


app = FastAPI(
    title="Dukaan Saathi API",
    description="Voice-first merchant intelligence assistant",
    version="1.0.0",
    lifespan=lifespan,
)

configured_frontend_origins = [
    origin.strip().rstrip("/")
    for origin in os.getenv("FRONTEND_ORIGINS", "").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(
        dict.fromkeys(
            [
                "http://localhost:3000",
                "http://127.0.0.1:3000",
                *configured_frontend_origins,
            ]
        )
    ),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODELS
# ============================================================

class AskRequest(BaseModel):
    question: str
    language: Optional[str] = None


# ============================================================
# WORKBOOK VALIDATION
# ============================================================

def load_payment_workbook(file_bytes):
    df = pd.read_excel(
        BytesIO(file_bytes),
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
        df["Payment Status"].fillna("")
        .astype(str).str.strip().str.title()
    )

    df["Payment Source"] = (
        df["Payment Source"].fillna("Unknown")
        .astype(str).str.strip()
    )

    df["Customer Name"] = (
        df["Customer Name"].fillna("Unknown")
        .astype(str).str.strip()
    )

    df = df.dropna(
        subset=["Amount (INR)", "Transaction Date & Time"]
    ).copy()

    if df.empty:
        raise ValueError("No valid payment transactions were found.")

    return df


def load_inventory_workbook(file_bytes):
    df = pd.read_excel(
        BytesIO(file_bytes),
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


def initialize_sample_workbooks():
    global payments_df, inventory_df

    backend_dir = Path(__file__).resolve().parent
    payments_path = backend_dir / "Dukaan_Saathi_POS_Dummy_Transactions.xlsx"
    inventory_path = backend_dir / "inventory.xlsx"

    missing_paths = [
        path.name for path in (payments_path, inventory_path)
        if not path.is_file()
    ]
    if missing_paths:
        raise RuntimeError(
            "Bundled workbook file(s) are missing: "
            + ", ".join(missing_paths)
        )

    loaded_payments = load_payment_workbook(payments_path.read_bytes())
    loaded_inventory = load_inventory_workbook(inventory_path.read_bytes())
    payments_df = loaded_payments
    inventory_df = loaded_inventory


# ============================================================
# CONNECTION STATUS
# ============================================================

@app.get("/")
def home():
    return {
        "name": "Dukaan Saathi",
        "status": "running",
        "message": "Your merchant intelligence assistant is ready.",
    }


@app.get("/status")
def connection_status():
    return {
        "payments_connected": payments_df is not None,
        "inventory_connected": inventory_df is not None,
        "payment_transactions": (
            len(payments_df) if payments_df is not None else 0
        ),
        "inventory_products": (
            len(inventory_df) if inventory_df is not None else 0
        ),
    }


# ============================================================
# CONNECT BUNDLED SAMPLE DATA
# ============================================================

@app.post("/connect/{dataset}/sample")
def connect_sample_data(dataset: str):
    global payments_df, inventory_df

    sample_workbooks = {
        "transactions": (
            Path(__file__).resolve().parent
            / "Dukaan_Saathi_POS_Dummy_Transactions.xlsx"
        ),
        "inventory": Path(__file__).resolve().parent / "inventory.xlsx",
    }

    if dataset not in sample_workbooks:
        raise HTTPException(
            status_code=404,
            detail="Unknown dataset. Choose transactions or inventory.",
        )

    workbook_path = sample_workbooks[dataset]

    try:
        if not workbook_path.is_file():
            raise HTTPException(
                status_code=500,
                detail=f"The bundled {dataset} sample workbook is missing.",
            )

        file_bytes = workbook_path.read_bytes()
        if dataset == "transactions":
            new_df = load_payment_workbook(file_bytes)
            payments_df = new_df
        else:
            new_df = load_inventory_workbook(file_bytes)
            inventory_df = new_df

        return {
            "success": True,
            "message": f"Sample {dataset} connected successfully.",
            "records": len(new_df),
            "status": "connected",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load bundled {dataset} sample data: {exc}",
        ) from exc


# ============================================================
# CONNECT TRANSACTIONS WORKBOOK
# ============================================================

@app.post("/connect/transactions")
async def connect_transactions(file: UploadFile = File(...)):
    global payments_df

    if not file.filename.lower().endswith(".xlsx"):
        raise HTTPException(
            status_code=400,
            detail="Please upload an .xlsx transactions workbook.",
        )

    try:
        file_bytes = await file.read()
        new_df = load_payment_workbook(file_bytes)

        # Replace the current data only after validation succeeds.
        payments_df = new_df

        return {
            "success": True,
            "message": "Transactions connected successfully.",
            "records": len(payments_df),
            "status": "connected",
        }

    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ============================================================
# CONNECT INVENTORY WORKBOOK
# ============================================================

@app.post("/connect/inventory")
async def connect_inventory(file: UploadFile = File(...)):
    global inventory_df

    if not file.filename.lower().endswith(".xlsx"):
        raise HTTPException(
            status_code=400,
            detail="Please upload an .xlsx inventory workbook.",
        )

    try:
        file_bytes = await file.read()
        new_df = load_inventory_workbook(file_bytes)

        inventory_df = new_df

        return {
            "success": True,
            "message": "Inventory connected successfully.",
            "records": len(inventory_df),
            "status": "connected",
        }

    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ============================================================
# SARVAM CHAT: NATURAL LANGUAGE -> STRUCTURED QUERY
# ============================================================

def validate_plan(plan):
    if not isinstance(plan, dict):
        raise ValueError("AI returned an invalid query plan.")

    dataset = str(plan.get("dataset", "")).lower()
    intent = str(plan.get("intent", "unsupported")).lower()

    if dataset not in ALLOWED_DATASETS:
        raise ValueError("AI selected an invalid dataset.")

    if intent not in ALLOWED_INTENTS:
        intent = "unsupported"

    payment_intents = {
        "revenue", "transaction_count", "payment_status",
        "payment_sources", "top_customers", "recent_transactions",
    }

    inventory_intents = {
        "product_stock", "low_stock", "expiry", "reorder",
        "inventory_value", "product_list",
    }

    if intent in payment_intents and dataset != "payments":
        intent = "unsupported"

    if intent in inventory_intents and dataset != "inventory":
        intent = "unsupported"

    try:
        top_n = int(plan.get("top_n") or 5)
    except (TypeError, ValueError):
        top_n = 5

    top_n = max(1, min(top_n, 20))

    start_date = plan.get("start_date")
    end_date = plan.get("end_date")

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


def call_sarvam_chat(question):
    if not API_KEY:
        raise RuntimeError("SARVAM_API_KEY is missing.")

    system_prompt = """
You are the query understanding engine for Dukaan Saathi,
an AI assistant for small shop owners in India.

Convert the merchant's question into ONE JSON object.
Do not answer the merchant.

Return these fields:
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

Allowed datasets: payments, inventory.

Allowed intents:
revenue, transaction_count, payment_status, payment_sources,
top_customers, recent_transactions, product_stock, low_stock,
expiry, reorder, inventory_value, product_list, unsupported.

Rules:
- Understand English, Hindi, Telugu and mixed-language questions.
- Sales, payments, customers and revenue use payments.
- Stock, products, reorder and expiry use inventory.
- "How much did I sell?" means revenue.
- "How many transactions?" means transaction_count.
- "Who are my best customers?" means top_customers.
- "Which payment methods are popular?" means payment_sources.
- "Show latest payments" means recent_transactions.
- "How much stock of rice?" means product_stock and product_name rice.
- "Which products are running low?" means low_stock.
- "Which products expire tomorrow?" means expiry and period tomorrow.
- "Which products expire in next 7 days?" means expiry and period next_7_days.
- "What should I reorder?" means reorder.
- "What is my inventory value?" means inventory_value.
- "Show all products" means product_list.
- Recognise today, yesterday, tomorrow, day_after_tomorrow,
  last_hour, this_week and this_month.
- Preserve the product name if the user gives one.
- Explicit date ranges should use YYYY-MM-DD.
- top_n must be between 1 and 20.
- Never invent a product name or date.
- If not supported, use unsupported.
- Return valid JSON only.
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
            f"Sarvam query understanding failed "
            f"({response.status_code}): {response.text[:500]}"
        )

    content = response.json()["choices"][0]["message"]["content"]
    return validate_plan(json.loads(content))


# ============================================================
# FALLBACK QUERY UNDERSTANDING
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

    elif any(x in q for x in ["stock", "quantity", "remaining"]):
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
    except Exception:
        return validate_plan(fallback_plan(question)), "Fallback"


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
        target = today - timedelta(days=1)
        return target, target

    if period == "tomorrow":
        target = today + timedelta(days=1)
        return target, target

    if period == "day_after_tomorrow":
        target = today + timedelta(days=2)
        return target, target

    if period.startswith("next_") and period.endswith("_days"):
        match = re.search(r"next_(\d+)_days", period)
        days = min(int(match.group(1)), 365) if match else 7
        return today, today + timedelta(days=days)

    if period.startswith("last_") and period.endswith("_days"):
        match = re.search(r"last_(\d+)_days", period)
        days = min(int(match.group(1)), 365) if match else 7
        return today - timedelta(days=days - 1), today

    if period == "this_week":
        return today - timedelta(days=today.weekday()), today

    if period == "this_month":
        return today.replace(day=1), today

    return None, None


def filter_payments_by_period(df, plan):
    if plan["period"] == "last_hour":
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
        if counts.empty:
            return "No payment records were found in this period."

        return "Payment status summary: " + "; ".join(
            f"{name}: {count}" for name, count in counts.items()
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
            .sum().nlargest(plan["top_n"])
        )

        return "Your top customers are: " + "; ".join(
            f"{name}, ₹{amount:,.2f}" for name, amount in top.items()
        ) + "."

    if intent == "recent_transactions":
        recent = df.sort_values(
            "Transaction Date & Time", ascending=False
        ).head(plan["top_n"])

        if recent.empty:
            return "No transactions were found."

        items = [
            f"{r['Customer Name']}, ₹{r['Amount (INR)']:,.2f}, "
            f"{r['Payment Status']}"
            for _, r in recent.iterrows()
        ]
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

    exact = inventory[(names == query) | (ids == query)]
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


def infer_product_reference(question, inventory):
    question_words = set(re.findall(r"[a-z0-9]+", question.casefold()))
    stop_words = {
        "how", "much", "many", "is", "are", "the", "a", "an", "in",
        "on", "of", "left", "remaining", "remain", "stock", "have",
        "got", "there", "any", "please", "tell", "me", "what",
        "quantity", "product", "products", "item", "items", "available",
        "currently", "my", "do", "we", "you",
    }
    question_words.difference_update(stop_words)
    if not question_words:
        return ""

    product_names = inventory["Product Name"].fillna("").astype(str)
    normalized_question = " ".join(
        re.findall(r"[a-z0-9]+", question.casefold())
    )
    for product_name in product_names:
        normalized_name = " ".join(
            re.findall(r"[a-z0-9]+", product_name.casefold())
        )
        if normalized_name and normalized_name in normalized_question:
            return product_name

    matches = []
    for product_name in product_names:
        name_words = set(
            re.findall(r"[a-z0-9]+", product_name.casefold())
        )
        overlap = question_words & name_words
        if overlap:
            matches.append((product_name, overlap))

    if not matches:
        return ""

    highest_score = max(len(overlap) for _, overlap in matches)
    best_matches = [
        (name, overlap)
        for name, overlap in matches
        if len(overlap) == highest_score
    ]
    if len(best_matches) == 1:
        return best_matches[0][0]

    shared_terms = set.intersection(
        *(overlap for _, overlap in best_matches)
    )
    if shared_terms:
        return max(
            shared_terms,
            key=lambda word: sum(word in overlap for _, overlap in matches),
        )

    return ""


def execute_inventory_query(plan, inventory):
    intent = plan["intent"]

    if intent == "product_stock":
        matches, error = match_product(inventory, plan["product_name"])
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

    if intent in {"low_stock", "reorder"}:
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
            + "; ".join(items[:10]) + "."
        )

        if intent == "reorder":
            answer += (
                " I can identify products that need reordering, "
                "but the workbook does not contain target stock, "
                "sales demand, supplier lead time or pack size. "
                "I cannot calculate a reliable order quantity."
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
                f"between {start:%d %b %Y} and {end:%d %b %Y}."
            )

        items = [
            f"{r['Product Name']} on {r['Expiry Date']:%d %b %Y}"
            for _, r in expiring.head(15).iterrows()
        ]

        return (
            f"{len(expiring)} products expire between "
            f"{start:%d %b} and {end:%d %b}: "
            + "; ".join(items) + "."
        )

    if intent == "inventory_value":
        valid = inventory.dropna(
            subset=["Quantity in Stock", "Purchase Price (₹)"]
        )
        value = (
            valid["Quantity in Stock"] * valid["Purchase Price (₹)"]
        ).sum()

        return (
            f"The estimated inventory purchase value is ₹{value:,.2f}. "
            "This uses quantity in stock multiplied by purchase price."
        )

    if intent == "product_list":
        products = inventory["Product Name"].dropna().tolist()
        if not products:
            return "No products were found."

        return (
            f"You have {len(products)} products in your inventory: "
            + ", ".join(products[:20]) + "."
        )

    return None


# ============================================================
# QUESTION ANSWERING
# ============================================================

def answer_question(question):
    if not question.strip():
        return "Please ask me a question.", {}

    plan, engine = understand_question(question)
    if (
        plan["dataset"] == "inventory"
        and plan["intent"] == "product_stock"
        and not plan["product_name"]
        and inventory_df is not None
    ):
        plan["product_name"] = infer_product_reference(
            question, inventory_df
        )

    if plan["intent"] == "unsupported":
        return (
            "I couldn't understand that request. Ask me about sales, "
            "transactions, customers, payment methods, stock, "
            "low-stock products, expiry dates or inventory value."
        ), {"plan": plan, "engine": engine}

    if plan["dataset"] == "payments":
        if payments_df is None:
            return (
                "Please connect your transactions workbook first.",
                {"plan": plan, "engine": engine},
            )

        answer = execute_payment_query(plan, payments_df)

    else:
        if inventory_df is None:
            return (
                "Please connect your inventory workbook first.",
                {"plan": plan, "engine": engine},
            )

        answer = execute_inventory_query(plan, inventory_df)

    return (
        answer or "I couldn't process that question. Please try again.",
        {"plan": plan, "engine": engine},
    )


# ============================================================
# SARVAM SPEECH-TO-TEXT
# ============================================================

def transcribe_audio(audio_bytes, language_code=None, mode=None):
    if not API_KEY:
        raise RuntimeError("SARVAM_API_KEY is missing.")

    response = requests.post(
        f"{BASE_URL}/speech-to-text",
        headers={"api-subscription-key": API_KEY},
        files={
            "file": ("merchant_question.wav", audio_bytes, "audio/wav")
        },
        data={
            "model": STT_MODEL,
            "language_code": language_code or STT_LANGUAGE,
            "mode": mode or STT_MODE,
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
        raise RuntimeError("SARVAM_API_KEY is missing.")
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

def speak_answer(text, language_code=None):
    if not API_KEY:
        raise RuntimeError("SARVAM_API_KEY is missing.")

    target_language = language_code or TTS_LANGUAGE
    tts_model = TTS_MODEL
    speaker = TTS_SPEAKER

    if TTS_MODEL == "bulbul:v4-flash" and target_language not in {
        "en-IN", "hi-IN"
    }:
        tts_model = "bulbul:v3"
        speaker = "shubh"
    elif TTS_MODEL == "bulbul:v3":
        speaker = "shubh"
    elif (
        TTS_MODEL == "bulbul:v4-flash"
        and target_language == "hi-IN"
        and "_hi_" not in speaker
        and "_enhi_" not in speaker
    ):
        speaker = "shubh_hi_customer"

    payload = {
        "text": text[:2400],
        "language_code": target_language,
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
# TEXT QUESTION ENDPOINT
# ============================================================

@app.post("/ask")
def ask(request: AskRequest):
    try:
        if not request.question.strip():
            return {
                "success": True,
                "question": request.question,
                "answer": "Please ask me a question.",
                "detected_language": request.language,
                "metadata": {},
            }

        input_language = request.language or "auto"
        if input_language in {"unknown", "auto"}:
            english_question, input_language = translate_text(
                request.question, "auto", "en-IN"
            )
        else:
            english_question, _ = translate_text(
                request.question, input_language, "en-IN"
            )

        answer_english, metadata = answer_question(english_question)
        answer = answer_english
        if input_language != "en-IN":
            answer, _ = translate_text(
                answer_english, "en-IN", input_language
            )

        return {
            "success": True,
            "question": request.question,
            "translated_question": english_question,
            "answer": answer,
            "detected_language": input_language,
            "metadata": metadata,
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ============================================================
# VOICE QUESTION ENDPOINT
# ============================================================

@app.post("/ask-voice")
async def ask_voice(
    file: UploadFile = File(...),
    language: str = STT_LANGUAGE,
    mode: str = STT_MODE,
):
    try:
        audio_bytes = await file.read()

        if not audio_bytes:
            raise HTTPException(
                status_code=400,
                detail="The audio recording is empty.",
            )

        transcript, detected_language = transcribe_audio(
            audio_bytes,
            language,
            mode,
        )

        if detected_language in (None, "", "unknown"):
            english_question, detected_language = translate_text(
                transcript, "auto", "en-IN"
            )
        else:
            english_question, _ = translate_text(
                transcript, detected_language, "en-IN"
            )

        answer_english, metadata = answer_question(english_question)
        answer = answer_english
        if detected_language != "en-IN":
            answer, _ = translate_text(
                answer_english, "en-IN", detected_language
            )

        audio_error = None
        speech = None
        if detected_language in TTS_SUPPORTED_LANGUAGES:
            speech = speak_answer(answer, detected_language)
        else:
            audio_error = (
                "The answer is translated, but voice playback is not "
                "available for this language."
            )

        return {
            "success": True,
            "transcript": transcript,
            "translated_question": english_question,
            "answer": answer,
            "audio_base64": (
                base64.b64encode(speech).decode("utf-8")
                if speech is not None else None
            ),
            "audio_content_type": "audio/mpeg" if speech is not None else None,
            "audio_error": audio_error,
            "detected_language": detected_language,
            "metadata": metadata,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
