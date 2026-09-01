import io
import json
import re
import sqlite3

import pandas as pd
import plotly.express as px
import plotly.io as pio
import ollama
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Local NLP to Graph Insights Agent")

# Allow the JS frontend (served from the same app, or a separate dev server)
# to call this API. Tighten allow_origins to a specific origin in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

LOCAL_MODEL = "llama3.2:3b"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB

# Chart types the agent (or the user, via manual override) can produce.
# "auto" means: let the LLM decide.
SUPPORTED_GRAPH_TYPES = [
    "bar", "line", "pie", "donut", "scatter", "area", "histogram", "box"
]


def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Replaces spaces/special characters in column names so LLM-generated SQL
    doesn't break on unquoted identifiers like `Asset - Cost Price ($)`.
    Collapses repeated underscores, trims leading/trailing ones, and
    de-duplicates any resulting collisions."""
    seen = {}
    clean_cols = []
    for col in df.columns:
        c = re.sub(r"[^0-9a-zA-Z_]", "_", str(col))
        c = re.sub(r"_+", "_", c).strip("_") or "column"
        if c in seen:
            seen[c] += 1
            c = f"{c}_{seen[c]}"
        else:
            seen[c] = 0
        clean_cols.append(c)
    df.columns = clean_cols
    return df


def clean_currency_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Detects columns that look like currency (e.g. "$500.00", "$1,000.00")
    and converts them to real numeric values, stripping the $ and thousands
    separators. Without this, SQLite treats them as text and SUM/AVG/ORDER BY
    silently produce wrong (lexicographic) results instead of an error.

    Works across pandas versions: pandas 3.x defaults string columns to
    StringDtype rather than classic `object`, so we check with
    pd.api.types.is_string_dtype() instead of `dtype == object`."""
    currency_pattern = re.compile(r"^\s*\$\s*-?[\d,]*\.?\d*\s*$")

    for col in df.columns:
        if not (pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])):
            continue

        sample = df[col].dropna().astype(str).head(20)
        if sample.empty:
            continue

        looks_like_currency = sample.apply(lambda v: bool(currency_pattern.match(v))).mean() > 0.6
        if not looks_like_currency:
            continue

        cleaned = (
            df[col]
            .astype(str)
            .str.replace(r"[\$,]", "", regex=True)
            .str.strip()
        )
        df[col] = pd.to_numeric(cleaned, errors="coerce")

    return df


def get_llm_sql_and_graph_config(schema: str, query: str, forced_graph_type: str | None) -> dict:
    """Uses a local LLM to generate the SQL query and recommend graph axes.
    If forced_graph_type is set, the LLM is told to use that type instead of choosing."""
    graph_instruction = (
        f'Use exactly graph_type "{forced_graph_type}".'
        if forced_graph_type and forced_graph_type != "auto"
        else f"Suggest the best graph type from this list: {', '.join(SUPPORTED_GRAPH_TYPES)}."
    )

    prompt = f"""
You are a data analyst. You have a table named `report` with this schema:
{schema}

The user asked: "{query}"

1. Write a valid SQL SELECT query to answer this request. Only ever write a SELECT statement — never modify, delete, or alter data. Reference only the exact column names given above.
2. {graph_instruction} Also provide x_axis and y_axis column names from the query result. For "histogram", y_axis may be left as an empty string. For "pie" or "donut", x_axis is the category (names) column and y_axis is the value column.

Return ONLY a JSON object. Use exactly these keys: "sql", "graph_type", "x_axis", "y_axis".
"""
    response = ollama.chat(
        model=LOCAL_MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json"
    )
    try:
        config = json.loads(response['message']['content'].strip())
    except Exception:
        raise ValueError(f"Failed to parse LLM response: {response['message']['content']}")

    if forced_graph_type and forced_graph_type != "auto":
        config["graph_type"] = forced_graph_type

    return config


def generate_insights(data_json: list, query: str) -> str:
    """Uses a local LLM to generate business insights based on the queried data."""
    prompt = f"""
The user asked: "{query}"

The database returned the following data:
{json.dumps(data_json)}

Provide 2-3 sentences of sharp, professional insights based on this data. Do not include any code or JSON in your response.
"""
    response = ollama.chat(
        model=LOCAL_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content'].strip()


def build_figure(result_df: pd.DataFrame, graph_type: str, x_col: str, y_col: str):
    """Builds a Plotly figure for any of the supported graph types.
    Returns None if the requested columns/type combination isn't viable."""
    cols = set(result_df.columns)

    if graph_type in ("histogram",):
        if x_col not in cols:
            return None
        return px.histogram(result_df, x=x_col)

    if graph_type == "box":
        if y_col in cols and x_col in cols:
            return px.box(result_df, x=x_col, y=y_col)
        if y_col in cols:
            return px.box(result_df, y=y_col)
        if x_col in cols:
            return px.box(result_df, y=x_col)
        return None

    if graph_type in ("pie", "donut"):
        if x_col not in cols or y_col not in cols:
            return None
        hole = 0.5 if graph_type == "donut" else 0.0
        return px.pie(result_df, names=x_col, values=y_col, hole=hole)

    # bar, line, scatter, area all need both an x and y column
    if x_col not in cols or y_col not in cols:
        return None

    if graph_type == "line":
        return px.line(result_df, x=x_col, y=y_col)
    if graph_type == "scatter":
        return px.scatter(result_df, x=x_col, y=y_col)
    if graph_type == "area":
        return px.area(result_df, x=x_col, y=y_col)

    # default fallback
    return px.bar(result_df, x=x_col, y=y_col)


@app.post("/analyze")
async def analyze_report(
    file: UploadFile,
    query: str = Form(...),
    graph_type: str = Form("auto"),
):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    if graph_type not in SUPPORTED_GRAPH_TYPES + ["auto"]:
        raise HTTPException(status_code=400, detail=f"Unsupported graph_type. Choose from: auto, {', '.join(SUPPORTED_GRAPH_TYPES)}")

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="CSV file is too large.")

    try:
        df = pd.read_csv(io.BytesIO(contents))
        df = sanitize_columns(df)
        df = clean_currency_columns(df)

        schema = df.dtypes.to_dict()
        schema_str = ", ".join([f"{col} ({dtype})" for col, dtype in schema.items()])

        llm_config = get_llm_sql_and_graph_config(schema_str, query, graph_type)
        sql_query = (llm_config.get("sql") or "").strip()

        if not sql_query:
            raise HTTPException(status_code=502, detail="LLM did not return a valid SQL query.")
        if not sql_query.lower().lstrip().startswith("select"):
            raise HTTPException(status_code=400, detail="Generated query was not a SELECT statement.")

        conn = sqlite3.connect(':memory:')
        try:
            df.to_sql('report', conn, index=False)
            result_df = pd.read_sql(sql_query, conn)
        finally:
            conn.close()

        data_json = result_df.to_dict(orient="records")

        chosen_type = (llm_config.get("graph_type") or "bar").lower()
        if chosen_type not in SUPPORTED_GRAPH_TYPES:
            chosen_type = "bar"

        x_col = llm_config.get("x_axis") or ""
        y_col = llm_config.get("y_axis") or ""

        fig = build_figure(result_df, chosen_type, x_col, y_col) if not result_df.empty else None
        graph_json = json.loads(pio.to_json(fig)) if fig else None

        insights = generate_insights(data_json, query)

        return {
            "query_used": sql_query,
            "graph_type_used": chosen_type if fig else None,
            "data": data_json,
            "insights": insights,
            "graph": graph_json
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Serve the JS frontend. Keep this mount AFTER the /analyze route above —
# Starlette matches routes in the order they were added, so /analyze still
# takes priority over the static file catch-all.
app.mount("/", StaticFiles(directory="static", html=True), name="static")