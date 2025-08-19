# api/index.py
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp")  # avoid MPL cache issues on HF

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.openapi.docs import get_swagger_ui_html
import pandas as pd
import io
import requests
import json
import re

from utils.analysis import fetch_wiki_top_films, auto_analyze

# ---- App instance ----
app = FastAPI(
    title="Data Analyst Agent API",
    description="Upload questions.txt (and optionally data.csv/.xlsx) to get analysis via LLM + pandas",
    version="3.3.0",
)

# ---- LLM config ----
AIPIPE_BASE = os.getenv("AIPIPE_BASE", "https://aipipe.org/openrouter/v1")
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")
AIPIPE_MODEL = os.getenv("AIPIPE_MODEL", "google/gemini-2.0-flash-lite-001")

# Debug check (don’t print actual token, just True/False)
print("DEBUG: AIPIPE_TOKEN loaded?", bool(AIPIPE_TOKEN))


# ----------------- Helpers -----------------
def _read_dataframe(upload: UploadFile | None) -> pd.DataFrame | None:
    if not upload:
        return None
    name = (upload.filename or "").lower()
    raw = upload.file.read()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(io.BytesIO(raw))
        if name.endswith(".xlsx") or name.endswith(".xls"):
            return pd.read_excel(io.BytesIO(raw))
        return pd.read_csv(io.BytesIO(raw))  # fallback
    except Exception:
        return None


def _df_preview(df: pd.DataFrame | None) -> str:
    if df is None or df.empty:
        return "No DataFrame provided."
    try:
        cols = ", ".join(map(str, df.columns.tolist()))
        head_csv = df.head(5).to_csv(index=False)
        return f"Columns: {cols}\n\nFirst 5 rows (CSV):\n{head_csv}"
    except Exception:
        return "DataFrame present but could not preview."


# ----------------- normalize_output function -----------------
def normalize_output(questions_text: str, preview: str, raw_result: dict) -> dict:
    prompt = f"""
    You are a strict JSON reformatter.

    Question:
    {questions_text}

    Data preview:
    {preview}

    Raw analysis result:
    {json.dumps(raw_result)}

    Task:
    - Return ONLY valid JSON with exactly the keys requested in the question.
    - If a chart is requested, map the closest one from raw_result["plots"].
    - Do not include explanations, just the JSON.
    """

    headers = {"Authorization": f"Bearer {AIPIPE_TOKEN}", "Content-Type": "application/json"}
    try:
        r = requests.post(
            f"{AIPIPE_BASE}/chat/completions",
            headers=headers,
            json={
                "model": AIPIPE_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 800,
            },
            timeout=30,
        )
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"].strip()

        # --- Clean wrappers like ```json ... ```
        if content.startswith("```"):
            content = re.sub(r"^```(json)?", "", content).strip("` \n")

        # --- Extract the first valid { ... } JSON block ---
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            snippet = match.group(0)
            # Truncate at the last closing brace to avoid trailing junk
            last_brace = snippet.rfind("}")
            snippet = snippet[: last_brace + 1]
            return json.loads(snippet)

        # Fallback: try raw parse
        return json.loads(content)

    except Exception as e:
        print("LLM normalization failed:", e)
        try:
            if 'r' in locals():
                print("LLM raw response (truncated):", r.text[:300])
        except Exception:
            pass
        return raw_result

# ----------------- Endpoints -----------------
@app.post("/analyze")
async def analyze_data(
    questions: UploadFile = File(...),
    data: UploadFile | None = File(None),
    mode: str = Form("eval"),              # eval (default) or play
    temperature: float = Form(0.7),
    top_p: float = Form(0.9),
    top_k: int = Form(40),
    presence_penalty: float = Form(0.0),
    frequency_penalty: float = Form(0.0),
    max_tokens: int = Form(200),
):
    # Read questions.txt
    try:
        questions_text = (await questions.read()).decode("utf-8", errors="ignore")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid questions.txt: {e}")

    # Optional dataset
    df = _read_dataframe(data) if data else None
    preview = _df_preview(df)

    # ---- Core pandas analysis ----
    try:
        raw_result = auto_analyze(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Auto analysis failed: {e}")

    # ---- Schema normalization via LLM ----
    final_result = normalize_output(questions_text, preview, raw_result)
    return JSONResponse(final_result)


@app.get("/wiki")
def get_wiki_films(n: int = 10):
    try:
        out = fetch_wiki_top_films(n)
        return JSONResponse(out)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch wiki films: {e}")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def go_to_docs():
    return RedirectResponse(url="/docs")


@app.get("/docs", include_in_schema=False)
def overridden_swagger():
    return get_swagger_ui_html(openapi_url="/openapi.json", title="Swagger UI")


# Entrypoint
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))  # HF injects PORT
    uvicorn.run("api.index:app", host="0.0.0.0", port=port)
