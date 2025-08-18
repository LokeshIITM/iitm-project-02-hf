# api/index.py
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp")  # avoid MPL cache issues on HF

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.openapi.docs import get_swagger_ui_html
import pandas as pd
import io
import re
import requests
import json

from utils.analysis import fetch_wiki_top_films

# ---- App instance ----
app = FastAPI(
    title="Data Analyst Agent API",
    description="Upload questions.txt (and optionally data.csv/.xlsx) to get analysis via LLM",
    version="2.0.0",
)

# ---- LLM config ----
AIPIPE_BASE = os.getenv("AIPIPE_BASE", "https://aipipe.org/openrouter/v1")
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")
AIPIPE_MODEL = os.getenv("AIPIPE_MODEL", "google/gemini-2.0-flash-lite-001")


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
    max_tokens: int = Form(800),
    stop: str | None = Form(None),         # optional stop sequence
):
    # Read questions.txt
    try:
        questions_text = (await questions.read()).decode("utf-8", errors="ignore")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid questions.txt: {e}")

    # Optional dataset preview
    df = _read_dataframe(data) if data else None
    preview = _df_preview(df)

    # Prompt for LLM
    user_prompt = f"questions.txt:\n{questions_text}\n\nData preview:\n{preview}"

    payload = {
        "model": AIPIPE_MODEL,
        "messages": [
            {"role": "system", "content": "You are a data analyst agent. Return JSON only with the required keys."},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
    }

    if mode == "eval":
        # strict deterministic config
        payload.update({
            "temperature": 0,
            "top_p": 1,
            "top_k": 1,
            "stop": ["}"],   # cut off after JSON
        })
    else:
        # play mode with knobs
        payload.update({
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
        })
        if stop:
            payload["stop"] = [stop]

    # Call LLM
    url = f"{AIPIPE_BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {AIPIPE_TOKEN}", "Content-Type": "application/json"}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        llm_output = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM request failed: {e}")

    # Ensure valid JSON
    try:
        return JSONResponse(json.loads(llm_output))
    except Exception:
        return JSONResponse({"error": "Invalid JSON from LLM", "raw": llm_output})


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
