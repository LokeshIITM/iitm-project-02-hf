from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse
from fastapi.openapi.docs import get_swagger_ui_html
from urllib.parse import urlsplit, urlunsplit
import pandas as pd
from io import BytesIO

# Your analysis logic
from utils.analysis import process_questions

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Data Analyst Agent API",
    description="Upload questions.txt (and optionally data.csv) to get analysis + plot",
    version="1.0.0",
)

# -----------------------------------------------------------------------------
# Core endpoint
# -----------------------------------------------------------------------------
@app.post("/analyze")
async def analyze_data(
    questions: UploadFile = File(...),
    data: UploadFile = File(None)
):
    # Read questions.txt
    questions_text = (await questions.read()).decode("utf-8")

    # Read data.csv if provided
    df = pd.read_csv(BytesIO(await data.read())) if data else None

    # Call your analysis function (returns JSON-serializable result)
    results = process_questions(questions_text, df)
    return JSONResponse(results)

# Optional health check
@app.get("/healthz")
def healthz():
    return {"status": "ok"}

# -----------------------------------------------------------------------------
# Make the App button open Swagger directly (no blue button HTML)
# Use a hard redirect so it works in the HF preview iframe.
# -----------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def go_to_docs(request: Request):
    parts = urlsplit(str(request.base_url))
    https_base = urlunsplit(("https", parts.netloc, "", "", ""))
    return RedirectResponse(url=https_base + "docs", status_code=307)

# -----------------------------------------------------------------------------
# Force Swagger UI to render behind HF proxy with absolute HTTPS OpenAPI URL
# -----------------------------------------------------------------------------
@app.get("/docs", include_in_schema=False)
def overridden_swagger(request: Request):
    parts = urlsplit(str(request.base_url))
    https_base = urlunsplit(("https", parts.netloc, "", "", ""))
    openapi_abs = https_base + "openapi.json"
    # Return Swagger UI with absolute openapi URL (works in iframe/proxy)
    return get_swagger_ui_html(openapi_url=openapi_abs, title="Swagger UI")

# Keep the default OpenAPI generator at /openapi.json (FastAPI provides it)