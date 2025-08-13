from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.openapi.docs import get_swagger_ui_html
from urllib.parse import urlsplit, urlunsplit
import pandas as pd
from io import BytesIO

# Your analysis logic
from utils.analysis import process_questions

# -------------------------------------------------------------------
# App
# -------------------------------------------------------------------
app = FastAPI(
    title="Data Analyst Agent API",
    description="Upload questions.txt (and optionally data.csv) to get analysis + plot",
    version="1.0.0",
)

# -------------------------------------------------------------------
# Core endpoint
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# Redirect root to Swagger docs
# -------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def go_to_docs(request: Request):
    parts = urlsplit(str(request.base_url))
    https_base = urlunsplit(("https", parts.netloc, "", "", ""))
    return RedirectResponse(url=https_base + "docs", status_code=307)

# -------------------------------------------------------------------
# Force Swagger UI with absolute OpenAPI URL (works in proxy/iframe)
# -------------------------------------------------------------------
@app.get("/docs", include_in_schema=False)
def overridden_swagger(request: Request):
    parts = urlsplit(str(request.base_url))
    https_base = urlunsplit(("https", parts.netloc, "", "", ""))
    openapi_abs = https_base + "openapi.json"
    return get_swagger_ui_html(openapi_url=openapi_abs, title="Swagger UI")

# -------------------------------------------------------------------
# Local dev entry point (ignored by Vercel)
# -------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
