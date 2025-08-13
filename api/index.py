from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.openapi.docs import get_swagger_ui_html
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

# -------------------------------------------------------------------
# Optional health check
# -------------------------------------------------------------------
@app.get("/healthz")
def healthz():
    return {"status": "ok"}

# -------------------------------------------------------------------
# Redirect root to Swagger docs (static redirect)
# -------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def go_to_docs():
    return RedirectResponse(url="/docs")

# -------------------------------------------------------------------
# Force Swagger UI with absolute OpenAPI URL (optional tweak)
# -------------------------------------------------------------------
@app.get("/docs", include_in_schema=False)
def overridden_swagger():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Swagger UI"
    )

# -------------------------------------------------------------------
# Local dev entry point
# -------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("index:app", host="0.0.0.0", port=8000, reload=True)
