# api/index.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.openapi.docs import get_swagger_ui_html
import pandas as pd
from io import BytesIO

from utils.analysis import process_questions, fetch_wiki_top_films

app = FastAPI(
    title="Data Analyst Agent API",
    description="Upload questions.txt (and optionally data.csv) to get analysis + plot",
    version="1.0.0",
)

# -------------------------------------------------------------------
# Wikipedia endpoint
# -------------------------------------------------------------------
@app.get("/wiki")
def get_wiki_films(n: int = 10):
    try:
        out = fetch_wiki_top_films(n)  # returns list[dict]
        return JSONResponse(out)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch wiki films: {e}")

# -------------------------------------------------------------------
# Core endpoint (file-upload analysis)
# -------------------------------------------------------------------
@app.post("/analyze")
async def analyze_data(
    questions: UploadFile = File(...),
    data: UploadFile = File(None)
):
    # Read questions.txt
    try:
        questions_text = (await questions.read()).decode("utf-8", errors="ignore")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid questions.txt: {e}")

    # Read data.csv if provided
    df = None
    if data:
        try:
            csv_bytes = await data.read()
            df = pd.read_csv(BytesIO(csv_bytes))
        except Exception:
            # Proceed without data if CSV is malformed
            df = None

    # Run analysis (returns list with text/numbers/base64 image)
    try:
        results = process_questions(questions_text, df)
        return JSONResponse(results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

# -------------------------------------------------------------------
# Health + Docs
# -------------------------------------------------------------------
@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/", include_in_schema=False)
def go_to_docs():
    return RedirectResponse(url="/docs")

@app.get("/docs", include_in_schema=False)
def overridden_swagger():
    return get_swagger_ui_html(openapi_url="/openapi.json", title="Swagger UI")

if __name__ == "__main__":
    import uvicorn
    # Use the full module path for local dev if this file lives in api/index.py
    uvicorn.run("api.index:app", host="0.0.0.0", port=8000, reload=True)
