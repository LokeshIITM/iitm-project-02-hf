from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from utils.analysis import process_questions  # Add this at the top if not present
import pandas as pd
from io import BytesIO
# import base64
# import matplotlib.pyplot as plt

app = FastAPI()


@app.post("/analyze")
async def analyze_data(questions: UploadFile = File(...), data: UploadFile = File(None)):
    questions_text = (await questions.read()).decode("utf-8")
    df = pd.read_csv(BytesIO(await data.read())) if data else None
    results = process_questions(questions_text, df)
    return JSONResponse(results)



##################--old########################
# @app.post("/analyze")
# async def analyze_data(questions: UploadFile = File(...), data: UploadFile = File(None)):
#     questions_content = await questions.read()
#     questions_text = questions_content.decode("utf-8")

#     if data:
#         if data.filename.endswith(".csv"):
#             df = pd.read_csv(BytesIO(await data.read()))
#         else:
#             return JSONResponse({"error": "Only CSV files are supported."})

    # Sample output
    # result = [1, "Titanic", 0.485782]

    # Sample plot
    fig, ax = plt.subplots()
    ax.scatter([1, 2, 3], [1, 4, 9])
    ax.plot([1, 2, 3], [1, 4, 9], 'r--')
    ax.set_title("Sample Plot")
    buf = BytesIO()
    fig.savefig(buf, format="png")
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()

    result.append(f"data:image/png;base64,{img_base64}")
    return JSONResponse(result)
