import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO

def process_questions(questions_text: str, df: pd.DataFrame) -> list:
    questions_text = questions_text.lower()
    results = []
    image_base64 = ""

    if "average age" in questions_text and "age" in df.columns:
        avg = round(df["age"].mean(), 2)
        results.append(f"The average age of passengers is {avg}.")

    if "plot age vs fare" in questions_text and {"age", "fare"}.issubset(df.columns):
        corr = round(df["age"].corr(df["fare"]), 6)
        results.append(f"The correlation between age and fare is {corr}.")

        fig, ax = plt.subplots()
        x = df["age"]
        y = df["fare"]
        ax.scatter(x, y)
        m, b = pd.Series(y).corr(x), 0
        ax.plot(x, m * x + b, 'r--')
        ax.set_xlabel("Age")
        ax.set_ylabel("Fare")
        ax.set_title("Age vs Fare")

        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)

        results.append(f"data:image/png;base64,{image_base64}")

    return results
