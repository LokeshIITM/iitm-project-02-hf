# utils/analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import numpy as np

def process_questions(questions_text: str, df: pd.DataFrame) -> list:
    questions_text = (questions_text or "").lower()
    results = []
    image_base64 = ""

    if df is None or not isinstance(df, pd.DataFrame):
        results.append("No data provided.")
        return results

    # 1) Average age
    if "average age" in questions_text and "age" in df.columns:
        avg = round(df["age"].mean(), 2)
        results.append(f"The average age of passengers is {avg}.")

    # 2) Scatter + red dotted regression line for Age vs Fare
    if "plot age vs fare" in questions_text and {"age", "fare"}.issubset(df.columns):
        # Clean rows
        sub = df[["age", "fare"]].dropna()
        if not sub.empty:
            x = sub["age"].astype(float).values
            y = sub["fare"].astype(float).values

            # Pearson correlation
            corr = float(pd.Series(x).corr(pd.Series(y)))
            results.append(f"The correlation between age and fare is {corr:.6f}.")

            # Regression line via least squares
            m, b = np.polyfit(x, y, 1)

            # Plot
            fig, ax = plt.subplots()
            ax.scatter(x, y)
            ax.plot(x, m * x + b, "r--", linewidth=2)  # red dotted regression
            ax.set_xlabel("Age")
            ax.set_ylabel("Fare")
            ax.set_title("Age vs Fare")

            # Tight PNG under ~100 KB (adjust dpi if needed)
            buf = BytesIO()
            plt.tight_layout()
            fig.savefig(buf, format="png", dpi=100)
            buf.seek(0)
            image_base64 = base64.b64encode(buf.read()).decode("utf-8")
            plt.close(fig)

            results.append(f"data:image/png;base64,{image_base64}")
        else:
            results.append("Not enough data to plot age vs fare (after dropping NaNs).")

    return results
