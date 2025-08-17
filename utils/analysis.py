# utils/analysis.py
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp")  # avoid MPL cache warning in read-only envs

import base64
from io import BytesIO
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _figure_to_base64(fig) -> str:
    """Encode a Matplotlib figure to a base64 PNG (kept compact)."""
    buf = BytesIO()
    plt.tight_layout()
    # Slightly lower DPI + bbox_inches to help keep under ~100 KB
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{b64}"


def fetch_wiki_top_films(n: int = 10) -> List[Dict]:
    """
    Fetch the 'Highest-grossing films' table from Wikipedia and return the
    first n rows as a list of dicts (JSON-serializable).
    """
    url = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
    tables = pd.read_html(url)  # requires lxml
    # Find a table that has Rank and Title columns (case-insensitive)
    chosen = None
    for t in tables:
        cols = {str(c).strip().lower() for c in t.columns}
        if {"rank", "title"}.issubset(cols):
            chosen = t
            break
    if chosen is None:
        chosen = tables[0]

    # Standardize column names a bit (optional)
    chosen.columns = [str(c).strip() for c in chosen.columns]

    # Keep top n rows
    n = max(1, int(n))
    out = chosen.head(n).to_dict(orient="records")
    return out


def process_questions(questions_text: str, df: Optional[pd.DataFrame]) -> list:
    """
    Handle simple analysis queries against an optional DataFrame.
    Returns a list with strings/numbers/base64 image strings.
      - If df is None -> ["No data provided."]
      - If 'average age' in questions and 'age' column exists -> append avg string
      - If 'plot age vs fare' and both columns exist -> append correlation + base64 PNG
    """
    questions_text = (questions_text or "").lower()
    results: list = []

    if df is None or not isinstance(df, pd.DataFrame):
        results.append("No data provided.")
        return results

    # 1) Average age
    if "average age" in questions_text and "age" in df.columns:
        try:
            avg = float(round(df["age"].astype(float).mean(), 2))
            results.append(f"The average age of passengers is {avg}.")
        except Exception:
            results.append("Could not compute average age due to invalid data types.")

    # 2) Scatter + red dotted regression line for Age vs Fare
    if "plot age vs fare" in questions_text and {"age", "fare"}.issubset(df.columns):
        sub = df[["age", "fare"]].dropna()
        if not sub.empty:
            try:
                x = sub["age"].astype(float).to_numpy()
                y = sub["fare"].astype(float).to_numpy()

                # Pearson correlation
                corr = float(pd.Series(x).corr(pd.Series(y)))
                results.append(f"The correlation between age and fare is {corr:.6f}.")

                # Regression line (least squares)
                m, b = np.polyfit(x, y, 1)

                # Plot
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.scatter(x, y, s=10)                   # points
                xs = np.sort(x)  # ensure monotonic x for a clean line
                # ax.plot(x, m * x + b, "r--", linewidth=2)  # red dotted regression
                ax.plot(xs, m * xs + b, linestyle=':', linewidth=2, color='red')  # red dotted regression
                ax.set_xlabel("Age")
                ax.set_ylabel("Fare")
                ax.set_title("Age vs Fare")

                # Encode figure
                image_b64 = _figure_to_base64(fig)
                results.append(image_b64)
            except Exception:
                results.append("Failed to generate Age vs Fare plot due to bad data.")
        else:
            results.append("Not enough data to plot age vs fare (after dropping NaNs).")

    return results
