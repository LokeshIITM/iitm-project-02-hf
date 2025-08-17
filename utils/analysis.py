# utils/analysis.py
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp")  # avoid MPL cache warning in read-only envs

import base64
from io import BytesIO
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Plot utility
# ----------------------------
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


# ----------------------------
# Simple Wikipedia fetch (for /wiki endpoint)
# ----------------------------
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
    if chosen is None and tables:
        chosen = tables[0]

    # Standardize column names a bit (optional)
    chosen = chosen.copy()
    chosen.columns = [str(c).strip() for c in chosen.columns]

    # Keep top n rows
    n = max(1, int(n))
    out = chosen.head(n).to_dict(orient="records")
    return out


# ----------------------------
# Titanic-style example analysis (for /analyze with CSV)
# ----------------------------
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


# ----------------------------
# Helpers for robust parsing from Wikipedia table
# ----------------------------
def _num_from_money(val) -> float:
    """Parse money strings like '$2,923,706,026' -> 2923706026.0"""
    if pd.isna(val):
        return float('nan')
    s = str(val)
    # Remove everything except digits and dot
    s = "".join(ch for ch in s if ch.isdigit() or ch == ".")
    try:
        return float(s) if s else float('nan')
    except Exception:
        return float('nan')


def _num_from_intlike(val) -> float:
    """Parse int-ish strings like '1' or '1[nb 2]' -> 1.0"""
    if pd.isna(val):
        return float('nan')
    s = "".join(ch for ch in str(val) if ch.isdigit())
    try:
        return float(s) if s else float('nan')
    except Exception:
        return float('nan')


def _year_from_any(val) -> float:
    """Extract a 4-digit year like '1997[1]' -> 1997.0 (bounds 1870..2100)."""
    if pd.isna(val):
        return float('nan')
    s = str(val)
    for i in range(len(s) - 3):
        chunk = s[i:i+4]
        if chunk.isdigit():
            y = int(chunk)
            if 1870 <= y <= 2100:
                return float(y)
    return float('nan')


# ----------------------------
# Wikipedia Questions Handler
# ----------------------------
def answer_wiki_film_questions() -> List[str]:
    """
    Scrape Wikipedia Highest-grossing films table and answer:
      1) # of $2bn movies released before 2000
      2) Earliest film that grossed > $1.5bn (title)
      3) Correlation between Rank and Peak (6 decimals, as string)
      4) Scatterplot (Rank vs Peak) with dotted red regression line, base64 data URI
    Returns a list[str] of length 4, in that order.
    """
    url = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
    tables = pd.read_html(url)  # needs lxml

    chosen = None
    # Prefer a table that clearly has these columns
    for t in tables:
        cols = {str(c).strip().lower() for c in t.columns}
        if {"rank", "peak", "title"}.issubset(cols) and any("gross" in c for c in cols) and "year" in cols:
            chosen = t
            break
    # Fallbacks
    if chosen is None:
        for t in tables:
            cols = {str(c).strip().lower() for c in t.columns}
            if {"rank", "title"}.issubset(cols):
                chosen = t
                break
    if chosen is None:
        return ["0", "Unknown", "0.000000", "data:image/png;base64,"]

    df = chosen.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Resolve best-matching column names flexibly
    def col_like(options):
        for name in df.columns:
            low = name.lower()
            for opt in options:
                if opt in low:
                    return name
        return None

    col_rank = col_like(["rank"])
    col_peak = col_like(["peak"])
    col_title = col_like(["title"])
    col_year  = col_like(["year"])
    col_gross = col_like(["worldwide gross", "global", "gross"])

    # Numeric coercions
    if col_rank: df["_rank"] = df[col_rank].map(_num_from_intlike)
    if col_peak: df["_peak"] = df[col_peak].map(_num_from_intlike)
    if col_year: df["_year"] = df[col_year].map(_year_from_any)
    if col_gross: df["_gross"] = df[col_gross].map(_num_from_money)

    # 1) $2bn movies before 2000
    if "_gross" in df and "_year" in df:
        cnt_2bn_pre2000 = int(((df["_gross"] >= 2_000_000_000) & (df["_year"] < 2000)).sum())
        ans1 = str(cnt_2bn_pre2000)
    else:
        ans1 = "0"

    # 2) Earliest film > $1.5bn (by year)
    if "_gross" in df and "_year" in df and col_title:
        over = df[df["_gross"] >= 1_500_000_000].dropna(subset=["_year"])
        if not over.empty:
            earliest = over.sort_values("_year").iloc[0]
            ans2 = str(earliest[col_title])
        else:
            ans2 = "Unknown"
    else:
        ans2 = "Unknown"

    # 3) Corr(Rank, Peak) with 6 decimals
    if "_rank" in df and "_peak" in df:
        sub = df[["_rank", "_peak"]].dropna()
        if not sub.empty and sub["_rank"].nunique() > 1 and sub["_peak"].nunique() > 1:
            corr = float(pd.Series(sub["_rank"]).corr(pd.Series(sub["_peak"])))
            ans3 = f"{corr:.6f}"
        else:
            ans3 = "0.000000"
    else:
        ans3 = "0.000000"

    # 4) Scatter Rank vs Peak, dotted red regression line
    if "_rank" in df and "_peak" in df:
        sub = df[["_rank", "_peak"]].dropna()
        if not sub.empty:
            x = sub["_rank"].astype(float).to_numpy()
            y = sub["_peak"].astype(float).to_numpy()
            # Regression
            try:
                m, b = np.polyfit(x, y, 1)
            except Exception:
                m, b = 0.0, float(np.nan)

            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(x, y, s=10)
            xs = np.sort(x)
            ax.plot(xs, m * xs + b, linestyle=":", linewidth=2, color="red")
            ax.set_xlabel("Rank")
            ax.set_ylabel("Peak")
            ax.set_title("Rank vs Peak")
            ans4 = _figure_to_base64(fig)
        else:
            ans4 = "data:image/png;base64,"
    else:
        ans4 = "data:image/png;base64,"

    return [ans1, ans2, ans3, ans4]
