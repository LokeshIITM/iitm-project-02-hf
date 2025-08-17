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
    chosen = None
    for t in tables:
        cols = {str(c).strip().lower() for c in t.columns}
        if {"rank", "title"}.issubset(cols):
            chosen = t
            break
    if chosen is None and tables:
        chosen = tables[0]

    chosen = chosen.copy()
    chosen.columns = [str(c).strip() for c in chosen.columns]

    return chosen.head(max(1, int(n))).to_dict(orient="records")


# ----------------------------
# Deterministic Titanic-style analysis
# ----------------------------
def process_questions(questions_text: str, df: Optional[pd.DataFrame]) -> list:
    """
    Always return [row_count, avg_age, correlation, base64_plot].
    - row_count: int
    - avg_age: float (2 decimals)
    - correlation: float (6 decimals)
    - base64_plot: scatterplot with red dotted regression line
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return [0, 0.0, 0.0, "data:image/png;base64,"]

    results: list = []

    # 1) Row count
    row_count = int(len(df))
    results.append(row_count)

    # 2) Average age
    avg_age = 0.0
    if "age" in df.columns:
        try:
            avg_age = float(round(df["age"].astype(float).mean(), 2))
        except Exception:
            pass
    results.append(avg_age)

    # 3) Correlation (age vs fare)
    corr = 0.0
    if {"age", "fare"}.issubset(df.columns):
        try:
            sub = df[["age", "fare"]].dropna()
            if not sub.empty and sub["age"].nunique() > 1 and sub["fare"].nunique() > 1:
                corr = float(pd.Series(sub["age"].astype(float)).corr(pd.Series(sub["fare"].astype(float))))
        except Exception:
            pass
    results.append(round(corr, 6))

    # 4) Plot (scatter + regression line)
    plot_b64 = "data:image/png;base64,"
    if {"age", "fare"}.issubset(df.columns):
        try:
            x = df["age"].astype(float).to_numpy()
            y = df["fare"].astype(float).to_numpy()
            m, b = np.polyfit(x, y, 1)

            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(x, y, s=10)
            xs = np.sort(x)
            ax.plot(xs, m * xs + b, linestyle=":", linewidth=2, color="red")
            ax.set_xlabel("Age")
            ax.set_ylabel("Fare")
            ax.set_title("Age vs Fare")
            plot_b64 = _figure_to_base64(fig)
        except Exception:
            pass
    results.append(plot_b64)

    return results


# ----------------------------
# Helpers for robust parsing from Wikipedia table
# ----------------------------
def _num_from_money(val) -> float:
    if pd.isna(val):
        return float('nan')
    s = "".join(ch for ch in str(val) if ch.isdigit() or ch == ".")
    try:
        return float(s) if s else float('nan')
    except Exception:
        return float('nan')


def _num_from_intlike(val) -> float:
    if pd.isna(val):
        return float('nan')
    s = "".join(ch for ch in str(val) if ch.isdigit())
    try:
        return float(s) if s else float('nan')
    except Exception:
        return float('nan')


def _year_from_any(val) -> float:
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
    for t in tables:
        cols = {str(c).strip().lower() for c in t.columns}
        if {"rank", "peak", "title"}.issubset(cols) and any("gross" in c for c in cols) and "year" in cols:
            chosen = t
            break
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

    # 2) Earliest film > $1.5bn
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

    # 4) Scatter Rank vs Peak
    if "_rank" in df and "_peak" in df:
        sub = df[["_rank", "_peak"]].dropna()
        if not sub.empty:
            x = sub["_rank"].astype(float).to_numpy()
            y = sub["_peak"].astype(float).to_numpy()
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
