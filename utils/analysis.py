# utils/analysis.py
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp")  # avoid MPL cache warning in read-only envs

import base64
from io import BytesIO
from typing import List, Dict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


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
# Auto analysis (generic, dtype-driven)
# ----------------------------
def auto_analyze(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {"error": "Empty dataset"}

    # Try to coerce date-like columns
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    date_cols = df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()

    out = {
        "row_count": len(df),
        "columns": list(df.columns),
        "stats": {},
        "plots": []
    }

    # ---- numeric summaries ----
    for col in num_cols:
        col_series = df[col].dropna()
        out["stats"][col] = {
            "mean": float(np.nan_to_num(col_series.mean(), nan=0.0)),
            "median": float(np.nan_to_num(col_series.median(), nan=0.0)),
            "std": float(np.nan_to_num(col_series.std(), nan=0.0))
        }
        # histogram
        fig, ax = plt.subplots()
        col_series.hist(ax=ax)
        ax.set_title(f"Histogram of {col}")
        out["plots"].append(_figure_to_base64(fig))

    # ---- scatter + regression line (if ≥2 numeric) ----
    if len(num_cols) >= 2:
        x, y = num_cols[:2]
        corr = df[[x, y]].corr().iloc[0, 1]
        out["stats"]["correlation"] = {f"{x}-{y}": round(float(np.nan_to_num(corr, nan=0.0)), 4)}

        fig, ax = plt.subplots()
        ax.scatter(df[x], df[y], alpha=0.5)

        # regression line with safety
        try:
            xs = np.sort(df[x].dropna())
            if len(xs) > 1:
                coeffs = np.polyfit(df[x].dropna(), df[y].dropna(), 1)
                ys = np.polyval(coeffs, xs)
                ax.plot(xs, ys, linestyle=":", color="red", label="Regression line")
                ax.legend()
        except Exception:
            pass

        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"{x} vs {y}")
        out["plots"].append(_figure_to_base64(fig))

    # ---- category vs numeric (bar) ----
    if num_cols and cat_cols:
        n, c = num_cols[0], cat_cols[0]
        try:
            fig, ax = plt.subplots()
            df.groupby(c)[n].mean().plot(kind="bar", ax=ax)
            ax.set_title(f"Mean {n} by {c}")
            out["plots"].append(_figure_to_base64(fig))
        except Exception:
            pass

    # ---- datetime vs numeric (line) ----
    if num_cols and date_cols:
        n, d = num_cols[0], date_cols[0]
        try:
            fig, ax = plt.subplots()
            ax.plot(df[d], df[n], marker="o")
            ax.set_title(f"{n} over {d}")
            out["plots"].append(_figure_to_base64(fig))
        except Exception:
            pass

    # ---- network-like (if 2+ string cols, no numerics) ----
    if len(cat_cols) >= 2 and not num_cols:
        try:
            src, tgt = cat_cols[:2]
            G = nx.from_pandas_edgelist(df, source=src, target=tgt)
            fig, ax = plt.subplots()
            nx.draw(G, with_labels=True, node_size=500, ax=ax)
            out["plots"].append(_figure_to_base64(fig))
        except Exception:
            pass

    return out


# ----------------------------
# Wikipedia fetch (used by /wiki endpoint)
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
