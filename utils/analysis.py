# utils/analysis.py
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp")  # avoid MPL cache warning in read-only envs

import base64
from io import BytesIO
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx   # NEW for graph metrics


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


# ----------------------------
# New: Sales Analysis
# ----------------------------
def analyze_sales(df: pd.DataFrame) -> list:
    """
    Return [total_sales, top_region, day_sales_correlation,
            median_sales, total_sales_tax, bar_chart_base64]
    """
    if df is None or df.empty:
        return [0, "", 0.0, 0.0, 0.0, "data:image/png;base64,"]

    results = []

    # 1) Total sales
    total_sales = int(df["sales"].sum())
    results.append(total_sales)

    # 2) Top region
    top_region = df.groupby("region")["sales"].sum().idxmax()
    results.append(str(top_region))

    # 3) Correlation (day vs sales)
    corr = 0.0
    if {"day", "sales"}.issubset(df.columns):
        sub = df[["day", "sales"]].dropna()
        if not sub.empty and sub["day"].nunique() > 1:
            corr = float(sub["day"].astype(float).corr(sub["sales"].astype(float)))
    results.append(round(corr, 4))

    # 4) Median sales
    median_sales = float(df["sales"].median())
    results.append(int(median_sales))

    # 5) Total sales tax (10%)
    total_sales_tax = int(round(total_sales * 0.1))
    results.append(total_sales_tax)

    # 6) Plot (bar chart by region)
    plot_b64 = "data:image/png;base64,"
    try:
        region_sales = df.groupby("region")["sales"].sum()
        fig, ax = plt.subplots(figsize=(5, 4))
        region_sales.plot(kind="bar", ax=ax)
        ax.set_xlabel("Region")
        ax.set_ylabel("Total Sales")
        ax.set_title("Sales by Region")
        plot_b64 = _figure_to_base64(fig)
    except Exception:
        pass
    results.append(plot_b64)

    return results


# ----------------------------
# New: Network Analysis
# ----------------------------
def analyze_network(df: pd.DataFrame) -> list:
    """
    Compute graph metrics from edge list.
    Return [edge_count, highest_degree_node, degree_histogram_plot, network_plot]
    """
    if df is None or df.empty or not {"source", "target"}.issubset(df.columns.str.lower()):
        return [0, "", "data:image/png;base64,", "data:image/png;base64,"]

    # Normalize column names
    df = df.rename(columns={c.lower(): c for c in df.columns})
    G = nx.from_pandas_edgelist(df, source="source", target="target")

    results = []

    # 1) Edge count
    edge_count = G.number_of_edges()
    results.append(edge_count)

    # 2) Highest degree node
    if G.number_of_nodes() > 0:
        node, deg = max(G.degree, key=lambda x: x[1])
        results.append(str(node))
    else:
        results.append("")

    # 3) Degree histogram plot
    degrees = [d for _, d in G.degree()]
    fig1, ax1 = plt.subplots(figsize=(5, 4))
    ax1.hist(degrees, bins=range(1, max(degrees)+2))
    ax1.set_xlabel("Degree")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Degree Histogram")
    hist_b64 = _figure_to_base64(fig1)
    results.append(hist_b64)

    # 4) Network plot
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, ax=ax2, node_size=50, with_labels=False)
    ax2.set_title("Network Graph")
    net_b64 = _figure_to_base64(fig2)
    results.append(net_b64)

    return results
# ----------------------------
# New: Weather Analysis
# ----------------------------
def analyze_weather(df: pd.DataFrame) -> list:
    """
    Compute weather stats:
      - average_temp_c (float, 1 decimal)
      - max_precip_date (string, YYYY-MM-DD)
      - min_temp_c (int)
      - temp_precip_correlation (float, 5 decimals)
      - average_precip_mm (float, 1 decimal)
      - 2 plots: temperature trend, precipitation trend
    """
    if df is None or df.empty or not {"date", "temp_c", "precip_mm"}.issubset(df.columns.str.lower()):
        return [0.0, "", 0.0, 0.0, 0.0, "data:image/png;base64,", "data:image/png;base64,"]

    # Normalize column names
    df = df.rename(columns={c.lower(): c for c in df.columns})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "temp_c", "precip_mm"]).copy()

    results = []

    # 1) Average temperature (1 decimal)
    avg_temp = round(df["temp_c"].astype(float).mean(), 1)
    results.append(avg_temp)

    # 2) Date with maximum precipitation
    idx = df["precip_mm"].astype(float).idxmax()
    max_precip_date = str(df.loc[idx, "date"].date()) if pd.notna(idx) else ""
    results.append(max_precip_date)

    # 3) Minimum temperature
    min_temp = int(df["temp_c"].astype(float).min())
    results.append(min_temp)

    # 4) Correlation between temp and precip (5 decimals)
    corr = 0.0
    sub = df[["temp_c", "precip_mm"]].dropna()
    if not sub.empty and sub["temp_c"].nunique() > 1 and sub["precip_mm"].nunique() > 1:
        corr = float(sub["temp_c"].astype(float).corr(sub["precip_mm"].astype(float)))
    results.append(round(corr, 5))

    # 5) Average precipitation (1 decimal)
    avg_precip = round(df["precip_mm"].astype(float).mean(), 1)
    results.append(avg_precip)

    # 6) Plot: temperature trend line
    fig1, ax1 = plt.subplots(figsize=(5, 4))
    ax1.plot(df["date"], df["temp_c"], marker="o", linestyle="-")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Temp (°C)")
    ax1.set_title("Daily Temperature")
    temp_plot = _figure_to_base64(fig1)

    # 7) Plot: precipitation trend line
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    ax2.plot(df["date"], df["precip_mm"], marker="s", linestyle="-", color="blue")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Precip (mm)")
    ax2.set_title("Daily Precipitation")
    precip_plot = _figure_to_base64(fig2)

    results.extend([temp_plot, precip_plot])

    return results
