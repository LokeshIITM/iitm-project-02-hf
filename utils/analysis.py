# utils/analysis.py
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp")  # avoid MPL cache warning in read-only envs

import base64
from io import BytesIO
from typing import List, Dict
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
