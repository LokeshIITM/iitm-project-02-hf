# tests/test_wiki.py
import re
import base64
from fastapi.testclient import TestClient
from api.index import app

def test_wikipedia_handler_smoke_and_image_size():
    client = TestClient(app)

    q = (
        "Scrape the list of highest grossing films from Wikipedia. "
        "It is at the URL: https://wikipedia.org/wiki/List_of_highest-grossing_films\n\n"
        "Answer the following questions...\n"
        "1. How many $2 bn movies were released before 2000?\n"
        "2. Which is the earliest film that grossed over $1.5 bn?\n"
        "3. What's the correlation between the Rank and Peak?\n"
        "4. Draw a scatterplot of Rank and Peak with a dotted red regression line.\n"
    )

    files = {"questions": ("questions.txt", q.encode("utf-8"), "text/plain")}
    r = client.post("/analyze", files=files)
    assert r.status_code == 200

    data = r.json()
    # Expect exactly 4 strings: [count, earliest_title, corr_6dp, base64_png]
    assert isinstance(data, list) and len(data) == 4
    assert all(isinstance(x, str) for x in data)

    # 1) count is numeric string
    assert data[0].isdigit()

    # 2) earliest title is non-empty
    assert len(data[1]) > 0

    # 3) correlation has exactly 6 decimals
    assert re.fullmatch(r"-?\d+\.\d{6}", data[2]) is not None

    # 4) image data URI, PNG, under 100 KB
    assert data[3].startswith("data:image/png;base64,")
    b64 = data[3].split(",", 1)[1]
    raw = base64.b64decode(b64)
    assert raw.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(raw) < 100_000
