from fastapi import Form  # add this import at top

@app.post("/analyze")
async def analyze_data(
    questions: UploadFile = File(...),
    data: UploadFile | None = File(None),
    mode: str = Form("eval"),              # eval or play
    temperature: float = Form(0.7),
    top_p: float = Form(0.9),
    top_k: int = Form(40),
    presence_penalty: float = Form(0.0),
    frequency_penalty: float = Form(0.0),
    max_tokens: int = Form(800),
    stop: str | None = Form(None),         # optional stop sequence
):
    # Read questions.txt
    try:
        questions_text = (await questions.read()).decode("utf-8", errors="ignore")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid questions.txt: {e}")

    # Read optional dataset just for preview (lightweight)
    df = _read_dataframe(data) if data else None
    preview = _df_preview(df)

    # Prompt
    user_prompt = f"questions.txt:\n{questions_text}\n\nData preview:\n{preview}"

    payload = {
        "model": AIPIPE_MODEL,
        "messages": [
            {"role": "system", "content": "You are a data analyst agent. Return JSON only with the required keys."},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
    }

    if mode == "eval":
        payload.update({
            "temperature": 0,
            "top_p": 1,
            "top_k": 1,
            "stop": ["}"],   # truncate after JSON
        })
    else:  # play mode — use user-supplied knobs
        payload.update({
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
        })
        if stop:
            payload["stop"] = [stop]

    # Call LLM
    url = f"{AIPIPE_BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {AIPIPE_TOKEN}", "Content-Type": "application/json"}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        llm_output = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM request failed: {e}")

    # Ensure JSON
    try:
        return JSONResponse(json.loads(llm_output))
    except Exception:
        return JSONResponse({"error": "Invalid JSON from LLM", "raw": llm_output})
