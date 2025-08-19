# run.py
import os, sys, requests

# Set this to your deployed Hugging Face Space URL
SPACE_URL = "https://lokeshiitm-iitm-project-fastapi-02.hf.space"
ANALYZE_URL = f"{SPACE_URL.rstrip('/')}/analyze"

def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py <csvfile>")
        sys.exit(1)

    csvfile = sys.argv[1]

    if not os.path.exists("questions.txt"):
        print("questions.txt not found")
        sys.exit(1)
    if not os.path.exists(csvfile):
        print(f"{csvfile} not found")
        sys.exit(1)

    with open("questions.txt", "rb") as qf, open(csvfile, "rb") as df:
        files = {
            "questions": ("questions.txt", qf, "text/plain"),
            "data": (os.path.basename(csvfile), df, "text/csv"),
        }
        # Always run in eval mode for grading
        data = {"mode": "eval"}
        r = requests.post(ANALYZE_URL, files=files, data=data, timeout=120)

    print("Mode: eval")
    print("Status:", r.status_code)
    print("Response:")
    print(r.text)

if __name__ == "__main__":
    main()
