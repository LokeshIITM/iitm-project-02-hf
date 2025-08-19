# run.py
import os
import sys
import requests

def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py <csvfile-or-url>")
        sys.exit(1)

    arg = sys.argv[1]

    # Case 1: Evaluator passes your Space URL
    if arg.startswith("http://") or arg.startswith("https://"):
        space_url = arg.rstrip("/")
        analyze_url = f"{space_url}/analyze"

        if not os.path.exists("questions.txt"):
            print("questions.txt not found")
            sys.exit(1)

        with open("questions.txt", "rb") as qf:
            files = {"questions": ("questions.txt", qf, "text/plain")}
            data = {"mode": "eval"}  # always eval mode
            r = requests.post(analyze_url, files=files, data=data, timeout=120)

        print("Mode: eval")
        print("Status:", r.status_code)
        print("Response:")
        print(r.text)
        return

    # Case 2: Local CSV file provided
    csvfile = arg
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
        data = {"mode": "eval"}
        space_url = "https://lokeshiitm-iitm-project-fastapi-02.hf.space"
        analyze_url = f"{space_url}/analyze"
        r = requests.post(analyze_url, files=files, data=data, timeout=120)

    print("Mode: eval")
    print("Status:", r.status_code)
    print("Response:")
    print(r.text)

if __name__ == "__main__":
    main()
