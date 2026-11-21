import argparse
import json
import os
import subprocess
import sys
import uuid
from collections import Counter
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def post(args, path, payload):
    headers = {"Content-Type": "application/json"}
    if key := os.environ.get("VLLM_API_KEY"):
        headers["Authorization"] = f"Bearer {key}"
    request = Request(args.url.rstrip("/") + path, json.dumps(payload).encode(), headers)
    with urlopen(request, timeout=args.timeout) as response:
        return json.load(response)


def chat_prompt(args):
    target = args.context - 1
    low, high = 1, target
    while low <= high:
        repeats = (low + high) // 2
        messages = [{"role": "user", "content": "hello" + " hello" * (repeats - 1)}]
        count = post(args, "/tokenize", {"model": args.model, "messages": messages})["count"]
        if count == target:
            return messages
        if count < target:
            low = repeats + 1
        else:
            high = repeats - 1
    raise ValueError(f"could not construct a chat prompt of {target} tokens")


def chat(args, messages):
    try:
        post(args, "/v1/chat/completions", {
            "model": args.model,
            "messages": messages,
            "max_completion_tokens": 1,
            "cache_salt": uuid.uuid4().hex,
        })
        return "accepted"
    except HTTPError as err:
        err.close()
        return f"rejected (HTTP {err.code})"
    except (OSError, ValueError) as err:
        return f"error ({err})"


def memory_used(gpu):
    output = subprocess.check_output([
        "nvidia-smi", "-i", gpu, "--query-gpu=memory.used", "--format=csv,noheader,nounits",
    ], text=True, timeout=5)
    return int(output.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--users", type=int, required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if args.users < 1 or args.context < 2 or not args.timeout > 0:
        parser.error("expected positive users and timeout, and context of at least 2")

    baseline = peak = memory_used(args.gpu)
    messages = chat_prompt(args)
    with ThreadPoolExecutor(max_workers=args.users) as pool:
        futures = [pool.submit(chat, args, messages) for _ in range(args.users)]
        while True:
            peak = max(peak, memory_used(args.gpu))
            if not wait(futures, timeout=0.2, return_when=ALL_COMPLETED).not_done:
                break

    counts = Counter(future.result() for future in futures)
    print(f"GPU memory used (MiB): {baseline} baseline, {peak} peak sampled")
    print(f"{args.users} chats at {args.context} tokens (including 1 generated token)")
    for status, count in sorted(counts.items()):
        print(f"{status}: {count}")
    return int(counts["accepted"] != args.users)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as err:
        sys.exit(str(err))
