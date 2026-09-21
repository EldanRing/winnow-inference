"""Both interfaces on one local server; Python standard library only."""

import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def post(path, body):
    request = Request(
        "http://127.0.0.1:8091" + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=180) as response:
        return json.load(response)


if __name__ == "__main__":
    try:
        decisions = post(
            "/v1/systemone",
            json.loads(Path(__file__).with_name("decisions.json").read_text()),
        )
        print("Typed decisions:\n" + json.dumps(decisions, indent=2))
        chat = post(
            "/v1/chat/completions",
            {
                "model": "Winnow-12B",
                "messages": [
                    {
                        "role": "user",
                        "content": "Explain prefix caching in two sentences.",
                    }
                ],
                "max_tokens": 4096,
            },
        )
        print("\nChat response:\n" + chat["choices"][0]["message"]["content"])
    except HTTPError as error:
        print(
            f"Server returned HTTP {error.code}. Check the server terminal for details.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    except (URLError, TimeoutError) as error:
        print(
            f"Cannot reach Winnow: {error.reason if isinstance(error, URLError) else error}\nStart python3 scripts/serve.py in another terminal and wait for the model to finish loading.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
