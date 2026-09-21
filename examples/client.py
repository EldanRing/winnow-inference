"""Both interfaces on one local server; Python standard library only."""

import json
from pathlib import Path
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
    print(
        post(
            "/v1/systemone",
            json.loads(Path(__file__).with_name("decisions.json").read_text()),
        )
    )
    print(
        post(
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
    )
