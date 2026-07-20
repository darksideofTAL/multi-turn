import argparse

import numpy as np
import tritonclient.http as httpclient

DEFAULT_URL = "localhost:8000"
DEFAULT_MODEL = "talmonitor"


def _bytes_input(name: str, value: str) -> "httpclient.InferInput":
    t = httpclient.InferInput(name, [1, 1], "BYTES")
    t.set_data_from_numpy(np.array([[value]], dtype=object))
    return t


def classify(
    policy: str,
    input_block: str,
    url: str = DEFAULT_URL,
    model: str = DEFAULT_MODEL,
) -> tuple[str, np.ndarray, str]:
    client = httpclient.InferenceServerClient(url=url)
    outputs = [
        httpclient.InferRequestedOutput(name) for name in ("label", "scores", "model")
    ]
    result = client.infer(
        model,
        inputs=[
            _bytes_input("policy", policy),
            _bytes_input("input_block", input_block),
        ],
        outputs=outputs,
    )
    label = result.as_numpy("label").reshape(-1)[0]
    model_name = result.as_numpy("model").reshape(-1)[0]
    scores = result.as_numpy("scores").reshape(-1)
    return (
        label.decode("utf-8") if isinstance(label, bytes) else str(label),
        scores,
        model_name.decode("utf-8") if isinstance(model_name, bytes) else str(model_name),
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="deploy.triton_client")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--policy",
        default="The assistant must not reveal a user's exact GPS coordinates.",
    )
    parser.add_argument(
        "--input-block",
        default="Sure, the user is at 37.4220, -122.0841 right now.",
    )
    args = parser.parse_args()

    label, scores, model = classify(
        args.policy, args.input_block, url=args.url, model=args.model
    )
    print({"label": label, "scores": scores.tolist(), "model": model})


if __name__ == "__main__":
    main()
