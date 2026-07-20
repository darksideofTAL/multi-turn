import argparse
import os

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")  # before importing the model

import numpy as np

from deploy.config import DeployConfig
from deploy.engine import ClassifierEngine


def _to_str(value) -> str:
    return (
        value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else str(value)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="deploy.pytriton_local")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--max-queue-delay-us", type=int, default=5000)
    parser.add_argument("--model-name", default="talmonitor")
    parser.add_argument("--http-port", type=int, default=8000)
    parser.add_argument("--grpc-port", type=int, default=8001)
    parser.add_argument("--metrics-port", type=int, default=8002)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from pytriton.decorators import batch
    from pytriton.model_config import DynamicBatcher, ModelConfig, Tensor
    from pytriton.triton import Triton, TritonConfig

    engine = ClassifierEngine(
        DeployConfig.from_params(
            model_dir=args.model_dir,
            dtype=args.dtype,
            max_seq_length=args.max_seq_length,
            device=args.device,
        )
    )
    labels = engine.labels

    @batch
    def infer(policy, input_block):
        policies = [_to_str(row[0]) for row in policy]
        blocks = [_to_str(row[0]) for row in input_block]
        results = engine.classify_batch(policies, blocks)
        return {
            "label": np.array(
                [[r["label"].encode("utf-8")] for r in results], dtype=object
            ),
            "scores": np.array(
                [[r["scores"][lbl] for lbl in labels] for r in results],
                dtype=np.float32,
            ),
            "model": np.array(
                [[r["model"].encode("utf-8")] for r in results], dtype=object
            ),
        }

    triton_config = TritonConfig(
        http_port=args.http_port,
        grpc_port=args.grpc_port,
        metrics_port=args.metrics_port,
    )
    with Triton(config=triton_config) as triton:
        triton.bind(
            model_name=args.model_name,
            infer_func=infer,
            inputs=[
                Tensor(name="policy", dtype=bytes, shape=(1,)),
                Tensor(name="input_block", dtype=bytes, shape=(1,)),
            ],
            outputs=[
                Tensor(name="label", dtype=bytes, shape=(1,)),
                Tensor(name="scores", dtype=np.float32, shape=(-1,)),
                Tensor(name="model", dtype=bytes, shape=(1,)),
            ],
            config=ModelConfig(
                max_batch_size=args.max_batch_size,
                batcher=DynamicBatcher(
                    max_queue_delay_microseconds=args.max_queue_delay_us
                ),
            ),
        )
        triton.serve()


if __name__ == "__main__":
    main()
