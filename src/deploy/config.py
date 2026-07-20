from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DeployConfig:
    model_dir: Path
    dtype: str = "bfloat16"
    max_seq_length: int = 4096
    trust_remote_code: bool = False
    device: str | None = None
    labels: list[str] | None = None
    max_forward_batch_tokens: int = 4096
    max_forward_batch_items: int = 16
    # torch.compile the backbone. no-cudagraphs + dynamic suits the variable
    # (batch, seq) shapes from length-bucketing without per-shape recompiles.
    compile: bool = True
    compile_mode: str = "max-autotune-no-cudagraphs"
    compile_dynamic: bool = True

    @classmethod
    def from_params(
        cls,
        *,
        model_dir: str | Path,
        dtype: str = "bfloat16",
        max_seq_length: int = 4096,
        trust_remote_code: bool = False,
        device: str | None = None,
        labels: list[str] | None = None,
        max_forward_batch_tokens: int = 4096,
        max_forward_batch_items: int = 16,
        compile: bool = True,
        compile_mode: str = "max-autotune-no-cudagraphs",
        compile_dynamic: bool = True,
    ) -> "DeployConfig":
        return cls(
            model_dir=Path(model_dir),
            dtype=dtype,
            max_seq_length=int(max_seq_length),
            trust_remote_code=bool(trust_remote_code),
            device=device,
            labels=list(labels) if labels else None,
            max_forward_batch_tokens=int(max_forward_batch_tokens),
            max_forward_batch_items=int(max_forward_batch_items),
            compile=bool(compile),
            compile_mode=compile_mode,
            compile_dynamic=bool(compile_dynamic),
        )
