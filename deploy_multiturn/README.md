# talmonitor_multiturn — multi-turn conversation monitor

Triton deployment of the hierarchical multi-turn monitor: the frozen 12B
single-turn classifier encodes each turn to one policy-conditioned latent, and
a 15M causal transformer aggregates the latent sequence into a
conversation-so-far verdict at every turn. Design and evaluation:
`experiments/multiturn/ARCHITECTURE.md` and `experiments/multiturn/RESULTS.md`.

Division of labor: the per-turn score (`turn_violation_prob`, the shipped
single-turn classifier) catches turns that violate on their own; the
aggregator score (`conversation_violation_prob`) targets **compositional**
violations — conversations whose turns are individually benign but violating
read together. The aggregator is trained comp-only, so act on BOTH scores:
`turn_p` for blatant turns, `conv_p >= tau` for compositional ones.

## Build

Self-contained image from the public Triton base (repo root as context):

    docker build -f deploy_multiturn/Dockerfile -t talmonitor-multiturn .

The 15M aggregator checkpoint (44 MB, `cscale ALL` seed 0) and the Triton
config ship inside the repo under `model_repository_multiturn/`. The 12B turn
encoder does NOT — bind-mount the `google-gemma-3-12b-it-v4-nofreeze-n7500`
checkpoint at run time (path must match `parameters.model_dir` in
`config.pbtxt`).

## Run

    docker run -d --name talmonitor-multiturn \
      --gpus '"device=<GPU-or-MIG-UUID>"' --shm-size 1g \
      -p 8700:8000 -p 8701:8001 -p 8702:8002 \
      -v /path/to/google-gemma-3-12b-it-v4-nofreeze-n7500:/models/google-gemma-3-12b-it-v4-nofreeze-n7500:ro \
      talmonitor-multiturn

The checkpoint lives at `/raid/frontiers_ashoka/BARRED/models/google-gemma-3-12b-it-v4-nofreeze-n7500` on the B200 cluster — copy it to the target machine and point the `-v` mount at it.

Needs ~24 GB of GPU memory (bf16 12B + aggregator); a 1g.45gb MIG slice of a
B200 is enough. On a MIG-partitioned host prefer pinning by MIG UUID
(`nvidia-smi -L`) — `device=N:S` indices remap when instances are recreated,
e.g. after a reboot. No torch.compile: cold start to READY is ~75 s.
Readiness: `curl localhost:8700/v2/models/talmonitor_multiturn/ready`.

## Protocol

One Triton model, four string inputs (`op`, `conv_id`, `role`, `text`), one
JSON string output (`result`), on `/v2/models/talmonitor_multiturn/infer`:

    op=start   text=<policy prompt>          -> {"conv_id"}
    op=feed    conv_id, role=user|agent,     -> {"turn_index", "turn_violation_prob",
               text=<turn text>                  "conversation_violation_prob",
                                                 "per_turn_conversation_probs",
                                                 "flagged", "tau"}
    op=finish  conv_id                       -> final verdict, drops state
    op=abort   conv_id                       -> {"aborted": bool}

Per-conversation server state is ~8 KB/turn (latents only; nothing persists
inside the 12B between turns). A feed costs one single-turn encode plus a
sub-millisecond aggregator pass (~60 ms/turn on a B200 MIG slice). Requests
for one `conv_id` must arrive in order — a synchronous client is enough. Idle
conversations are evicted after `conv_ttl_seconds` (default 600).

`tau` (0.99196) is calibrated on 2.8k benign conversations for a 2%
PER-CONVERSATION false-positive rate; `flagged` is `conv_p >= tau`. Override
via `parameters.tau` in `config.pbtxt`.

## Client

    printf '<User>hi</User>\n<Agent>hello</Agent>\n' | \
      python3 deploy_multiturn/multiturn_client.py \
        --policy "<policy prompt>" [--url http://localhost:8700]

Feeds a `<User>`/`<Agent>`-tagged transcript turn by turn and prints per-turn
and final verdicts.
