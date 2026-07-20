not that greatly written readme

* A frozen 12B single-turn classifier that evaluates each turn independently and produces a policy-conditioned latent representation.

* A 15M causal transformer that reads the sequence of turn latents and produces a conversation-level verdict after every turn.

contact aritra - for evals but its also there in folders

## Scores

The thingy returns two violation scores:

* `turn_violation_prob`: detects content that violates the policy within a single turn.

* `conversation_violation_prob`: detects compositional violations that only become clear when multiple individually benign turns are read together.

The conversation model was trained only on compositional violations. Applications should therefore check both scores:


* Use `turn_violation_prob` for direct, single-turn violations.

* Use `conversation_violation_prob >= tau` for multi-turn violations.

## Build

Build the Docker image from the repository root:

```bash

docker build \

  -f deploy_multiturn/Dockerfile \

  -t talmonitor-multiturn .

```

The repository already includes:


* The 15M aggregator checkpoint, approximately 44 MB

* The Triton model configuration

* The `cscale ALL`, seed 0 checkpoint


These files are stored under:

```text

model_repository_multiturn/

```

The repository does not include the 12B turn encoder. Mount the following checkpoint when starting the container:


```text

google-gemma-3-12b-it-v4-nofreeze-n7500

```

The mount path must match `parameters.model_dir` in `config.pbtxt`.

## Run

```bash

docker run -d \

  --name talmonitor-multiturn \

  --gpus '"device=<GPU-or-MIG-UUID>"' \

  --shm-size 1g \

  -p 8700:8000 \

  -p 8701:8001 \

  -p 8702:8002 \

  -v /path/to/google-gemma-3-12b-it-v4-nofreeze-n7500:/models/google-gemma-3-12b-it-v4-nofreeze-n7500:ro \

  talmonitor-multiturn

```

On the B200 cluster, the checkpoint is located at ( idk where it is a100, but can be replaced ig 


```text

/raid/frontiers_ashoka/BARRED/models/google-gemma-3-12b-it-v4-nofreeze-n7500

```


Copy it to the target machine, then update the host side of the `-v` mount.


## basic eh usage


The thingy exposes one Triton model:


```text

/v2/models/talmonitor_multiturn/infer

```

It accepts four string inputs:

* `op`

* `conv_id`

* `role`

* `text`


It returns one JSON string output:

* `result`

### Start a conversation

```text

op=start

text=<policy prompt>

```
Response:

```json

{

  "conv_id": "..."

}

```

### Feed a turn

```text

op=feed

conv_id=<conversation ID>

role=user|agent

text=<turn text>

```

Response:



```json

{

  "turn_index": 0,

  "turn_violation_prob": 0.0,

  "conversation_violation_prob": 0.0,

  "per_turn_conversation_probs": [],

  "flagged": false,

  "tau": 0.99196

}

```

### Finish a conversation

```text

op=finish

conv_id=<conversation ID>

```

This returns the final verdict and removes the conversation state from the server.


### Abort a conversation



```text

op=abort

conv_id=<conversation ID>

```

Response:



```json

{

  "aborted": true

}

```

The 12B model does not keep conversation state between turns.



Each `feed` request performs:



1. One single-turn encoding pass

2. One aggregator pass





