"""Multi-turn monitor library (mtlib).

A hierarchical monitor: the frozen single-turn classifier encodes each turn to
one policy-conditioned latent (mtlib.encoder), and a small causal transformer
(mtlib.aggregator) attends over the sequence of latents to produce a
conversation-so-far verdict at every turn. Same philosophy as the single-turn
product: frozen backbone, tiny head, policy supplied only at inference.
"""
