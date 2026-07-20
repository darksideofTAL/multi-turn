"""DuoGuard × TALMONITOR two-player RL pipeline.

An adversarial generator/classifier loop on top of TALMONITOR's binary
"policy-as-input" classifier, reusing TALMONITOR's modules (``training.runner``,
``evaluations.inference_prompter.Prompter``, ``generation.policies``, the judge
prompts). The min-max game, preference levels, and math are in ``duoguard/plan.md``.

Layout: ``config`` (config/paths/resolvers), ``utils`` (io/filters/render/load),
``prompts`` (generator + judge prompts), ``hf_generate`` (batched HF generation),
``steps`` (generate/verify/classify/preference), ``train`` (classifier/generator),
``iterate`` (orchestrator), ``eval`` (adversarial build/score).
"""
