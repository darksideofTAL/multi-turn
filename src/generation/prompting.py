import json
import re
from pathlib import Path
from typing import Any, TypeVar

import yaml
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


class GenerationPrompter:
    def __init__(self, prompts_data: dict[str, Any]):
        self.prompts_data = prompts_data["generation"]

    @classmethod
    def from_default_file(cls) -> "GenerationPrompter":
        prompts_path = Path(__file__).parent.parent.parent / "prompts/generation.yaml.jinja2"
        with open(prompts_path) as f:
            return cls(yaml.safe_load(f))

    def system(self, name: str, **kwargs) -> str:
        return self.prompts_data[name]["system"].format(**kwargs).strip()

    def human(self, name: str, **kwargs) -> str:
        return self.prompts_data[name]["human"].format(**kwargs).strip()


def invoke_prompt(
    llm: BaseChatModel,
    prompter: GenerationPrompter,
    prompt_name: str,
    schema: type[T],
    **kwargs,
) -> T:
    messages = [
        SystemMessage(content=prompter.system(prompt_name, **kwargs)),
        HumanMessage(content=prompter.human(prompt_name, **kwargs)),
    ]
    return schema.model_validate(_extract_json(llm.invoke(messages).content))


def _extract_json(text: str):
    text = text.strip()

    # Reasoning models (Qwen <think>, DeepSeek-R1, ...) wrap chain-of-thought in a
    # <think>...</think> channel that often contains stray braces or example JSON.
    # The standard reasoning-parser step: keep only the answer after the final
    # </think> and parse that.
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Prefer a fenced ```json``` block; take the last one, since any example shown
    # earlier in the answer precedes the real result.
    fences = re.findall(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    for fenced in reversed(fences):
        try:
            return json.loads(fenced)
        except json.JSONDecodeError:
            continue

    # Tolerant fallback: return the last self-contained JSON object/array. Scan
    # opening braces from the end; raw_decode tolerates trailing text and ignores
    # earlier stray braces in surrounding prose.
    decoder = json.JSONDecoder()
    for idx in range(len(text) - 1, -1, -1):
        if text[idx] not in "{[":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, (dict, list)):
            return obj

    raise ValueError(f"No JSON object found in LLM output: {text}")
