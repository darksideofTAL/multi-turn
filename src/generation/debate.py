import json

from langchain_core.language_models.chat_models import BaseChatModel

from generation.prompting import GenerationPrompter, invoke_prompt
from generation.schemas import (
    AdvocateResponse,
    DebateResult,
    DebateTurn,
    GeneratedSample,
    JudgeResponse,
)


DEFAULT_JUDGES = [
    {
        "persona": "recall-oriented",
        "instructions": "Prioritize catching true positives. If the criterion reasonably holds, choose that label.",
    },
    {
        "persona": "strict precision-oriented",
        "instructions": "Use no flexibility or interpretive leeway. Choose the positive label only when the criterion clearly holds.",
    },
]


def normalize_label(label: str, labels: list[str]) -> str:
    clean_label = str(label).strip().strip('"\'')
    for configured_label in labels:
        if clean_label.lower() == configured_label.lower():
            return configured_label
    return clean_label


class DebateValidator:
    def __init__(
        self,
        llm: BaseChatModel,
        prompter: GenerationPrompter,
        criterion: str,
        labels: list[str],
        max_rounds: int = 2,
        judges: list[dict[str, str]] | None = None,
    ):
        self.llm = llm
        self.prompter = prompter
        self.criterion = criterion
        self.labels = labels
        self.max_rounds = max(1, max_rounds)
        self.judges = judges or DEFAULT_JUDGES

    def validate(self, sample: GeneratedSample, target_label: str) -> DebateResult:
        turns: list[DebateTurn] = []
        path: list[list[str]] = []
        advocate = self._advocate_first_round(sample, target_label)

        for round_index in range(self.max_rounds):
            if round_index > 0:
                advocate = self._advocate_followup_round(sample, target_label, advocate, turns[-1].judges)

            judge_responses = [
                self._judge_response(round_index, sample, advocate, idx, turns)
                for idx in range(len(self.judges))
            ]
            normalized_labels = [normalize_label(resp.label, self.labels) for resp in judge_responses]
            path.append(normalized_labels)
            turns.append(DebateTurn(round_index=round_index, advocate=advocate, judges=judge_responses))

            if all(label == target_label for label in normalized_labels):
                return DebateResult(valid=True, feedback="", path=path, turns=turns)

        feedback = self._feedback(turns[-1].judges, target_label)
        return DebateResult(valid=False, feedback=feedback, path=path, turns=turns)

    def _advocate_first_round(self, sample: GeneratedSample, target_label: str) -> AdvocateResponse:
        return invoke_prompt(
            self.llm,
            self.prompter,
            "advocate_first_round",
            AdvocateResponse,
            target_label=target_label,
            evaluation_criterion=self.criterion,
            sample=sample.input_block,
            reasoning=sample.reasoning,
        )

    def _advocate_followup_round(
        self,
        sample: GeneratedSample,
        target_label: str,
        previous: AdvocateResponse,
        judges: list[JudgeResponse],
    ) -> AdvocateResponse:
        return invoke_prompt(
            self.llm,
            self.prompter,
            "advocate_followup_round",
            AdvocateResponse,
            target_label=target_label,
            evaluation_criterion=self.criterion,
            sample=sample.input_block,
            previous_argument=previous.argument,
            judge_responses=json.dumps([judge.model_dump() for judge in judges], indent=2),
        )

    def _judge_response(
        self,
        round_index: int,
        sample: GeneratedSample,
        advocate: AdvocateResponse,
        judge_index: int,
        previous_turns: list[DebateTurn],
    ) -> JudgeResponse:
        judge = self.judges[judge_index]
        if round_index == 0:
            return invoke_prompt(
                self.llm,
                self.prompter,
                "judge_first_round",
                JudgeResponse,
                persona=judge["persona"],
                persona_instructions=judge["instructions"],
                labels=self.labels,
                sample=sample.input_block,
                evaluation_criterion=self.criterion,
                advocate_argument=advocate.argument,
            )

        previous_response = previous_turns[-1].judges[judge_index]
        other_responses = [
            response.model_dump()
            for idx, response in enumerate(previous_turns[-1].judges)
            if idx != judge_index
        ]
        return invoke_prompt(
            self.llm,
            self.prompter,
            "judge_followup_round",
            JudgeResponse,
            persona=judge["persona"],
            persona_instructions=judge["instructions"],
            labels=self.labels,
            sample=sample.input_block,
            evaluation_criterion=self.criterion,
            own_reasoning=previous_response.reasoning,
            own_label=previous_response.label,
            own_confidence=previous_response.confidence,
            advocate_argument=advocate.argument,
            previous_responses=json.dumps(other_responses, indent=2),
        )

    def _feedback(self, judges: list[JudgeResponse], target_label: str) -> str:
        dissent = [
            judge.reasoning
            for judge in judges
            if normalize_label(judge.label, self.labels) != target_label
        ]
        return "\n\n".join(dissent)
