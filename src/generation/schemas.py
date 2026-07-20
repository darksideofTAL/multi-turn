from pydantic import BaseModel


class GeneratedSample(BaseModel):
    input_block: str
    label: str
    reasoning: str


class AdvocateResponse(BaseModel):
    argument: str


class JudgeResponse(BaseModel):
    reasoning: str
    confidence: str
    label: str


class DebateTurn(BaseModel):
    round_index: int
    advocate: AdvocateResponse
    judges: list[JudgeResponse]


class DebateResult(BaseModel):
    valid: bool
    feedback: str
    path: list[list[str]]
    turns: list[DebateTurn]


class AcceptedSample(BaseModel):
    guardrail_id: str
    policy_prompt: str = ""
    input_block: str
    label: str
    reasoning: str
    refinement_round: int
    debate_path: list[list[str]]


class RejectedAttempt(BaseModel):
    guardrail_id: str
    sample: GeneratedSample
    target_label: str
    debate: DebateResult
