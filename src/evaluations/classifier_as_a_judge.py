import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data_handler.data_class import ClsfSample
from evaluations.inference_prompter import Prompter
from evaluations.model_runner import InferenceModel


class TransformerClassifier(InferenceModel):
    def __init__(
        self,
        model_name: str,
        prompter: Prompter,
        label_map: dict[str, str] | None = None,
        max_seq_length: int = 4096,
        trust_remote_code: bool = False,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.model_name = model_name
        self.prompter = prompter
        self.label_map = label_map or {}
        self.max_seq_length = max_seq_length

    def predict(self, clsf_sample: ClsfSample):
        system_msg = self.prompter.get_system_msg(clsf_sample)
        user_msg = self.prompter.get_user_msg(clsf_sample)
        text = f"{system_msg}\n\n{user_msg}"

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length,
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        label_id = int(torch.argmax(logits, dim=-1).item())
        label = self.model.config.id2label.get(label_id) or self.model.config.id2label.get(str(label_id), str(label_id))
        return self.label_map.get(label, label)

    def get_model_name(self) -> str:
        return self.model_name
