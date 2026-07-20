from common.settings import LLMConfig, Settings
from evaluations.classifier_as_a_judge import TransformerClassifier
from evaluations.inference_prompter import Prompter
from evaluations.llm_as_a_judge import LLMAsAJudge
from evaluations.slm_as_a_judge import TransformerSLM


def get_inference_model(settings: Settings, conf_elem: dict, prompter: Prompter):
    if conf_elem.get("llm_config", "") != "":
        return LLMAsAJudge(
            settings=settings,
            llm_config=LLMConfig(**conf_elem["llm_config"]),
            prompter=prompter,
        )

    elif conf_elem.get("slm_model", "") != "":
        return TransformerSLM(
            model_name=conf_elem["slm_model"],
            prompter=prompter,
        )

    elif conf_elem.get("classifier_model", "") != "":
        classifier_conf = conf_elem["classifier_model"]
        if isinstance(classifier_conf, str):
            classifier_conf = {"name": classifier_conf}
        return TransformerClassifier(
            model_name=classifier_conf["name"],
            prompter=prompter,
            label_map=classifier_conf.get("label_map"),
            max_seq_length=classifier_conf.get("max_seq_length", 4096),
            trust_remote_code=classifier_conf.get("trust_remote_code", False),
        )

    else:
        raise Exception("Missing inference model in configuration file")
