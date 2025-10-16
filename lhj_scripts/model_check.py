from transformers import AutoConfig, AutoModelForSequenceClassification

def classify_repo(repo):
    cfg = AutoConfig.from_pretrained(repo)
    arch_flag = any(n.endswith("ForSequenceClassification") for n in (cfg.architectures or []))
    task_flag = getattr(cfg, "problem_type", None) in {"single_label_classification","multi_label_classification"}

    model, info = AutoModelForSequenceClassification.from_pretrained(
        repo, output_loading_info=True, ignore_mismatched_sizes=True
    )
    missing = set(info.get("missing_keys", []))
    head_prefixes = ("classifier", "pre_classifier", "score", "logits_proj")
    has_head_weights = not any(k.split(".")[0] in head_prefixes for k in missing)

    return {
        "arch_says_sequence_classification": arch_flag,
        "config_says_classification_task": task_flag,
        "loaded_class_is": model.__class__.__name__,
        "checkpoint_has_head_weights": has_head_weights,
        "num_labels": getattr(cfg, "num_labels", None),
        "problem_type": getattr(cfg, "problem_type", None),
    }

if __name__ == "__main__":
    path = "/fs-computility/llm_fudan/shared/models/Qwen2.5-Math/Qwen2.5-Math-PRM-7B"
    check_dict = classify_repo(path)
    print(check_dict)