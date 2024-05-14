def ai2d_doc_to_text(doc, model_specific_prompt_kwargs=None):
    question, choices = doc["question"], doc["options"]
    len_choices = len(choices)
    post_prompt = model_specific_prompt_kwargs["post_prompt"]
    pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
    if model_specific_prompt_kwargs["prompt_format"] == "mcq":
        options = [chr(ord("A") + i) for i in range(len_choices)]
        choices_str = "\n".join([f"{option}. {choice}" for option, choice in zip(options, choices)])
        return f"{pre_prompt}{question}\n{choices_str}{post_prompt}"
    elif model_specific_prompt_kwargs["prompt_format"] == "qa":
        options = "\n".join(choices)
        return f"{pre_prompt}{question}{options}{post_prompt}"
    else:
        raise ValueError(f"Unknown prompt format: {model_specific_prompt_kwargs['prompt_format']}")


def ai2d_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def ai2d_doc_to_target(doc, model_specific_target_kwargs):
    if model_specific_target_kwargs == "mcq":
        len_choices = len(doc["options"])
        options = [chr(ord("A") + i) for i in range(len_choices)]
        doc["choices"] = options
        doc["target"] = options[int(doc["answer"])]
        return options[int(doc["answer"])]
    elif model_specific_target_kwargs == "qa":
        doc["choices"] = doc["options"]
        doc["target"] = doc["options"][int(doc["answer"])]
        return doc["options"][int(doc["answer"])]

import evaluate
exact_match = evaluate.load("exact_match")

def ai2d_process_results(doc, result, model_specific_processing_kwargs="mcq"):
    target = ai2d_doc_to_target(doc, model_specific_processing_kwargs) # TODO change to qa for Qwen, global prompt kwargs?
    choices = [chr(ord("A") + i) for i in range(len(doc["options"]))] if model_specific_processing_kwargs == "mcq" else doc["options"]
    accuracy = exact_match.compute(
        predictions=[result[0]],
        references=[target],
        ignore_case=True,
        ignore_punctuation=True)["exact_match"]
    
    options = [chr(ord("A") + i) for i in range(len(doc["options"]))]
    return {
        "exact_match": accuracy,
        "lave": {"exact_match": accuracy,
                 "refs": [target],
                 "answer": result[0],
                 "question": doc["question"] + " " + ", ".join([f"{option}. {choice}" for option, choice in zip(options, doc["options"])]),
                 "choices": choices}
    }