def sqa_doc_to_text(doc, model_specific_prompt_kwargs=None):
    context, question, choices = doc["hint"], doc["question"], doc["choices"]
    len_choices = len(choices)
    options = [chr(ord("A") + i) for i in range(len_choices)]
    choices_str = "\n".join([f"{option}. {choice}" for option, choice in zip(options, choices)])
    if model_specific_prompt_kwargs["format"] == "default":
        if context:
            context = f"Context: {context}\n"

        post_prompt = model_specific_prompt_kwargs["post_prompt"]
        pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
        return f"{pre_prompt}{context}{question}\n{choices_str}{post_prompt}"
    elif model_specific_prompt_kwargs["format"] == "qwen_vl":
        prompt = "Context: {}\nQuestion: {}\nOptions: {}"
        context = context if context else "N/A"
        prompt = prompt.format(context, question, choices_str)
        return prompt
    else:
        raise ValueError(f"Unknown prompt format: {model_specific_prompt_kwargs}")


def sqa_doc_to_visual(doc):
    if doc["image"] is None:
        return []
    return [doc["image"].convert("RGB")]


def sqa_doc_to_target(doc):
    len_choices = len(doc["choices"])
    options = [chr(ord("A") + i) for i in range(len_choices)]
    return options[doc["answer"]]


def sqa_process_results(doc, results):
    # I know this is weird, but it's how llava parse it.
    target = sqa_doc_to_target(doc)
    choices = [chr(ord("A") + i) for i in range(len(doc["choices"]))]
    pred = results[0]

    if pred == target:
        accuracy = 1.0
    # pattern: ^[A-Z]\. .*
    elif len(pred) >= 2 and pred[0].isupper() and pred[1] == ".":
        result = 1.0 if pred[0] == target else 0.0
        accuracy = result
    else:
        accuracy = 0.0

    options = [chr(ord("A") + i) for i in range(len(doc["choices"]))]

    return {"exact_match": accuracy, "lave": {"exact_match": accuracy,
                                              "refs": [target],
                                              "answer": pred,
                                              "question": doc["question"] + " " + ", ".join([f"{option}. {choice}" for option, choice in zip(options, doc["choices"])]),
                                              "choices": choices}}
