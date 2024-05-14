from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor

multiple_choice = True

def aok_vqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]

def aok_vqa_doc_to_text(doc, model_specific_prompt_kwargs=None):
    if multiple_choice:
        question, choices = doc["question"], doc["choices"]
        len_choices = len(choices)
        options = [chr(ord("A") + i) for i in range(len_choices)]
        choices_str = "\n".join([f"{option}. {choice}" for option, choice in zip(options, choices)])
        if model_specific_prompt_kwargs["format"] == "default":
            post_prompt = model_specific_prompt_kwargs["post_prompt"]
            pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
            return f"{pre_prompt}{question}\n{choices_str}{post_prompt}"
        elif model_specific_prompt_kwargs["format"] == "qwen_vl":
            prompt = "Question: {}\nOptions: {}"
            prompt = prompt.format(question, choices_str)
            return prompt
        else:
            raise ValueError(f"Unknown prompt format: {model_specific_prompt_kwargs}")
    else:
        question = doc["question"]
        if model_specific_prompt_kwargs is None:
            model_specific_prompt_kwargs = {}
        pre_prompt = ""
        post_prompt = ""
        if "pre_prompt" in model_specific_prompt_kwargs:
            pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
        if "post_prompt" in model_specific_prompt_kwargs:
            post_prompt = model_specific_prompt_kwargs["post_prompt"]
        return f"{pre_prompt}{question}{post_prompt}"

def aok_vqa_doc_to_target(doc):
    if multiple_choice:
        len_choices = len(doc["choices"])
        options = [chr(ord("A") + i) for i in range(len_choices)]
        return options[doc["correct_choice_idx"]]
    else:
        return doc["direct_answers"]

def aok_vqa_process_results(doc, result):

    # if multiple_choice is False:
    #     dataset = {k:v for k,v in dataset.items() if v['difficult_direct_answer'] is False}


    eval_ai_processor = EvalAIAnswerProcessor()
    assert len(result) == 1, f"The result should be a list of length 1, but got {len(result)}."
    pred = result[0]
    target = aok_vqa_doc_to_target(doc) # A,B,... multiple_choice, else list direct_answers 
    choices = [chr(ord("A") + i) for i in range(len(doc['choices']))]
    accuracy = 0

    ## Multiple Choice setting
    if multiple_choice:
        if pred == target:
            accuracy = 1.0
        # pattern: ^[A-Z]\. .*
        if len(pred) >= 2 and pred[0].isupper() and pred[1] == ".":
            accuracy = 1.0 if pred[0] == target else 0.0
    ## Direct Answer setting
    else:
        pred = eval_ai_processor(pred)
        num_match = sum([pred == eval_ai_processor(da) for da in target])
        accuracy = min(1.0, num_match / 3.0)


    options = [chr(ord("A") + i) for i in range(len(doc["choices"]))]

    return {"exact_match": accuracy, "lave": {"exact_match": accuracy,
                                              "refs": [target],
                                              "answer": pred,
                                              "question": doc["question"] + " " + ", ".join([f"{option}. {choice}" for option, choice in zip(options, doc["choices"])]),
                                              "choices": choices}}