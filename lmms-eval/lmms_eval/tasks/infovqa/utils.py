import json
import os
import logging


from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

lmms_logger = logging.getLogger("lmms-eval")


def infovqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def infovqa_doc_to_text(doc, model_specific_prompt_kwargs):
    question = doc["question"]
    pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
    post_prompt = model_specific_prompt_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"

from lmms_eval.api.metrics import anls
def infovqa_val_process_results(doc, results):
    anls_score = anls(doc["answers"], predictions=results)["anls"]
    return {"anls": anls_score,
        "lave": {"exact_match": anls_score, "refs": doc["answers"], "answer": results[0], "question": doc["question"]}
    }

def infovqa_test_process_results(doc, results):
    pred = results[0]
    questionId = doc["questionId"]
    return {"submission": {"questionId": int(questionId), "answer": pred}}


def infovqa_test_aggregate_results(results, args):
    # save results as json
    file = generate_submission_file("infovqa_test_for_submission.json", args)
    with open(file, "w") as f:
        json.dump(results, f)
    lmms_logger.info(f"Results saved to {file}")
