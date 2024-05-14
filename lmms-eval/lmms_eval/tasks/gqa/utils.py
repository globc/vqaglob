from datasets import load_dataset
import evaluate

GQA_RAW_IMAGE_DATASET = None
GQA_ID2IMAGE = None


def gqa_doc_to_visual(doc):
    global GQA_RAW_IMAGE_DATASET
    global GQA_ID2IMAGE
    if GQA_RAW_IMAGE_DATASET is None:
        GQA_RAW_IMAGE_DATASET = load_dataset("lmms-lab/GQA", "testdev_balanced_images", split="testdev", token=True)
        GQA_ID2IMAGE = {}
        for row in GQA_RAW_IMAGE_DATASET:
            GQA_ID2IMAGE[row["id"]] = row["image"].convert("RGB")
    image = GQA_ID2IMAGE[doc["imageId"]]
    return [image]


def gqa_doc_to_text(doc, model_specific_prompt_kwargs):
    question = doc["question"]
    pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
    post_prompt = model_specific_prompt_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"


exact_match = evaluate.load("exact_match")

def gqa_process_results(doc, result):
    accuracy = exact_match.compute(
        predictions=[result[0]],
        references=[doc["answer"]],
        ignore_case=True,
        ignore_punctuation=True)["exact_match"]
    
    return {
        "exact_match": accuracy,
        "lave": {"exact_match": accuracy, "refs": [doc["answer"]], "answer": result[0], "question": doc["question"]}
    }
