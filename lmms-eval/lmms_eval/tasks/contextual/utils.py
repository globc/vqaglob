import requests
from PIL import Image
from io import BytesIO

def contextual_doc_to_visual(doc):
    response = requests.get(doc["image_url"])
    return [Image.open(BytesIO(response.content)).convert("RGB")]

def contextual_doc_to_text(doc, model_specific_prompt_kwargs):
    instruction = doc["instruction"]
    pre_prompt = model_specific_prompt_kwargs["pre_prompt"]
    post_prompt = model_specific_prompt_kwargs["post_prompt"]
    return f"{pre_prompt}{instruction}{post_prompt}"

def contextual_process_results(doc, result):
    return {"lave": {"exact_match": 0, "refs": [doc["response"]], "answer": result[0], "question": doc["instruction"]}}