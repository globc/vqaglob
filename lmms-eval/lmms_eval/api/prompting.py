from .templates import SETUPS
from jinja2 import Template

def get_prompt(setup, context, model, task):
    setup = SETUPS[setup]

    description, context_template, task_descriptions = setup["description"], setup["context_template"], setup["task_descriptions"]
    
    post_prompt = ""
    if "skip_format" in setup:
        if not any(t in task for t in setup["skip_format"]):
            post_prompt = task_descriptions[2] if any(t in task for t in ["vizwiz", "ok_vqa_val"]) else (task_descriptions[1] if any(t in task for t in ["ai2d", "scienceqa", "aok_vqa"]) else task_descriptions[0])
    else:
        post_prompt = task_descriptions[2] if any(t in task for t in ["vizwiz", "ok_vqa_val"]) else (task_descriptions[1] if any(t in task for t in ["ai2d", "scienceqa", "aok_vqa"]) else task_descriptions[0])


    fewshot_delimiter = "</s>" if model == "llava" else "\n\n"


    prompt = description + fewshot_delimiter.join([Template(context_template).render(
        **ctx,
        post_prompt=post_prompt) for ctx in context]).rstrip(" ")
    return prompt

def get_idefics_prompt(setup, context, task, images):
    setup = SETUPS[setup]

    description, context_template, task_descriptions = setup["description"], setup["context_template"], setup["task_descriptions"]

    post_prompt = ""
    if "skip_format" in setup:
        if not any(t in task for t in setup["skip_format"]):
            post_prompt = task_descriptions[2] if any(t in task for t in ["vizwiz", "ok_vqa_val"]) else (task_descriptions[1] if any(t in task for t in ["ai2d", "scienceqa", "aok_vqa"]) else task_descriptions[0])
    else:
        post_prompt = task_descriptions[2] if any(t in task for t in ["vizwiz", "ok_vqa_val"]) else (task_descriptions[1] if any(t in task for t in ["ai2d", "scienceqa", "aok_vqa"]) else task_descriptions[0])


    if len(images) == 1:
        images = [None] * (len(context) - 1) + images

    index = context_template.find("{{ post_prompt }}")

    question_template = Template(str(context_template[:index + len("{{ post_prompt }}")]))
    answer_template = Template(str(context_template[index + len("{{ post_prompt }}"):]))

    prompt = [description] if description is not "" else []
    for image, ctx in zip(images,context):
        prompt.append("\n" + question_template.render(question=ctx["question"], answer=(ctx["answer"] if "answer" in ctx else None), post_prompt=post_prompt).strip(" "))
        if image is not None:
            prompt.append(image)
        
        prompt.append("<end_of_utterance>")

        if "answer" in ctx:
            ctx["answer"] = ctx["answer"] + "<end_of_utterance>"
        prompt.append("\n" + answer_template.render(**ctx).strip(" "))
            
    return prompt