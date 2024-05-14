## Installation
Run the following commands to build and run the docker container:
```
docker build -t globc/vqaglob .
docker run --gpus all -it --ipc=host globc/vqaglob bash
```

To run MultimodalCoT download mm-cot-large-rationale (see [mm-cot](https://github.com/amazon-science/mm-cot) for instructions) and move it to `\mm_cot\models\mm-cot-large-rationale`


## Instructions
Modify `/lmms-eval/vqa_eval.yaml` and `/lmms-eval/lmms_eval/tasks/glob/_default_template_0_yaml` as needed then run
```
accelerate launch --num_processes=1 -m lmms_eval --config lmms-eval/vqa_eval.yaml
```

See `/vqaglob/lmms-eval/lmms_eval/api/templates.py` for a list of available templates.

## Acknowledgments
The codebase is build on top of [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval), [llava](https://github.com/haotian-liu/LLaVA) and [mm-cot](https://github.com/amazon-science/mm-cot) repositories. We thank the authors for their amazing work.