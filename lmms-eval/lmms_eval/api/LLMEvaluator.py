PROMPTS = {
    "contextual": """You are ImageTaskEvaluatorGPT, an expert language model at judging whether or not a response adequately addresses an instruction in the context of an image. More specifically, you will be given the following:

1. An instruction: This is a question, an imperative request, or something similar about the image which requires a response.
2. A ground-truth response: This is the ground-truth response to the instruction in the context of the image annotated by the human annotator.
3. A predicted response: This response attempts to address the instruction in the context of the image without having access to the ground-truth response.

Your job is judge whether the predicted response is correct given the ground-truth response and the instruction.
 
Some things to remember:
- Even though you are just a language model, the instructions mostly require an objective answer i.e., the ground-truth response and instruction should be sufficient for you to judge the correctness of the predicted response. You do not need to have access to the complete image description.
- You are capable of judging response quality, accounting for important factors like correctness, relevance, fluency, specificity, etc. 
- You think step-by-step, and ultimately respond with your "Judgement: " as "Yes" or "No". Here, "Yes" implies that the predicted response is correct according to you, and "No" implies that the predicted response is not correct.
- Many times the predicted responses provide long explanations for their decision. In such cases, focus on whether the ground-truth response can be inferred from the predicted response or not. 

Instruction: {question}
Ground-truth Response: {ref_string}
Predicted Response: {pred}""",
    "glob": """You are VQAEvaluatorGPT. Given a question about an image and human reference answers (which may contain errors and inconsistencies), you are an expert at judging whether a candidate answer is correct. Think step by step and answer with 'yes' or 'no' followed by your reasoning.

Examples:

Question: 'Whats the weather like?'
Reference answers: 'bright', 'unanswarable', 'cloudy', 'bright and sunny', 'clear', 'sunny', 'sunny', 'cloudy', 'sunny', 'cloudy'
Candidate answer: 'Unanswarable'
Is the candidate answer correct?
Output: Yes, it seems there is much disagreement whether it is sunny or cloudy 

Question: 'What are the people in the picture doing?'
Reference answers: 'sitting', 'sitting', 'sitting', 'sitting'
Candidate answer: 'they are resting'
Is the candidate answer correct?
Output: Yes, it is common that people who are sitting are resting.

Question: 'What color are the base tiles?'
Reference answers: 'beige', 'beige', 'beige', 'tan', 'tan', 'brown', 'tan', 'tan', 'ten', 'white'
Candidate answer: 'The base tiles are brown.'
Is the candidate answer correct?
Output: Yes, 'brown' is similar to 'beige' or 'tan'

Question: 'Is the book on the table?'
Reference answers: 'no', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes'
Candidate answer: 'No, the book is on a shelf.'
Is the candidate answer correct?
Output: No, it seems the reference answers agree that the book is indeed on the table.

Question: 'How many people are in the picture?'
Reference answers: 'four', 'three', 'three', 'three', 'two', 'two'
Candidate answer: 'a few'
Is the candidate answer correct?
Output: No, 'a few' is too unspecific.

Now judge this:

Question: '{question}'
Reference answers: {ref_string}
Candidate answer: '{pred}'
Is the candidate answer correct?
Output:"""}

class LLMEvaluator:

    model = None
    tokenizer = None

    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained("Upstage/SOLAR-10.7B-Instruct-v1.0", padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
                "Upstage/SOLAR-10.7B-Instruct-v1.0",
                device_map="cuda",
                torch_dtype=torch.float16,
            )
            
    def run(self, batch):

        if self.model is None:
            self.load()

        prompts = []
        prompt = PROMPTS[batch[0]["prompt"]] if "prompt" in batch[0] else PROMPTS["glob"]
        for item in batch:

            ref_string = ", ".join(f"'{r if isinstance(r, str) else r['answer']}'" for r in item['refs'])

            # Chain-of-Thought Prompting, https://arxiv.org/pdf/2201.11903.pdf (Add CoT to demonstrations e.g. A: Let's think step by step)

            prompts.append(prompt.format(question=item["question"], ref_string=ref_string, pred=item["answer"]))

        model_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=2)
        generated_ids = generated_ids[:, model_inputs["input_ids"].shape[1] :]
        outputs = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


        scores = []
        for output, prompt, item in zip(outputs, prompts, batch):

            if not (output.lower().startswith("yes") or output.lower().startswith("no")):
                print("ERROR: Invalid response")
            lmm_score = int(output.lower().startswith("yes"))

            scores.append(max(lmm_score, item['exact_match']))   

            # print(prompt.split("Now judge this:")[-1] + output + " |Extracted rating: " + str(lmm_score))

        return scores
    
def eval():
    import datasets
    from lmms_eval.tasks.vqav2.utils import vqav2_process_results_val
    from lmms_eval.tasks.ok_vqa.utils import ok_vqa_process_results
    from scipy.stats import spearmanr
    from lmms_eval.api.metrics import lave

    human_scores = []
    mom_scores = []


    human_feedback = datasets.load_dataset(
            path="mair-lab/lave-human-feedback",
            download_mode=datasets.DownloadMode.REUSE_DATASET_IF_EXISTS,
            token=True,
            split="test"
        )
    
    human_feedback = human_feedback.rename_column("references", "answers")
    human_feedback = human_feedback.rename_column("qid", "question_id")

    human_feedback = [doc for doc in human_feedback]

    for doc in human_feedback:
        doc["answers"] = [{"answer": answer, "answer_id": i} for i,answer in enumerate(doc["answers"])]
        if doc["dataset"] in ["vqav2", "vgqa"]:
            processed = vqav2_process_results_val(doc, [doc["prediction"]])
        elif doc["dataset"] == "okvqa":
            processed = ok_vqa_process_results(doc, [doc["prediction"]])

        processed["lave"]["exact_match"] = 0
        doc["mom_score"] = lave([processed["lave"]])

    grouped = {}
    for doc in human_feedback:
        # doc["human_score"] = 1 if doc["human_score"] == 0.5 else doc["human_score"]
        key = (doc['dataset'], doc['model'])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(doc)

    correlation_results = {}
    for group_key, group_data in grouped.items():
        metric_scores = [doc['mom_score'] for doc in group_data]
        human_scores = [doc['human_score'] for doc in group_data]
        spearman_corr, p_value = spearmanr(metric_scores, human_scores)
        correlation_results[group_key] = {'Spearman Correlation': spearman_corr, 'P-value': p_value}

    # Calculate overall correlation
    overall_mom_scores = [doc['mom_score'] for doc in human_feedback]
    overall_human_scores = [doc['human_score'] for doc in human_feedback]
    overall_corr, overall_p_value = spearmanr(overall_mom_scores, overall_human_scores)

    print("Spearman Correlation for each group:")
    for group, result in correlation_results.items():
        print(f"Model: {group[0]}, Question Type: {group[1]}")
        print("Spearman Correlation:", result['Spearman Correlation'])
        print("P-value:", result['P-value'])
        print()

    print("Overall Spearman Correlation:", overall_corr)
    print("Overall P-value:", overall_p_value)

if __name__ == "__main__":
    eval()
        
        

        

