SCIENCEQA = None
AOKVQA = None
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import json
import os
from lmms_eval.api.instance import Instance
import importlib
import base64
from PIL import Image
from io import BytesIO
import numpy as np
import datasets

class ContextSampler:
    def __init__(self, docs, task, fewshot_indices=None, rnd=None, fewshot_config=None) -> None:
        self.rnd = rnd
        assert self.rnd, "must pass rnd to FewShotSampler!"

        self.task = task
        self.config = task._config

        self.fewshot_config = fewshot_config

        self.target_delimiter = self.config.target_delimiter
        self.fewshot_delimiter = self.config.fewshot_delimiter

        self.doc_to_text = self.task.doc_to_text
        self.doc_to_visual = self.task.doc_to_visual
        self.doc_to_target = self.task.doc_to_target
        self.doc_to_choice = self.task.doc_to_choice

        self.docs = docs  # HF dataset split, provided by task._fewshot_docs()
        if fewshot_indices:  # subset few-shot docs from
            self.docs = self.docs.select(fewshot_indices)

    def get_context(self, doc, num_fewshot):
        # draw an extra fewshot sample if using same split as evaluating on
        if self.config.fewshot_split is None:
            self.config.fewshot_split = self.config.test_split
        n_samples = num_fewshot + 1 if self.config.fewshot_split == self.config.test_split else num_fewshot

        # draw `n_samples` docs from fewshot_docs
        if num_fewshot == 0:
            return [],[]
        if self.fewshot_config.get("cot_mode", "").startswith("sample"):
            return self.sample_rationales(num_fewshot)
        elif self.fewshot_config.get("cot_mode", "").startswith("cluster"):
            return self.sample_cluster(num_fewshot, doc)
        

        fewshotex = self.sample(n_samples)

        # get rid of the doc that's the one we're evaluating, if it's in the fewshot
        # TODO: should we just stop people from using fewshot from same split as evaluating?
        selected_docs = [x for x in fewshotex if x != doc][:num_fewshot]
        context_text = []
        context_images = []

        for doc in selected_docs:
            image = self.doc_to_visual(doc)[0]
            question = (self.doc_to_text(doc) if (self.config.doc_to_choice is None or type(self.doc_to_text(doc)) is str) else self.doc_to_choice(doc)[self.doc_to_text(doc)])
            answer = self.doc_to_answer(doc)
            context_text.append({"question": question, "answer": answer})
            context_images.append(image)

        return context_text, context_images

    def doc_to_answer(self, doc):

        if type(self.doc_to_target(doc)) is list:
            if type(self.doc_to_target(doc)[0]) is dict:
                answer = str(self.doc_to_target(doc)[0]["answer"])
            else:
                answer = str(self.doc_to_target(doc)[0])
        else:
            if (self.config.doc_to_choice is None or type(self.doc_to_target(doc)) is str):
                answer = self.doc_to_target(doc)
            else:
                answer = str(self.doc_to_choice(doc)[self.doc_to_target(doc)])
        
        return answer

    def sample(self, n):
        """
        Draw `n` samples from our fewshot docs. This method should be overridden by subclasses.
        """

        return self.docs.shuffle(seed=self.rnd.seed()).select(range(n))
    
    def sample_rationales(self, n):
        import datasets
        import random
        global SCIENCEQA
        global AOKVQA

        context_text = []
        context_images = []

        if SCIENCEQA is None:
            scienceqa = datasets.load_dataset(
                    path="lmms-lab/ScienceQA",
                    name="ScienceQA-IMG",
                    download_mode=datasets.DownloadMode.REUSE_DATASET_IF_EXISTS,
                    token=True,
                    split="test"
                )
            
            if self.fewshot_config.get("cot_mode", "") == "sample_long":
                SCIENCEQA = [doc for doc in scienceqa if (len(doc["solution"]) > 200 and len(doc["solution"]) < 400)]
            else:
                SCIENCEQA = [doc for doc in scienceqa if doc["solution"] != ""]
        
        random.shuffle(SCIENCEQA)

        samples_scienceqa = SCIENCEQA[:int(n / 2)]
        for doc in samples_scienceqa:
            image = doc["image"].convert("RGB")
            question = doc["question"]
            answer = doc["choices"][doc["answer"]]
            rationale = doc["solution"]
            context_text.append({"question": question, "answer": answer, "rationale": rationale})
            context_images.append(image)

        if AOKVQA is None:
            aokvqa = datasets.load_dataset(
                    path="HuggingFaceM4/A-OKVQA",
                    download_mode=datasets.DownloadMode.REUSE_DATASET_IF_EXISTS,
                    token=True,
                    split="train"
                )
            
            if self.fewshot_config.get("cot_mode", "") == "sample_long":
                AOKVQA = [doc for doc in aokvqa if (len(doc["rationales"][0]) > 200 and len(doc["rationales"][0]) < 400)]
            else:
                AOKVQA = [doc for doc in aokvqa]
        
        random.shuffle(AOKVQA)
        samples_aokvqa = AOKVQA[:int(n / 2)]
        for doc in samples_aokvqa:
            image = doc["image"].convert("RGB")
            question = doc["question"]
            answer = doc["choices"][doc["correct_choice_idx"]]
            rationale = doc["rationales"][0]
            context_text.append({"question": question, "answer": answer, "rationale": rationale})
            context_images.append(image)

        return context_text, context_images

    def sample_cluster(self, num_fewshot, doc):
        # "Your task is to generate a rationale given an image, a question based on the image and an answer."
        # if "rationales" in doc
        from ..models import AVAILABLE_MODELS
        demos_path = "/vqaglob/lmms-eval/lmms_eval/api/demos/" + self.task.config.task

        num_clusters = num_fewshot + 1 if self.config.fewshot_split == self.config.test_split else num_fewshot
        
        if not os.path.exists(demos_path):
            encoder = SentenceTransformer("all-MiniLM-L6-v2")

            corpus = []

            for doc in self.docs:
                c_question = (self.doc_to_text(doc) if (self.config.doc_to_choice is None or type(self.doc_to_text(doc)) is str) else self.doc_to_choice(doc)[self.doc_to_text(doc)])
                corpus.append(c_question)
            
            corpus_embeddings = encoder.encode(corpus)

            # Perform kmean clustering
            clustering_model = KMeans(n_clusters=num_clusters, random_state=self.rnd.seed())
            clustering_model.fit(corpus_embeddings)
            cluster_assignment = clustering_model.labels_

            clustered_sentences = [[] for i in range(num_clusters)]

            dist = clustering_model.transform(corpus_embeddings)
            clustered_dists = [[] for i in range(num_clusters)]
            clustered_idx = [[] for i in range(num_clusters)]
            for sentence_id, cluster_id in enumerate(cluster_assignment):
                clustered_sentences[cluster_id].append(corpus[sentence_id])
                clustered_dists[cluster_id].append(dist[sentence_id][cluster_id])
                clustered_idx[cluster_id].append(sentence_id)

            sample_docs_idx = []

            for i in range(len(clustered_dists)):
                tmp = list(map(list, zip(range(len(clustered_dists[i])), clustered_dists[i])))
                top_min_dist = sorted(tmp, key=lambda x: x[1], reverse=False)
                # if not args.sampling == "center":
                #     random.shuffle(top_min_dist)
                for element in top_min_dist:
                    min_idx = element[0]
                    print(clustered_idx[i][min_idx])
                    sample_docs_idx.append(clustered_idx[i][min_idx])
                    break

            sample_docs = self.docs.select(sample_docs_idx)
            sample_docs.save_to_disk(demos_path)

        setup = self.fewshot_config.get("cluster_cot_prompt", "AUTO_COT")
        if not os.path.exists(demos_path + "-" + self.task.model_name + "-" + setup + ".json"):
            sample_docs = datasets.load_from_disk(demos_path)

            rationales = []
            for doc in sample_docs:
                if "scienceqa" in self.task.config.task:
                    rationales.append(doc["solution"])
                elif "aok_vqa" in self.task.config.task:
                    rationales.append(doc["rationales"][0])
                else:
                    from lmms_eval.evaluator import lm
                    prev_setup = lm.setup
                    prev_fewshot_images = lm.fewshot_images
                    lm.setup = setup
                    lm.fewshot_images = True

                    rat_gen_context_text = []
                    rat_gen_context_images = []
                    if setup != "AUTO_COT":
                        sample_scienceqa = datasets.load_from_disk("/vqaglob/lmms-eval/lmms_eval/api/demos/scienceqaglob")
                        for sqa_doc in sample_scienceqa:
                            image = sqa_doc["image"].convert("RGB")
                            question = sqa_doc["question"]
                            answer = sqa_doc["choices"][sqa_doc["answer"]]
                            rationale = sqa_doc["solution"]
                            rat_gen_context_text.append({"question": question, "answer": answer, "rationale": rationale})
                            rat_gen_context_images.append(image)
                    
                    question = (self.doc_to_text(doc) if (self.config.doc_to_choice is None or type(self.doc_to_text(doc)) is str) else self.doc_to_choice(doc)[self.doc_to_text(doc)])
                    answer = self.doc_to_answer(doc)
                    rat_gen_context_text.append({"question": question, "answer": answer})

                    def doc_to_visual(x):
                        return rat_gen_context_images + self.doc_to_visual(doc)
                    
                    arguments = (rat_gen_context_text, {}, doc_to_visual, 0, self.task.config.task, self.config.fewshot_split)

                    rationale = lm.generate_until([Instance(request_type="generate_until", arguments=arguments, idx=0)])
                    if isinstance(rationale, list):
                        rationale = rationale[0]
                    rationales.append(rationale)

                    lm.setup = prev_setup
                    lm.fewshot_images = prev_fewshot_images
            
            json.dump(rationales, open(demos_path + "-" + self.task.model_name + "-" + setup + ".json", 'w'))

        sample_docs = datasets.load_from_disk(demos_path)
        rationales = json.load(open(demos_path + "-" + self.task.model_name + "-" + setup + ".json", 'r'))
        context_text = []
        context_images = []

        selected_docs = [{"doc": doc_x, "rationale": rationale} for doc_x,rationale in zip(sample_docs,rationales) if doc_x != doc][:num_fewshot]
        for sel_doc in selected_docs:
            rationale = sel_doc["rationale"]
            doc = sel_doc["doc"]

            image = self.doc_to_visual(doc)[0]
            question = (self.doc_to_text(doc) if (self.config.doc_to_choice is None or type(self.doc_to_text(doc)) is str) else self.doc_to_choice(doc)[self.doc_to_text(doc)])
            answer = self.doc_to_answer(doc)

            if self.fewshot_config.get("cot_mode", "") == "cluster":
                context_text.append({"question": question, "answer": answer})
            elif self.fewshot_config.get("cot_mode", "") == "cluster_cot":
                context_text.append({"question": question, "answer": answer, "rationale": rationale})
            
            context_images.append(image)
            
        return context_text, context_images

class FirstNSampler(ContextSampler):
    def sample(self, n) -> None:
        """
        Draw the first `n` samples in order from the specified split.
        Used for tasks with "canonical" ordered fewshot examples, such as MMLU and CMMLU.
        """
        assert n <= len(self.docs), f"Error: number of fewshot samples requested exceeds the {len(self.docs)} that are available."
        return self.docs[:n]


class BalancedSampler(ContextSampler):
    def sample(self, n) -> None:
        """
        TODO: this should return approximately class-balanced samples from our fewshot examples.
        TODO: what order should they be in? maybe random?
        """

        pass


class ManualSampler(ContextSampler):
    def sample(self, n) -> None:
        """ """
        pass


SAMPLER_REGISTRY = {
    "default": ContextSampler,
    "first_n": FirstNSampler,
}


def get_sampler(name):
    try:
        return SAMPLER_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Attempted to use contextsampler '{name}', but no sampling strategy for this name found! Supported model names: {', '.join(SAMPLER_REGISTRY.keys())}")
