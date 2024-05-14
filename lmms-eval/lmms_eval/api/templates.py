NULL = ""

DESCR_EXPERT = "You are VQAGPT, an expert vision language model at answering questions based on a provided image. You may need to look closely at the image and use reasoning and world knowledge to answer, so let's think and go through the image step by step.\n"
DESCR_VICUNA_V1 = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
DESCR_REASON = "Your task is to answer questions based on a given image. First, analyze the image carefully, paying attention to details and relevant information. Then think step by step to solve the problem. Finally, provide your answer and append your reasoning.\n"
DESCR_COT_CHAT_EXPERT = "A chat between a curious user and an expert vision language assistant for answering questions based on a given image. Before answering, the assistant carefully analyzes the image, paying attention to details and relevant information, and thinks step by step to solve the question using common sense and world knowledge. Finally, the assistant provides a helpful answer to the user's question, followed by a detailed explanation.\n"
DESCR_AUTO_COT_CUSTOM = """A chat between a curious user and an artificial intelligence assistant. Given an image, a question based on the image and the correct answer to the question, the assistant gives helpful and detailed step-by-step instructions on how to arrive at the given answer.\n"""

LLAVA_CONTEXT_DEFAULT = "USER: {{ question }}{{ post_prompt }} ASSISTANT: {{ rationale }}{% if rationale %} So the answer is {% endif %}{{ answer }}"
QWEN_CONTEXT_DEFAULT = "{{ question }}{{ post_prompt }} Answer: {{ rationale }}{% if rationale %} So the answer is {% endif %}{{ answer }}" # No space as last token
COT_THINK = "{{ question }}{{ post_prompt }} Answer: Let's think step by step. {{ rationale }}{% if rationale %} So the answer is {% endif %}{{ answer }}"
COT_CHAT_THINK = "USER: {{ question }}{{ post_prompt }} ASSISTANT: Let's think step by step. {{ rationale }}{% if rationale %} So the answer is {% endif %}{{ answer }}"
COT_CHAT_EXPERT = "USER: {{ question }}{{ post_prompt }} ASSISTANT: {{ answer }}{% if rationale %}\nBecause: {% endif %}{{ rationale }}"
AUTO_COT_CUSTOM_CONTEXT = "USER: {{ question }}\nAnswer: {{ answer }}{{ post_prompt }}\nASSISTANT: {{ rationale }}"

DIRECT_DEFAULT = "\nAnswer the question using a single word or phrase."
CHOICE_DEFAULT = "\nAnswer with the option's letter from the given choices directly." # Bad: AI2D + Qwen-VL
UNANSWARABLE_DEFAULT = "\nWhen the provided information is insufficient, respond with 'Unanswerable'." # + DIRECT_DEFAULT for LLaVA
COT_AR = "\nAnswer the question, then give your reasoning."
COT_THINK_AR = "\nThink step by step. Answer the question, then give your reasoning after your answer."
COT_LONG = "\nYour task is to answer the question based on the given image. First, analyze the image carefully, paying attention to details and relevant information. Then think step by step to solve the problem. Finally, provide your answer and append your reasoning."
#  COT_LONG not for post_prompt. Rewrite or set before question

COT_IMAGE = "Let's look closely at the image and think step by step."

SETUPS = {
    "CHAT": {"description": DESCR_VICUNA_V1,
               "context_template": LLAVA_CONTEXT_DEFAULT,
               "task_descriptions": [DIRECT_DEFAULT, CHOICE_DEFAULT, UNANSWARABLE_DEFAULT + DIRECT_DEFAULT]},

    "CHAT_NULL": {"description": DESCR_VICUNA_V1,
               "context_template": LLAVA_CONTEXT_DEFAULT,
               "task_descriptions": [NULL, NULL, NULL]},

    "CHAT_CUSTOM": {"description": DESCR_VICUNA_V1,
               "context_template": LLAVA_CONTEXT_DEFAULT,
               "task_descriptions": [DIRECT_DEFAULT, CHOICE_DEFAULT, UNANSWARABLE_DEFAULT + DIRECT_DEFAULT],
               "skip_format": ["contextual", "ok_vqa_val"]},

    "COT_CHAT_EXPERT": {"description": DESCR_COT_CHAT_EXPERT,
                        "context_template": COT_CHAT_EXPERT,
                        "task_descriptions": [NULL, NULL, NULL] # LLaVA doesn't do what I want
                    },

    "COT_2": {"description": DESCR_VICUNA_V1,
                "context_template": COT_CHAT_EXPERT,
                "task_descriptions": [COT_AR, COT_AR, COT_AR] # LLaVA doesn't do what I want
                    },

    "COT_2.5": {"description": DESCR_VICUNA_V1,
                "context_template": COT_CHAT_EXPERT,
                "task_descriptions": [COT_THINK_AR, COT_THINK_AR, COT_THINK_AR] # LLaVA doesn't do what I want
                    },

    "COT_3": {"description": DESCR_VICUNA_V1,
                "context_template": COT_CHAT_EXPERT,
                "task_descriptions": [COT_LONG, COT_LONG, COT_LONG] # LLaVA doesn't do what I want
                    },
    
    
    "QWEN": {"description": NULL,
              "context_template": QWEN_CONTEXT_DEFAULT,
              "task_descriptions": [NULL, NULL, UNANSWARABLE_DEFAULT]},

    "EXPERT": {"description": DESCR_EXPERT,
                "context_template": QWEN_CONTEXT_DEFAULT,
                "task_descriptions": [NULL, NULL, NULL]},

    "NULL": {"description": NULL,
              "context_template": "{{ question }}{{ post_prompt }}\n{{ answer }}",
              "task_descriptions": [NULL, NULL, NULL]},

    "ANSWER": {"description": NULL,
              "context_template": QWEN_CONTEXT_DEFAULT,
              "task_descriptions": [NULL, NULL, NULL]},

    "FORMAT": {"description": NULL,
                "context_template": QWEN_CONTEXT_DEFAULT,
                "task_descriptions": [DIRECT_DEFAULT, CHOICE_DEFAULT, UNANSWARABLE_DEFAULT + DIRECT_DEFAULT]},

    "COT_BASIC": {"description": NULL,
                   "context_template": COT_THINK,
                   "task_descriptions": [NULL, NULL, NULL]}, # Doesn't output Chain.

    "COT_CHAT_BASIC": {"description": DESCR_VICUNA_V1,
                   "context_template": COT_CHAT_THINK,
                   "task_descriptions": [NULL, NULL, NULL]},
            
    "COT_TASK_AR": {"description": DESCR_REASON,
                  "context_template": QWEN_CONTEXT_DEFAULT,
                  "task_descriptions": [NULL, NULL, NULL]},

    "MM_COT": {"description": NULL,
                "context_template": "{{ question }}{{ post_prompt }}\nSolution: {{ rationale }} So the answer is {{ answer }}",
                "task_descriptions": [NULL, NULL, NULL]},

    "AUTO_COT": {"description": DESCR_VICUNA_V1, # Special: Generate without knowing answer
                   "context_template": COT_CHAT_THINK.replace("{{ answer }}", ""),
                   "task_descriptions": [NULL, NULL, NULL]},

    "AUTO_COT_CUSTOM": {"description": DESCR_AUTO_COT_CUSTOM, # Special: Generate without knowing answer
                   "context_template": AUTO_COT_CUSTOM_CONTEXT,
                   "task_descriptions": [NULL, NULL, NULL]},

    }               # TODO Generate with knowing answer