import torch
from transformers import AutoTokenizer, T5Tokenizer
from mm_cot.model import T5ForMultimodalGeneration
from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

class MMCOTRationale:

    rationale_model_dir = "/vqaglob/mm_cot/models/mm-cot-large-rationale"
    model = None
    tokenizer = None
    vit_model = None
    config = None
    transform = None

    def load(self):

        self.vit_model = timm.create_model("vit_large_patch32_384", pretrained=True, num_classes=0)
        self.vit_model.eval()
        self.config = resolve_data_config({}, model=self.vit_model)
        self.transform = create_transform(**self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.rationale_model_dir)
        self.model = T5ForMultimodalGeneration.from_pretrained(self.rationale_model_dir, patch_size=(145, 1024))
            
    def run(self, input_image, input_text):

        if self.model is None:
            self.load()

        with torch.no_grad():
            input = self.transform(input_image).unsqueeze(0)
            feature = self.vit_model.forward_features(input)
            image_features = torch.cat([feature.detach().cpu()]).cpu()

        source = self.tokenizer.batch_encode_plus(
                [input_text],
                max_length=512,
                pad_to_max_length=True,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
        source_ids = source["input_ids"]
        source_mask = source["attention_mask"]
        rationale = self.model.generate(
            input_ids=source_ids,
            attention_mask=source_mask,
            image_ids=image_features,
            max_length=512,
            num_beams=1,
            do_sample=False
        )

        rationale = self.tokenizer.batch_decode(rationale, skip_special_tokens=True)[0]
        while "Solution: " in rationale:
            rationale = rationale.replace("Solution: ", "")
        return rationale