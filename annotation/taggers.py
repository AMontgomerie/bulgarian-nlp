import torch
import numpy as np
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List, Mapping, Tuple


class Tagger:
    """Base class with generic tagging methods. Should not be instantiated
    directly.
    """

    def __init__(self, model_name: str) -> None:
        self.max_length = 128
        self._set_device()
        self._init_model(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _set_device(self) -> None:
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def _init_model(self, model_name: str) -> AutoModelForTokenClassification:
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def generate_tags(self, input_text: str) -> List[str]:
        processed_text = self._preprocess_punctuation(input_text)
        tokenized_text = self._tokenize_text(processed_text)
        predicted_labels = self._generate_prediction(
            tokenized_text["input_ids"],
            tokenized_text["attention_mask"],
            tokenized_text["offset_mapping"],
        )
        return [self.id_to_tag[_id] for _id in predicted_labels]

    def _preprocess_punctuation(self, text: str) -> str:
        text = text.replace("...", ".")
        text = text.replace("..", ".")
        text = re.sub('([,.:;?!\()""' "])", r" \1 ", text)
        text = re.sub("\s{2,}", " ", text)
        return text

    def _tokenize_text(self, text: str) -> Mapping[str, torch.Tensor]:
        tokenized_text = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        tokenized_text["input_ids"] = tokenized_text["input_ids"].to(self.device)
        tokenized_text["attention_mask"] = tokenized_text["attention_mask"].to(
            self.device
        )
        tokenized_text["offset_mapping"] = tokenized_text["offset_mapping"].squeeze()
        return tokenized_text

    @torch.no_grad()
    def _generate_prediction(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        offset_mapping: torch.Tensor,
    ) -> List[int]:
        output = self.model(input_ids, attention_mask)
        preds = np.argmax(output[0].cpu(), axis=2)
        relevant = self._get_relevant_labels(offset_mapping)
        predicted_labels = preds[0][relevant == True].tolist()
        return predicted_labels

    def _get_relevant_labels(self, offset_mapping: torch.Tensor) -> np.array:
        relevant_labels = np.zeros(len(offset_mapping), dtype=int)

        for i in range(1, len(offset_mapping) - 1):
            if offset_mapping[i][1] != offset_mapping[i + 1][0]:
                if not self._ignore_mapping(offset_mapping[i]):
                    relevant_labels[i] = 1
        return relevant_labels

    def _ignore_mapping(self, mapping: Tuple[int]) -> bool:
        return mapping[0] == mapping[1]


class POSTagger(Tagger):
    """Tagger which applies Part Of Speech Tags."""

    def __init__(self) -> None:
        super().__init__("iarfmoose/roberta-small-bulgarian-pos")
        self.tag_to_id = {
            "ADJ": 0,
            "ADP": 1,
            "PUNCT": 2,
            "ADV": 3,
            "AUX": 4,
            "SYM": 5,
            "INTJ": 6,
            "CCONJ": 7,
            "X": 8,
            "NOUN": 9,
            "DET": 10,
            "PROPN": 11,
            "NUM": 12,
            "VERB": 13,
            "PART": 14,
            "PRON": 15,
            "SCONJ": 16,
        }
        self.id_to_tag = {self.tag_to_id[tag]: tag for tag in self.tag_to_id}


class NERTagger(Tagger):
    """Tagger which finds Named Entities."""

    def __init__(self) -> None:
        super().__init__("iarfmoose/roberta-small-bulgarian-ner")
        self.tag_to_id = {
            "O": 0,
            "I-PRO": 1,
            "I-PER": 2,
            "I-ORG": 3,
            "I-LOC": 4,
            "I-EVT": 5,
            "B-PRO": 6,
            "B-PER": 7,
            "B-ORG": 8,
            "B-LOC": 9,
            "B-EVT": 10,
        }
        self.id_to_tag = {self.tag_to_id[tag]: tag for tag in self.tag_to_id}
