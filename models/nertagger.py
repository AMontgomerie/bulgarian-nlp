from transformers import RobertaForTokenClassification, RobertaTokenizerFast
import torch
import re
import numpy as np

MAX_LEN = 128

tag_to_id = {
    'O': 0,
    'I-PRO': 1,
    'I-PER': 2,
    'I-ORG': 3,
    'I-LOC': 4,
    'I-EVT': 5,
    'B-PRO': 6,
    'B-PER': 7,
    'B-ORG': 8,
    'B-LOC': 9,
    'B-EVT': 10
}

id_to_tag = {tag_to_id[tag]: tag for tag in tag_to_id}

class NERTagger():
    def __init__(self, use_gpu=True):
        self._set_device(use_gpu)
        MODEL_NAME = 'iarfmoose/roberta-small-bulgarian-ner'
        self.tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)
        self.model = RobertaForTokenClassification.from_pretrained(MODEL_NAME)
        self.model.to(self.device)
    
    def _set_device(self, use_gpu):
        if use_gpu:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                print("GPU unavailable, switching to CPU.")
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')

    def generate_tags(self, text):
        self.model.eval()
        text = self._preprocess_punctuation(text)
        tokenized_text = self.tokenizer(
            text, 
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            add_special_tokens=True,
            return_offsets_mapping=True,
            return_tensors='pt'
        )
        tokenized_text['input_ids'] = tokenized_text['input_ids'].to(self.device)
        tokenized_text['attention_mask'] = tokenized_text['attention_mask'].to(self.device)
        with torch.no_grad():
            output = self.model(
                input_ids=tokenized_text['input_ids'], 
                attention_mask=tokenized_text['attention_mask']
            )
        preds = np.argmax(output[0].cpu(), axis=2)
        relevant = self._get_relevant_labels(tokenized_text['offset_mapping'])
        predicted_labels = preds[0][relevant == True].tolist()
        return [id_to_tag[id] for id in predicted_labels]

    def _preprocess_punctuation(self, text):
        text = text.replace('...', '.')
        text = text.replace('..', '.')
        text = re.sub('([,.:;?!\()""''])', r' \1 ', text)
        text = re.sub('\s{2,}', ' ', text)
        return text

    def _get_relevant_labels(self, offset_mapping):
        relevant_labels = np.zeros(len(offset_mapping), dtype=int)

        for i in range(1, len(offset_mapping) - 1):
            if offset_mapping[i][1] != offset_mapping[i+1][0]:
                if not self._ignore_mapping(offset_mapping[i]):
                    relevant_labels[i] = 1
        return relevant_labels

    def _ignore_mapping(self, mapping):
        return mapping[0] == mapping[1]