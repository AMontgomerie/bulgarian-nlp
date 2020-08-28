from models.taggers import POSTagger, NERTagger
from transformers import RobertaTokenizerFast
import re

class TextAnnotator():
    def __init__(self, use_gpu=True):
        tokenizer = RobertaTokenizerFast.from_pretrained('iarfmoose/roberta-small-bulgarian')
        self.pos_tagger = POSTagger(use_gpu=use_gpu, tokenizer=tokenizer)
        self.ner_tagger = NERTagger(use_gpu=use_gpu, tokenizer=tokenizer)
        self.entity_types = {
            'PRO': 'PRODUCT',
            'PER': 'PERSON',
            'ORG': 'ORGANISATION',
            'LOC': 'LOCATION',
            'EVT': 'EVENT'
        }
    
    def __call__(self, text):
        pos_tags = self.pos_tagger.generate_tags(text)
        ner_tags = self.ner_tagger.generate_tags(text)
        tokens = self._split_tokens(text)
        assert len(tokens) == len(pos_tags)
        assert len(tokens) == len(ner_tags)
        annotation = {}
        annotation['tokens'] = [
            {'text': token[0], 'pos': token[1], 'entity': token[2]}
            for token in zip(tokens, pos_tags, ner_tags)
        ]
        annotation['entities'] = self._construct_entities(tokens, ner_tags)
        return annotation
    
    def get_device(self):
        print(self.pos_tagger.device)
    
    def _split_tokens(self, text):
        text = re.sub('([,.:;?!\()""''])', r' \1 ', text)
        text = re.sub('\s{2,}', ' ', text)
        text = [token for token in text.split(' ') if len(token) > 0]
        return text
    
    def _construct_entities(self, tokens, tags):
        entities = []
        entity_types = []
        current_entity = []
        for item in zip(tokens, tags):
            if 'B' in item[1]:
                if len(current_entity) > 0:
                    entities.append(' '.join(current_entity))
                current_entity = [item[0]]
                entity_types.append(self._get_entity_type(item[1]))
            elif 'I' in item[1]:
                current_entity.append(item[0])
        if len(current_entity) > 0:
            entities.append(' '.join(current_entity))       
        
        return [
            {'text': entity[0], 'type': entity[1]} 
            for entity in zip(entities, entity_types)
        ]
    
    def _get_entity_type(self, tag):
        if len(tag.split('-')) <= 1:
            return None
        tag = tag.split('-')[1]
        entity_type = self.entity_types[tag]
        return entity_type
