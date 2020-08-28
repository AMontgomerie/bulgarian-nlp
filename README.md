# bulgarian-nlp

Part-Of-Speech tagging and Named Entity Recognition for Bulgarian.

## Usage

First clone the repository and make sure the transformers library is installed
```
git clone https://github.com/amontgomerie/bulgarian-nlp
cd bulgarian-nlp
```

### Part-Of-Speech tagging
Now POS tags can be generated like this:
```python
from models.textannotation import POSTagger

pos_tagger = POSTagger()
pos_tagger.generate_tags('Аз сьм мьж.')
```
Which will generate:
```
['PRON', 'VERB', 'NOUN', 'PUNCT']
```
For more information about the POS tags, see https://universaldependencies.org/u/pos/

### Named Entity Recognition
NER tags can be generated using:
```python
from models.textannotation import NERTagger

ner_tagger = NERTagger()
ner_tagger.generate_tags('България е член на ЕС в Европа.')
```
Which outputs:
```
['B-LOC', 'O', 'O', 'O', 'B-ORG', 'O', 'B-LOC', 'O']
```
For more information about the NER tag format, see https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)

## Known Issues
- The POS tagger sometimes misclassifies verbs.
- The NER tagger is not very good at identifying products or events.
