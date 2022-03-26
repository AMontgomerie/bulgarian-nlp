# bulgarian-nlp

Part-Of-Speech tagging and Named Entity Recognition for Bulgarian.

## Usage

First clone the repository and make sure the transformers library is installed

```
git clone https://github.com/amontgomerie/bulgarian-nlp
cd bulgarian-nlp
```

### Text Annotation

POS and NER tags can be generated at the same time using the `TextAnnotator`.

```python
from annotation.annotators import TextAnnotator

annotator = TextAnnotator()
annotator('България е член на ЕС.')
```

This returns a dictionary containing `tokens` and `entities`.

- `tokens` contains:
  - `text`: the word or punctuation mark
  - `pos`: POS tag
  - `entity`: IOB tag
- `entities` contains:
  - `text`: an entity (may be made up of more than one token)
  - `type`: the type of entity

The output of the above example is:

```
{'entities': [{'text': 'България', 'type': 'LOCATION'},
  {'text': 'ЕС', 'type': 'ORGANISATION'}],
 'tokens': [{'entity': 'B-LOC', 'pos': 'PROPN', 'text': 'България'},
  {'entity': 'O', 'pos': 'AUX', 'text': 'е'},
  {'entity': 'O', 'pos': 'NOUN', 'text': 'член'},
  {'entity': 'O', 'pos': 'ADP', 'text': 'на'},
  {'entity': 'B-ORG', 'pos': 'PROPN', 'text': 'ЕС'},
  {'entity': 'O', 'pos': 'PUNCT', 'text': '.'}]}
```

If only one type of tag is required (POS or NER), the relevant tagger can be instantiated individually.

### Part-Of-Speech tagging

POS tags can be generated like this:

```python
from annotation.taggers import POSTagger

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
from annotation.taggers import NERTagger

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
