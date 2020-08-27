# bulgarian-nlp

This is a part-of-speech tagger for Bulgarian. 

## Usage

First clone the repository and make sure the transformers library is installed
```
git clone https://github.com/amontgomerie/bulgarian-nlp
cd bulgarian-nlp
```

Now POS tags can be generated like this:
```python
from models.postagger import POSTagger

pos_tagger = POSTagger()
pos_tagger.generate_tags('Аз сьм мьж.')
```
Which will generate:
```
['PRON', 'VERB', 'NOUN', 'PUNCT']
```
For more information about the POS tags, see https://universaldependencies.org/u/pos/
