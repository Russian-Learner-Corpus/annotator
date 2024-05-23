# Annotator

A tool for automatic error annotation in Russian sentences. Taking as input an original sentence and its correction, it extracts and classifies changes according to the [RLC type system](https://aclanthology.org/2024.lrec-main.1241.pdf). Developed as a modification of [ERRANT](https://github.com/chrisjbryant/errant) for Russian texts.

Usage: 

```
from annotator import Annotator
a = Annotator()
edits = a.annotate('К тому, что я считаю, что это единственное отличие на самом деле огромное, хотя его можно сформулировать всего лишь одно простое предложение.',
                   'К тому, что я считаю, что это единственное отличие на самом деле огромно, хотя его можно сформулировать всего лишь в одном простом предложении.')
for edit in edits:
      print(edit)
```

Result:
```
Orig: [14, 15, 'огромное'], Cor: [14, 15, 'огромно'], Type: 'Brev'
Orig: [22, 22, ''], Cor: [22, 23, 'в'], Type: 'Prep'
Orig: [22, 23, 'одно'], Cor: [23, 24, 'одном'], Type: 'Agrcase'
Orig: [23, 25, 'простое предложение'], Cor: [24, 26, 'простом предложении'], Type: 'Gov'
```
