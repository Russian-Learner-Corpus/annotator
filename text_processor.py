""" Main class for processing text fed to the annotator using external libraries """


from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    Doc,
    segment
)
from razdel import tokenize


class CyrLatSegmenter(Segmenter):
    def tokenize(self, text):
        tokens = []
        last = -1
        for t in tokenize(text):
            if last < 0:
                tokens = [t]
            elif t.start == last and tokens[-1].text[-1].isalnum() and t.text[0].isalnum():
                tokens[-1].text += t.text
                tokens[-1].stop = t.stop
            else:
                tokens.append(t)
            last = t.stop
        for t in tokens:
            yield segment.adapt_token(t)


class TextProcessor:
    def __init__(self):
        self.emb = NewsEmbedding()
        self.segmenter = CyrLatSegmenter()
        self.morph_vocab = MorphVocab()
        self.morph_tagger = NewsMorphTagger(self.emb)
        self.syntax_parser = NewsSyntaxParser(self.emb)

    def process(self, text):
        doc = Doc(text)
        # Sentence is split into tokens
        doc.segment(self.segmenter)
        if len(doc.tokens) > 0:
            # Every token is parsed by a morphology and syntax parser
            doc.tag_morph(self.morph_tagger)
            doc.parse_syntax(self.syntax_parser)

            # Additionally, lemmas of each token are extracted
            for token in doc.tokens:
                token.lemmatize(self.morph_vocab)
        return doc
