from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    Doc
)


class TextProcessor:
    def __init__(self):
        self.emb = NewsEmbedding()
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.morph_tagger = NewsMorphTagger(self.emb)
        self.syntax_parser = NewsSyntaxParser(self.emb)

    def process(self, text):
        doc = Doc(text)
        doc.segment(self.segmenter)
        if len(doc.tokens) > 0:
            doc.tag_morph(self.morph_tagger)
            doc.parse_syntax(self.syntax_parser)
            for token in doc.tokens:
                token.lemmatize(self.morph_vocab)
        return doc
