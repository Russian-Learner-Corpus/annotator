from annotator.alignment import Alignment
from annotator.text_processor import TextProcessor
from annotator.merger import get_rule_edits
from annotator.classifier import classify


class Annotator:

    def __init__(self):
        self.processor = TextProcessor()

    def process(self, text):
        return self.processor.process(text)

    def align(self, orig, corr):
        orig = self.process(orig)
        corr = self.process(corr)
        return Alignment(orig, corr)

    def merge(self, alignment, algorithm="rules"):
        if algorithm == "rules":
            edits = get_rule_edits(alignment)
        elif algorithm == "all-split":
            edits = alignment.get_all_split_edits()
        elif algorithm == "all-merge":
            edits = alignment.get_all_merge_edits()
        elif algorithm == "all-equal":
            edits = alignment.get_all_equal_edits()
        else:
            raise Exception("Unknown merging algorithm. Choose from: "
                            "rules, all-split, all-merge, all-equal.")
        return edits

    def classify(self, edit):
        return classify(edit)

    def annotate(self, orig, cor, merging="rules"):
        alignment = self.align(orig, cor)
        edits = self.merge(alignment, merging)
        for edit in edits:
            edit = self.classify(edit)
        return edits
