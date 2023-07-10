""" Module implementing merging rules. """


from edit import Edit
from itertools import combinations, groupby
from re import sub
import Levenshtein
from string import punctuation

open_pos = {'ADJ', 'AUX', 'ADV', 'NOUN', 'VERB'}


def get_rule_edits(alignment):
    """ Merges edits based on a set of rules """
    edits = []
    for op, group in groupby(alignment.align_seq,
                             lambda x: "T" if x[0][0] == "T" else False):
        group = list(group)
        if op == "T":
            for seq in group:
                edits.append(Edit(alignment.orig, alignment.cor, seq[1:]))
        else:
            processed = process_seq(group, alignment)
            for seq in processed:
                edits.append(Edit(alignment.orig, alignment.cor, seq[1:]))
    return edits


def process_seq(seq, alignment):
    """ Processes a given sequence for merging based on rules"""

    ops = [op[0] for op in seq]
    unique_ops = set(ops)
    if unique_ops == {"D"} or unique_ops == {"I"}:
        return merge_edits(seq)
    if unique_ops <= {"M"}:
        return []

    # ignore leading and trailing matches
    matches = ''.join(['M' if op == 'M' else 'N' for op in ops])
    left, right = matches.index('N'), matches.rindex('N') + 1
    seq = seq[left:right]
    ops = ops[left:right]

    if len(seq) == 1:
        return seq


    content = False
    # We loop through all possible combinations of tokens in the sequence, starting from the largest
    combos = list(combinations(range(0, len(seq)), 2))
    combos.sort(key=lambda x: x[1] - x[0], reverse=True)

    for start, end in combos:

        subops = ops[start : end + 1]
        o = alignment.orig[seq[start][1]:seq[end][2]]
        c = alignment.cor[seq[start][3]:seq[end][4]]
        o_toks = [tok.text.lower() for tok in o]
        c_toks = [tok.text.lower() for tok in c]

        if "S" not in subops:
            # Check for a word-order error:
            # contains the same number of inserts and deletes;
            # original and corrected contain the same tokens
            if (subops.count('I') == subops.count('D') > 0 and
                    set(o_toks) == set(c_toks)):
                return (merge_edits(seq[start:end + 1]) +
                        process_seq(seq[end + 1:], alignment))
            # If not word order, consider only sequences with substitutions
            continue

        # matches are allowed only in word-order errors
        if "M" in subops:
            continue

        if o_toks[-1] == c_toks[-1]:
            if start == 0 and ((len(o) == 1 and c[0].text[0].isupper()) or
                               (len(c) == 1 and o[0].text[0].isupper())):
                return merge_edits(seq[start:end + 1]) + \
                       process_seq(seq[end + 1:], alignment)

            if (len(o) > 1 and is_punct(o[-2])) or \
                    (len(c) > 1 and is_punct(c[-2])):
                return process_seq(seq[:end - 1], alignment) + \
                       merge_edits(seq[end - 1:end + 1]) + \
                       process_seq(seq[end + 1:], alignment)

        o_str = "".join(o_toks)
        s = sub("['-]", "", o_str)
        c_str = "".join(c_toks)
        t = sub("['-]", "", c_str)
        if s == t or (len(o) + len(c) <= 4 and
                      '-' in o_str + c_str and
                      Levenshtein.ratio(s, t) >= .75):
            return process_seq(seq[:start], alignment) + \
                   merge_edits(seq[start:end + 1]) + \
                   process_seq(seq[end + 1:], alignment)

        o_pos_set = set([tok.pos for tok in o])
        c_pos_set = set([tok.pos for tok in c])
        pos_set = set([tok.pos for tok in o] + [tok.pos for tok in c])
        if len(o) != len(c) and (len(pos_set) == 1 or
                                 (o_pos_set | c_pos_set).issubset({'AUX', 'PART', 'VERB'}) or
                                 {'VERB', 'PRON'} == o_pos_set or
                                 {'VERB', 'PRON'} == c_pos_set):
            return process_seq(seq[:start], alignment) + \
                   merge_edits(seq[start:end + 1]) + \
                   process_seq(seq[end + 1:], alignment)

        o_numcases = [(tok.feats.get('Number'), tok.feats.get('Case')) for tok in o]
        c_numcases = [(tok.feats.get('Number'), tok.feats.get('Case')) for tok in c]
        if len(set(o_numcases)) == 1 and len(set(c_numcases)) == 1:
            return process_seq(seq[:start], alignment) + \
                   merge_edits(seq[start:end + 1]) + \
                   process_seq(seq[end + 1:], alignment)

        if end - start < 2:
            if len(o) == len(c) == 2:
                return process_seq(seq[:start + 1], alignment) + \
                       process_seq(seq[start + 1:], alignment)
            if (ops[start] == "S" and char_cost(o[0], c[0]) > 0.75) or \
                    (ops[end] == "S" and char_cost(o[-1], c[-1]) > 0.75):
                return process_seq(seq[:start + 1], alignment) + \
                       process_seq(seq[start + 1:], alignment)
        if not pos_set.isdisjoint(open_pos):
            content = True

    if content:
        return merge_edits(seq)

    return [op for op in seq if op[0] != 'M']


def is_punct(token):
    return token.pos == "PUNCT" or token.text in punctuation


def char_cost(a, b):
    return Levenshtein.ratio(a.text, b.text)


def merge_edits(seq):
    if seq:
        return [("X", seq[0][1], seq[-1][2], seq[0][3], seq[-1][4])]
    else:
        return seq
