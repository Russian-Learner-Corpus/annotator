""" Module implementing merging rules. """

from edit import Edit
from itertools import combinations, groupby
from re import sub
import Levenshtein
from string import punctuation

open_pos = {'ADJ', 'AUX', 'ADV', 'NOUN'}#, 'VERB'}


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


def merge_by_indices(start, end, seq, alignment):
    return (process_seq(seq[:start], alignment) +
            merge_edits(seq[start : end + 1]) +
            process_seq(seq[end + 1:], alignment))


def number_case(seq):
    """If all elements have the same number and case,
    returns (number, case), otherwise None."""
    num, case = None, None
    for tok in reversed(seq):
        n, c = tok.feats.get('Number'), tok.feats.get('Case')
        if not num and not case:
            num, case = n, c
        elif num != n or (case != c and {case, c} != {'Acc', 'Nom'}):
            return None
    return num, case


def process_seq(seq, alignment):
    """ Processes a given sequence for merging based on rules"""
    ops = [op[0] for op in seq]

    # delete leading and trailing matches
    matches = ''.join(['M' if op == 'M' else 'N' for op in ops])
    left, right = matches.find('N'), matches.rfind('N') + 1
    seq = seq[left:right]
    if len(seq) <= 1:
        return seq

    ops = ops[left:right]
    unique_ops = set(ops)
    if unique_ops == {"D"} or unique_ops == {"I"}:
        return merge_edits(seq)

    # Loop through combinations of adjacent tokens, starting with the largest
    combos = list(combinations(range(0, len(seq)), 2))
    combos.sort(key=lambda x: x[1] - x[0], reverse=True)

    for start, end in combos:
        subops = ops[start: end + 1]
        o_seq = alignment.orig[seq[start][1]:seq[end][2]]
        c_seq = alignment.cor[seq[start][3]:seq[end][4]]
        o_toks = [tok.text.lower() for tok in o_seq]
        c_toks = [tok.text.lower() for tok in c_seq]

        if 'S' in subops:  # not just a word-order error
            if 'M' not in subops:  # else should be split into several edits

                # the last tokens differ in capitalization only
                if o_toks[-1] == c_toks[-1]:
                    if ((len(o_seq) == 1 and c_seq[0].text[0].isupper()) or
                            (len(c_seq) == 1 and o_seq[0].text[0].isupper())):
                        return merge_by_indices(start, end, seq, alignment)

                    # merge the last two tokens if the second to last is punct
                    if ((len(o_seq) > 1 and is_punct(o_seq[-2])) or
                            (len(c_seq) > 1 and is_punct(c_seq[-2]))):
                        return merge_by_indices(end - 1, end, seq, alignment)

                # hyphens and whitespace
                o_str = "".join(o_toks)
                s = sub("['-]", "", o_str)
                c_str = "".join(c_toks)
                t = sub("['-]", "", c_str)
                if s == t or (len(o_seq) + len(c_seq) <= 4 and
                              '-' in o_str + c_str and
                              Levenshtein.ratio(s, t) >= .75):
                    return merge_by_indices(start, end, seq, alignment)

                # the same POS, auxiliary and reflexive verbs
                o_pos_set = set([tok.pos for tok in o_seq])
                c_pos_set = set([tok.pos for tok in c_seq])
                pos_set = o_pos_set | c_pos_set
                if len(o_seq) != len(c_seq) and (len(pos_set) == 1 or
                                                 pos_set <= {'AUX', 'PART', 'VERB'} or
                                                 {'VERB', 'PRON'} == o_pos_set or
                                                 {'VERB', 'PRON'} == c_pos_set):
                    return merge_by_indices(start, end, seq, alignment)

                # the same number and case
                o_numcases = set([(tok.feats.get('Number'),
                                   tok.feats.get('Case'))
                                  for tok in o_seq])
                c_numcases = set([(tok.feats.get('Number'),
                                   tok.feats.get('Case'))
                                  for tok in c_seq])
                if len(o_numcases) == 1 == len(c_numcases) and o_numcases != c_numcases:
                    return merge_by_indices(start, end, seq, alignment)

                # don't merge
                if end - start < 2:
                    if (len(o_seq) == len(c_seq) == 2 or
                            (ops[start] == "S" and
                             char_cost(o_seq[0], c_seq[0]) > 0.75) or
                            (ops[end] == "S" and
                             char_cost(o_seq[-1], c_seq[-1]) > 0.75)):
                        return (process_seq(seq[:start + 1], alignment) +
                                process_seq(seq[start + 1:], alignment))

        # no substitutions; maybe, word-order error
        elif (subops[0] != 'M' != subops[-1] and
              subops.count('I') == subops.count('D') > 0 and
              set(o_toks) == set(c_toks)):
            return merge_by_indices(start, end, seq, alignment)

    # sequences containing content words
    seqs = []
    for merge, group in groupby(seq, lambda x: x[0][0] != "M"):
        if merge:
            seqs += merge_by_content(list(group), alignment)
    return seqs


def merge_by_content(seq, alignment):
    return seq if open_pos.isdisjoint(
        set([tok.pos for tok in alignment.orig[seq[0][1]:seq[-1][2]] +
             alignment.cor[seq[0][3]:seq[-1][4]]])) else merge_edits(seq)


def is_punct(token):
    return token.pos == "PUNCT" or token.text in punctuation


def char_cost(a, b):
    return Levenshtein.ratio(a.text, b.text)


def merge_edits(seq):
    if seq:
        return [("X", seq[0][1], seq[-1][2], seq[0][3], seq[-1][4])]
    else:
        return seq
