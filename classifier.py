""" Main collection of functions for classifying edits """
import token

import pymorphy2
import re

from Levenshtein import distance, ratio as lev
from nltk.stem.snowball import SnowballStemmer

ML = False
# ML = True

if ML:
    from morph_ortho import is_morph

pymorphy_parser = pymorphy2.MorphAnalyzer()
stemmer = SnowballStemmer('russian')


def get_normal_form(word):
    return pymorphy_parser.parse(word)[0].normal_form


def get_pos(word, min_score=-1):
    p = pymorphy_parser.parse(word)[0]
    return p.tag.POS if p.score > min_score else None


def get_number(word):
    return pymorphy_parser.parse(word)[0].tag.number


def get_gender(word):
    first = True
    for p in pymorphy_parser.parse(word):
        if first or p.score > 0.001:
            if p.tag.gender:
                return p.tag.gender
            first = False
        else:
            return None
    return None


def get_tense(verb):
    return pymorphy_parser.parse(verb)[0].tag.tense


def get_possible_aspects(verb):
    aspects = [p.tag.aspect for p in pymorphy_parser.parse(verb) if p.tag.aspect]
    return list(dict.fromkeys(aspects))  # remove duplicates


def get_aspect(verb):
    aspects = get_possible_aspects(verb)
    return aspects[0] if aspects else None


def get_possible_num_cases(word):
    return set((p.tag.number, p.tag.case) for p in pymorphy_parser.parse(word)
               if p.tag.case)


def classify(edit):
    if not edit.o_toks and not edit.c_toks:
        edit.type = "UNK"
    elif not edit.o_toks and edit.c_toks:
        edit.type = get_one_sided_type(edit.c_toks)
    elif edit.o_toks and not edit.c_toks:
        edit.type = get_one_sided_type(edit.o_toks)
    else:
        if edit.o_str == edit.c_str:
            edit.type = "UNK"
        else:
            edit.type = get_two_sided_type(edit.o_toks, edit.c_toks)
    return edit


def get_one_sided_type(toks):
    if one_sided_tense(toks):
        return "Tense"
    if one_sided_mode(toks):
        return 'Mode'
    if one_sided_aux(toks):
        return 'Aux'
    if one_sided_conj(toks):
        return 'Conj'
    if one_sided_ref(toks):
        return 'Ref'
    if one_sided_prep(toks):
        return 'Prep'
    # if one_sided_cs(toks):
    #    return 'CS'
    if one_sided_punct(toks):
        return 'Punct'
    return 'Lex' if len(toks) == 1 else 'Constr'


def one_sided_cs(toks):
    for tok in toks:
        if tok.feats.get('Foreign') == 'Yes':
            return True
    return False


def one_sided_mode(toks):
    if len(toks) == 1 and toks[0].pos == 'AUX' and toks[0].feats.get('Mood') == 'Cnd':
        return True
    return False


def one_sided_ref(toks):
    pos_set = {tok.pos for tok in toks}
    return pos_set.issubset({'DET', 'PRON', 'ADP'}) and pos_set != {'ADP'}


def one_sided_conj(toks):
    pos_set = {tok.pos for tok in toks}

    if pos_set.issubset({'CCONJ', 'SCONJ'}):
        return True
    return False


def one_sided_aux(toks):
    if ((len(toks) == 1) and
            (toks[0].lemma == 'быть' or toks[0].lemma == 'стать') and
            (toks[0].feats.get('Tense') == 'Pres')):
        return True
    return False


def one_sided_tense(toks):
    if ((len(toks) == 1) and
            (toks[0].lemma == 'быть') and
            (toks[0].feats.get('Tense') != 'Pres')):
        return True
    return False


def one_sided_punct(toks):
    pos_set = {tok.pos for tok in toks}
    if pos_set.issubset({'PUNCT'}):
        return True


def one_sided_prep(toks):
    pos_set = {tok.pos for tok in toks}
    if pos_set.issubset({'ADP'}):
        return True
    return False


def one_sided_lex(toks):
    # pymorphy_parser = pymorphy2.MorphAnalyzer()
    if len(toks) == 1 and pymorphy_parser.word_is_known(toks[0].text):
        return True
    return False


def get_two_sided_type(o_toks, c_toks):
    if punct(o_toks, c_toks):
        return "Punct"
    if capitalization(o_toks, c_toks):
        return "Ortho"
    if word_order(o_toks, c_toks):
        return "WO"
    if graph(o_toks, c_toks):
        return "Graph"
    if cs(o_toks, c_toks):
        return "CS"
    if brev(o_toks, c_toks):
        return "Brev"
    if tense(o_toks, c_toks):
        return "Tense"
    if passive(o_toks, c_toks):
        return "Passive"
    if num(o_toks, c_toks):
        return "Num"
    if gender(o_toks, c_toks):
        return "Gender"
    if wrong_case(o_toks, c_toks):
        if noun_case(o_toks, c_toks):
            return "Nominative" if nominative(c_toks) else "Gov"
        else:
            return "Agrcase"
    if agrnum(o_toks, c_toks):
        return "Agrnum"
    if agrpers(o_toks, c_toks):
        return "Agrpers"
    if agrgender(o_toks, c_toks):
        return "Agrgender"

    if refl(o_toks, c_toks):
        return "Refl"
    if asp(o_toks, c_toks):
        return "Asp"

    if impers(o_toks, c_toks):
        return "Impers"
    if com(o_toks, c_toks):
        return "Com"
    if mode(o_toks, c_toks):
        return "Mode"

    if hyphen_ins(o_toks, c_toks):
        return "Hyphen+Ins"
    if hyphen_del(o_toks, c_toks):
        return "Hyphen+Del"

    if space_ins(o_toks, c_toks):
        return "Space+Ins"
    if space_del(o_toks, c_toks):
        return "Space+Del"

    if conj(o_toks, c_toks):
        return "Conj"
    if ref(o_toks, c_toks):
        return "Ref"
    if prep(o_toks, c_toks):
        return "Prep"

    if infl(o_toks, c_toks):
        return "Infl"
    if lex(o_toks, c_toks):
        return "Lex"
    if constr(o_toks, c_toks):
        return "Constr"

    if typical_ortho(o_toks, c_toks):
        return "Ortho"
    if morph(o_toks, c_toks):
        return "Morph"
    if ortho(o_toks, c_toks):
        return "Ortho"
    return "Misspell"


def punct(o_toks, c_toks):
    return one_sided_punct(o_toks) and one_sided_punct(c_toks)

# ORTHOGRAPHY

def capitalization(o_toks, c_toks):
    if len(o_toks) != len(c_toks):
        return False
    for o, c in zip(o_toks, c_toks):
        if o.text.lower() != c.text.lower():
            return False
    return True


def graph(o_toks, _):
    for tok in o_toks:
        if re.search('[а-яА-Я]', tok.text) and tok.feats.get('Foreign') == 'Yes':
            return True
    return False


def space_del(o_toks, c_toks):
    o_join = "".join([o.text.lower() for o in o_toks])
    c_join = "".join([c.text.lower() for c in c_toks])
    if o_join == c_join and len(o_toks) < len(c_toks):
        return True
    return False


def space_ins(o_toks, c_toks):
    o_join = "".join([o.text.lower() for o in o_toks])
    c_join = "".join([c.text.lower() for c in c_toks])
    if o_join == c_join and len(o_toks) > len(c_toks):
        return True
    return False


def hyphen_del(o_toks, c_toks):
    o_join = "".join([o.text.lower() for o in o_toks])
    c_join = "".join([c.text.lower() for c in c_toks])
    if '-' in c_join and '-' not in o_join and lev(o_join, re.sub('-', '', c_join)) >= .75:
        return True
    return False


def hyphen_ins(o_toks, c_toks):
    o_join = "".join([o.text.lower() for o in o_toks])
    c_join = "".join([c.text.lower() for c in c_toks])
    if '-' in o_join and '-' not in c_join and lev(re.sub('-', '', o_join), c_join) >= .75:
        return True
    return False


# MORPHOLOGY

def infl(o_toks, c_toks):
    if ((len(o_toks) == len(c_toks) == 1) and
            ((stemmer.stem(o_toks[0].text) == stemmer.stem(c_toks[0].text)) or
             (o_toks[0].lemma == c_toks[0].lemma))):
        parses = pymorphy_parser.parse(c_toks[0].lemma)
        if o_toks[0].text.lower() not in [form.word for p in parses for form in p.lexeme]:
            return True
    return False


def num(o_toks, c_toks):
    if len(o_toks) != len(c_toks):
        return False

    o_nums = [o_tok.feats.get('Number', None) for o_tok in o_toks]
    c_nums = [c_tok.feats.get('Number', None) for c_tok in c_toks]
    pos_set = {get_pos(tok.text) for tok in o_toks + c_toks}
    lemmas_match_flags = [(o_toks[i].lemma == c_toks[i].lemma) for i in range(len(o_toks))]

    if ((len(set(o_nums)) == len(set(c_nums)) == 1) and
            (not (None in o_nums)) and
            (not (None in c_nums)) and
            (o_nums != c_nums) and
            (sum(lemmas_match_flags) == len(lemmas_match_flags)) and
            (len(pos_set & {'NOUN', 'NPRO'}) > 0)):
        return True

    return False


def gender(o_toks, c_toks):
    if len(o_toks) != len(c_toks):
        return False

    # o_genders = [o_tok.feats.get('Gender', None) for o_tok in o_toks]
    o_genders = [pymorphy_parser.parse(o_tok.text)[0].tag.gender for o_tok in o_toks]
    # c_genders = [c_tok.feats.get('Gender', None) for c_tok in c_toks]
    c_genders = [pymorphy_parser.parse(c_tok.text)[0].tag.gender for c_tok in c_toks]
    if (o_genders == c_genders or
            len(set(o_genders)) > 1 or
            len(set(c_genders)) > 1 or
            None in o_genders or
            None in c_genders):
        return False

    if not all([(stemmer.stem(o_toks[i].lemma) ==
                 stemmer.stem(c_toks[i].lemma))
                for i in range(len(o_toks))]):
        return False

    pos_set = {tok.pos for tok in c_toks}
    return len(pos_set & {'PROPN', 'NOUN', 'PRON'}) > 0


# SYNTAX


def asp(o_toks, c_toks):
    o_asps = set.union(*(set(get_possible_aspects(t.text)) for t in o_toks))
    c_asps = set.union(*(set(get_possible_aspects(t.text)) for t in c_toks))
    if o_asps & c_asps or len(o_asps) == 0 or len(c_asps) == 0:
        return False

    o_verbs = [t for t in o_toks if t.pos == 'VERB']
    c_verbs = [t for t in c_toks if t.pos == 'VERB']
    if len(o_verbs) != len(c_verbs) or len(c_verbs) == 0:
        return False

    for ov, cv in zip(o_verbs, c_verbs):
        if not related_stems(ov.lemma, cv.lemma):
            return False

    return True


def passive(o_toks, c_toks):
    o_pos = [get_pos(o_tok.text) for o_tok in o_toks]
    c_pos = [get_pos(c_tok.text) for c_tok in c_toks]
    if not ({'VERB', 'PRTS'} & set(o_pos) and {'VERB', 'PRTS'} & set(c_pos)):
        return False

    o_voices = [o_tok.feats.get('Voice', None) for o_tok in o_toks]
    c_voices = [c_tok.feats.get('Voice', None) for c_tok in c_toks]
    if 'Pass' not in o_voices and 'Pass' in c_voices:
        return True

    return ('Pass' in o_voices and
            'Pass' not in c_voices
            and pymorphy_parser.word_is_known(o_toks[o_voices.index('Pass')].text))


def brev_natasha(o_toks, c_toks):
    for o_tok in o_toks:
        for c_tok in c_toks:
            if ((o_tok.lemma == c_tok.lemma
                 or stemmer.stem(o_tok.text) == stemmer.stem(c_tok.text))
                    and o_tok.feats.get('Variant') != c_tok.feats.get('Variant')):
                return True
    return False


def get_adjectives(toks):
    return [p for t in toks for p in pymorphy_parser.parse(t.text) if p.tag.POS in ['ADJF', 'ADJS', 'PRTS', 'PRTF']]


def brev(o_toks, c_toks):
    o_adjs = get_adjectives(o_toks)
    c_adjs = get_adjectives(c_toks)
    for o in o_adjs:
        for c in c_adjs:
            if (o.tag.POS != c.tag.POS and
                    (o.normal_form == c.normal_form or
                     stemmer.stem(o.word) == stemmer.stem(c.word))):
                return True
    return False


def extract_aux_tense_asp(toks):
    extracted_tense = None
    extracted_asp = None
    for t in toks:
        if t.lemma == 'быть' and t.pos == 'AUX' and t.feats.get('Tense') == 'Pres':
            return True, get_tense(t.text), get_aspect(t.text)
        if t.pos == 'VERB':
            extracted_tense = get_tense(t.text)
            extracted_asp = get_aspect(t.text)
    return False, extracted_tense, extracted_asp


def tense(o_toks, c_toks):
    # Past <-> Present switch
    if len(o_toks) == len(c_toks) == 1:
        o_tok = o_toks[0]
        c_tok = c_toks[0]
        if (o_tok.pos == c_tok.pos == 'VERB' and
                'Tense' in o_tok.feats and
                'Tense' in c_tok.feats and
                o_tok.lemma == c_tok.lemma and
                o_tok.feats['Tense'] != c_tok.feats['Tense'] and
                o_tok.rel == c_tok.rel):
            return True
    # Past/Present <-> Future switch
    else:
        aux_in_o, o_tense, _ = extract_aux_tense_asp(o_toks)
        aux_in_c, c_tense, _ = extract_aux_tense_asp(c_toks)
        return aux_in_o != aux_in_c and o_tense != c_tense
    return False


def remove_refl_postfix(text):
    potential_postfixes = ['ся', 'сь']

    for i in range(len(potential_postfixes)):
        position = text.rfind(potential_postfixes[i])
        if position != -1 and len(text) - position < 4:
            text = text[:position] + text[position + len(potential_postfixes[i]):]
            break

    return text


def related_stems(first, second):
    first_stem = stemmer.stem(first)
    second_stem = stemmer.stem(second)
    min_length = min(len(first_stem), len(second_stem))
    return (distance(first_stem[:min_length], second_stem[:min_length]) < 2 or
            distance(first_stem[-min_length:], second_stem[-min_length:]) < 2 or
            distance(first_stem, second_stem) <= 2 or
            first_stem in second_stem or
            second_stem in first_stem)


def morph(o_toks, c_toks):
    if ML:
        return morph_ml(o_toks, c_toks)
    return (len(o_toks) == len(c_toks) == 1
            and o_toks[0].lemma != c_toks[0].lemma
            and related_stems(o_toks[0].lemma, c_toks[0].lemma))


def morph_ml(o_toks, c_toks):
    return (ML and
            len(o_toks) == len(c_toks) == 1
            and is_morph(o_toks[0].text, c_toks[0].text))


def refl(o_toks, c_toks):
    o_verbs = [t for t in o_toks if get_pos(t.text) == 'VERB']
    c_verbs = [t for t in c_toks if get_pos(t.text) == 'VERB']
    if len(o_verbs) == len(c_verbs) == 1:
        o_basic = remove_refl_postfix(o_verbs[0].lemma)
        c_basic = remove_refl_postfix(c_verbs[0].lemma)
        if (((o_basic != o_verbs[0].lemma and c_basic == c_verbs[0].lemma) or
             (o_basic == o_verbs[0].lemma and c_basic != c_verbs[0].lemma)) and
                (related_stems(o_basic, c_basic) or
                 o_basic == c_verbs[0].text or
                 c_basic == o_verbs[0].text)):
            return True
    return False


def agrnum(o_toks, c_toks):
    if len(o_toks) != len(c_toks):
        return False

    o_nums = [get_number(o_tok.text) for o_tok in o_toks]
    c_nums = [get_number(c_tok.text) for c_tok in c_toks]
    lemmas_match_flags = [(o_toks[i].lemma == c_toks[i].lemma) for i in range(len(o_toks))]

    if ((len(set(o_nums)) == len(set(c_nums)) == 1) and
            (not (None in o_nums)) and
            (not (None in c_nums)) and
            (o_nums != c_nums) and
            (sum(lemmas_match_flags) == len(lemmas_match_flags))):
        return True

    return False


def wrong_case(o_toks, c_toks):
    if len(o_toks) != len(c_toks):
        return False

    o_num_cases = set.intersection(*[get_possible_num_cases(t.text) for t in o_toks])
    if not o_num_cases:
        return False

    c_num_cases = set.intersection(*[get_possible_num_cases(t.text) for t in c_toks])
    if not c_num_cases:
        return False

    if o_num_cases & c_num_cases:  # can be the same case
        return False

    if not ({n for (n, c) in o_num_cases} & {n for (n, c) in c_num_cases}):
        # different numbers
        return False

    for o, c in zip(o_toks, c_toks):
        if o.lemma != c.lemma:
            return False

    return True


def noun_case(o_toks, c_toks):
    pos_set = {tok.pos for tok in o_toks + c_toks}
    return len(pos_set & {'PROPN', 'NOUN', 'PRON'}) > 0


def nominative(c_toks):
    c_cases = [c_tok.feats.get('Case', None) for c_tok in c_toks]
    return 'Nom' in c_cases


def agrgender(o_toks, c_toks):
    if len(o_toks) != len(c_toks):
        return False

    # o_genders = [o_tok.feats.get('Gender', None) for o_tok in o_toks]
    # c_genders = [c_tok.feats.get('Gender', None) for c_tok in c_toks]
    o_genders = [get_gender(o_tok.text) for o_tok in o_toks]
    c_genders = [get_gender(o_tok.text) for o_tok in c_toks]
    lemmas_match_flags = [(o_toks[i].lemma == c_toks[i].lemma) for i in range(len(o_toks))]

    if ((len(set(o_genders)) == len(set(c_genders)) == 1) and
            (not (None in o_genders)) and
            (not (None in c_genders)) and
            (o_genders != c_genders) and
            (sum(lemmas_match_flags) == len(lemmas_match_flags))):
        return True
    return False


def agrpers(o_toks, c_toks):
    if len(o_toks) != len(c_toks):
        return False

    o_pers = [o_tok.feats.get('Person', None) for o_tok in o_toks]
    c_pers = [c_tok.feats.get('Person', None) for c_tok in c_toks]
    lemmas_match_flags = [(o_toks[i].lemma == c_toks[i].lemma) for i in range(len(o_toks))]

    if ((len(set(o_pers)) == len(set(c_pers)) == 1) and
            (not (None in o_pers)) and
            (not (None in c_pers)) and
            (set(o_pers) != set(c_pers)) and
            (sum(lemmas_match_flags) == len(lemmas_match_flags))):
        return True

    return False


def mode(o_toks, c_toks):
    aux_in_o_toks = [(tok.pos == 'AUX' and tok.feats.get('Mood') == 'Cnd') for tok in o_toks]
    aux_in_c_toks = [(tok.pos == 'AUX' and tok.feats.get('Mood') == 'Cnd') for tok in c_toks]

    if ((any(aux_in_o_toks) and not any(aux_in_c_toks)) or
            (any(aux_in_c_toks) and not any(aux_in_o_toks))):
        return True
    return False


def is_pronoun(tok):
    return tok.pos in {'DET', 'PRON'}


def ref(o_toks, c_toks):
    return (all(is_pronoun(t) for t in o_toks) or
            all(is_pronoun(t) for t in c_toks))


def is_introductory_word(text):
    return 'Prnt' in pymorphy_parser.parse(text)[0].tag


def is_conj(tok):
    return tok.pos in {'CCONJ', 'SCONJ'} or (get_pos(tok.text, 0.6) == 'CONJ' and
                                             not is_introductory_word(tok.text))


def conj(o_toks, c_toks):
    return any(is_conj(t) for t in o_toks + c_toks)


def com(o_toks, c_toks):
    o_cmp_flags = [True if tok.feats.get('Degree') == 'Cmp' else False for tok in o_toks]
    c_cmp_flags = [True if tok.feats.get('Degree') == 'Cmp' else False for tok in c_toks]
    if any(o_cmp_flags + c_cmp_flags):
        return True
    else:
        return False


def impers(o_toks, c_toks):
    o_rel = {tok.rel for tok in o_toks}
    c_rel = {tok.rel for tok in c_toks}
    if ((('nsubj' in o_rel and 'nsubj' not in c_rel) or
         ('nsubj' in c_rel and 'nsubj' not in o_rel)
    ) and
            ((len(o_toks) > 1) and
             (len(c_toks) > 1)
            )
    ):
        return True
    return False


def cs(o_toks, c_toks):
    for tok in o_toks:
        if tok.feats.get('Foreign') == 'Yes':
            return True
    return False


def lex(o_toks, c_toks):
    if len(o_toks) != 1 or len(c_toks) != 1:
        return False

    o = o_toks[0].text
    c = c_toks[0].text
    if not pymorphy_parser.word_is_known(o) or get_normal_form(o) == get_normal_form(c):
        return False

    o_letters = set(o)
    c_letters = set(c)
    # double letters and letters in a wrong order are ortho
    if o_letters == c_letters:
        return False
    # one-letter errors are ortho
    return (len(o) - len(c) > 1 or
            len(c) - len(o) > 1 or
            len(o_letters - c_letters) > 1 or
            len(c_letters - o_letters) > 1)


def prep(o_toks, c_toks):
    return {tok.pos for tok in o_toks + c_toks}.issubset({'ADP'})


def ortho(o_toks, c_toks):
    if ((len(o_toks) == len(c_toks) == 1) and
            (lev(o_toks[0].text, c_toks[0].text) >= 0.8)):
        return True
    return False


def typical_ortho(o_toks, c_toks):
    if len(o_toks) == len(c_toks) > 1:
        return False
    ot, ct = o_toks[0].text, c_toks[0].text
    if len(ot) != len(ct):
        return False
    for o, c in zip(ot, ct):
        if o != c and {o, c} != {'е', 'э'} and {o, c} != {'и', 'ы'} and {o, c} != {'а', 'я'}:
            return False
    return True


def aux(o_toks, c_toks):
    o_aux_flags = [(tok.lemma == 'быть' or tok.lemma == 'стать') for tok in o_toks]
    c_aux_flags = [(tok.lemma == 'быть' or tok.lemma == 'стать') for tok in c_toks]
    if ((len(o_toks) > 1) and
            (len(c_toks) > 1) and
            ((sum(o_aux_flags)) != (sum(c_aux_flags)))):
        return True
    return False


def constr(o_toks, c_toks):
    if len(o_toks) > 1 or len(c_toks) > 1:
        return True
    return False


def word_order(o_toks, c_toks):
    o_set = sorted([o.text.lower() for o in o_toks])
    c_set = sorted([c.text.lower() for c in c_toks])
    if o_set == c_set and len(o_set) > 1:
        return True
    return False
