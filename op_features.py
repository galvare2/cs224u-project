from stop_words import get_stop_words
from collections import Counter
from posNegLoader import loadPosNegList
import re
import spacy

FIRST_PERSON_PRONOUNS_SNG = ["i", "im", "i'll", "me", "my", "mine"]
FIRST_PERSON_PRONOUNS_PLR = ["we", "we'll", "ours", "our", "us"]

def add_formatting_features(data_point, features):
    text = data_point[0] + data_point[1]
    num_bold = len(re.findall("[^\*]\*\*[^\*]", text))
    features["num bold"] = num_bold
    num_paragraphs = text.count("\n\n")
    features["num paragraphs"] = num_paragraphs

def add_pronoun_features(data_point, features):
    text = data_point[0] + data_point[1]
    comment = text.lower().split()
    prons_sng = [word for word in comment if word in FIRST_PERSON_PRONOUNS_SNG]
    prons_plr = [word for word in comment if word in FIRST_PERSON_PRONOUNS_PLR]
    features["First Person sng"] = len(prons_sng)
    features["First Person plr"] = len(prons_sng)
    features["First Person sng frac"] = float(len(prons_sng)) / len(comment)
    features["First Person plr frac"] = float(len(prons_plr)) / len(comment)



# POSLIST, NEGLIST = loadPosNegList("../posnegdata.csv")

def positive_words_intersection_features(body, features):
    comment = body.upper().split()
    pos_intersect = set(POSLIST).intersection(comment)
    posCount = len(pos_intersect)

    # featurize the posword count as the value
    features["pos_words"] += posCount

    # featurize the posword count as part of the key
    # features["pos_words:", str(posCount)] += 1.0


def negative_words_intersection_features(body, features):
    comment = body.upper().split()
    neg_intersect = set(NEGLIST).intersection(comment)
    negCount = len(neg_intersect)

    # featurize the negword count as the value
    features["neg_words"] += negCount

    # featurize the negword count as part of the key
    # features["neg_words:", str(negCount)] += 1.0




# dictionary of discourse markers indicative of disagreement
disagreement_markers = {
    "really": 67,
    "no": 66,
    "actually": 60,
    "but": 58,
    "so": 58,
    "you mean": 57
}

# dictionary of discourse markers indicative of agreement
agreement_markers = {
    "yes": 73,
    "i know": 64,
    "i believe": 62,
    "i think": 61,
    "just": 57
}


# returns the size of the intersection of the disagreement markers keys and the comment
def disagreement_intersection_length(comment):
    com_list = comment.lower().split()
    return len(set(com_list).intersection(disagreement_markers.keys()))


# returns the size of the intersection of the agreement markers keys and the comment
def agreement_intersection_length(comment):
    com_list = comment.lower().split()
    return len(set(com_list).intersection(agreement_markers.keys()))


def disagreement_intersection_sum(comment):
    com_list = comment.lower().split()
    dis_list = set(com_list).intersection(disagreement_markers.keys())
    dis_sum = 0
    for dis_mark in dis_list:
    	dis_sum += disagreement_markers[dis_mark]
    return dis_sum

def agreement_intersection_sum(comment):
    com_list = comment.lower().split()
    agr_list = set(com_list).intersection(agreement_markers.keys())
    agr_sum = 0
    for agr_mark in agr_list:
    	agr_sum += agreement_markers[agr_mark]
    return agr_sum


def add_discourse_markers_features(body, features):
    features["disagree_intersect_len:", disagreement_intersection_length(body)] += 1.0
    # features["agree_intersect_len:", agreement_intersection_length(body)] += 1.0
    # features["disagree_intersect_sum:", disagreement_intersection_sum(body)] += 1.0
    # features["agree_intersect_sum:", agreement_intersection_sum(body)] += 1.0





def op_test_phi(data_point, nlp):
    title, body = data_point
    features = Counter()
    add_pronoun_features(data_point, features)
    add_formatting_features(data_point, features)
    # positive_words_intersection_features(body, features)
    # negative_words_intersection_features(body, features)
    add_discourse_markers_features(body, features)
    return features
