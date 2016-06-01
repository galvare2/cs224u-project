from stop_words import get_stop_words
from collections import Counter
from posNegLoader import loadPosNegList
import re
import spacy

FIRST_PERSON_PRONOUNS_SNG = ["i", "im", "i'll", "me", "my", "mine"]
FIRST_PERSON_PRONOUNS_PLR = ["we", "we'll", "ours", "our", "us"]

def add_pronoun_features(data_point, features):
    comment = data_point.lower().split()
    prons_sng = [word for word in comment if word in FIRST_PERSON_PRONOUNS_SNG]
    prons_plr = [word for word in comment if word in FIRST_PERSON_PRONOUNS_PLR]
    features["First Person sng"] = len(prons_sng)
    features["First Person plr"] = len(prons_sng)
    features["First Person sng frac"] = float(len(prons_sng)) / len(comment)
    features["First Person plr frac"] = float(len(prons_plr)) / len(comment)



def op_test_phi(data_point, nlp):
    features = Counter()
    add_pronoun_features(data_point, features)
    return features