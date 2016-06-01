from stop_words import get_stop_words
from collections import Counter
from posNegLoader import loadPosNegList
import re
import spacy

FIRST_PERSON_PRONOUNS_SNG = ["i", "im", "i'll", "me", "my", "mine"]
FIRST_PERSON_PRONOUNS_PLR = ["we", "we'll", "ours", "our", "us"]

def add_formatting_features(data_point, features):
    num_bold = len(re.findall("[^\*]\*\*[^\*]", data_point))
    features["num bold"] = num_bold
    num_paragraphs = data_point.count("\n\n")
    features["num paragraphs"] = num_paragraphs

def add_pronoun_features(data_point, features):
    comment = data_point.lower().split()
    prons_sng = [word for word in comment if word in FIRST_PERSON_PRONOUNS_SNG]
    prons_plr = [word for word in comment if word in FIRST_PERSON_PRONOUNS_PLR]
    features["First Person sng"] = len(prons_sng)
    features["First Person plr"] = len(prons_sng)
    features["First Person sng frac"] = float(len(prons_sng)) / len(comment)
    features["First Person plr frac"] = float(len(prons_plr)) / len(comment)



POSLIST, NEGLIST = loadPosNegList("../posnegdata.csv")

def positive_words_intersection_features(data_point, features):
	comment = data_point.upper().split()
	posWords = [word for word in comment if word in POSLIST]
	posCount = len(posWords)

	# featurize the posword count as the value
	features["pos_words"] += posCount

	# featurize the posword count as part of the key
	# features["pos_words:", str(posCount)] += 1.0


def negative_words_intersection_features(data_point, features):
	comment = data_point.upper().split()
	negWords = [word for word in comment if word in NEGLIST]
	negCount = len(negWords)

	# featurize the negword count as the value
	features["neg_words"] += negCount

	# featurize the negword count as part of the key
	# features["neg_words:", str(negCount)] += 1.0



def op_test_phi(data_point, nlp):
    features = Counter()
    add_pronoun_features(data_point, features)
    positive_words_intersection_features(comment, features)
    negative_words_intersection_features(comment, features)
    add_formatting_features(data_point, features)
    return features
