from stop_words import get_stop_words
from collections import Counter
from posNegLoader import loadPosNegList
import re
import spacy



def op_test_phi(data_point, nlp):
    features = Counter()
    return features