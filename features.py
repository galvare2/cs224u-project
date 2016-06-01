from stop_words import get_stop_words
from collections import Counter
from posNegLoader import loadPosNegList
import re
import spacy

"""
    data_point - dict that has metadata and key "content" as well as "op_text"
    data_point["content"] - dict that has metadata and key "comments"
    data_point["content"]["comments"] - list of comments. Each comment is
    a dict with fields:

    [u'subreddit_id', u'banned_by', u'link_id', u'likes', u'replies', u'user_reports',
    u'saved', u'id', u'gilded', u'report_reasons', u'author', u'parent_id', u'score',
    u'approved_by', u'controversiality', u'body', u'edited', u'author_flair_css_class',
    u'downs', u'body_html', u'subreddit', u'score_hidden', u'name', u'created',u'author_flair_text',
    u'created_utc', u'distinguished', u'mod_reports', u'num_reports', u'ups']

    data_point["content"]["comments"][i]["body"] - contains the actual text of the ith comment
"""


#POSLIST, NEGLIST = loadPosNegList("../posnegdata.csv")




def positive_words_intersection_features(comment, features):
	comment = comment.upper().split()
	posWords = [word for word in comment if word in POSLIST]
	posCount = len(posWords)

	# featurize the posword count as the value
	features["pos_words"] += posCount

	# featurize the pronoun count as part of the key
	# features["pos_words:", str(posCount)] += 1.0


def negative_words_intersection_features(comment, features):
	comment = comment.upper().split()
	negWords = [word for word in comment if word in NEGLIST]
	negCount = len(negWords)

	# featurize the posword count as the value
	features["neg_words"] += negCount

	# featurize the pronoun count as part of the key
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


def add_discourse_markers_features(data_point, features):
	op_text = data_point["op_text"]
	root_reply = data_point["content"]["comments"][0]["body"]
	last_reply = data_point["content"]["comments"][-1]["body"]

	#op
	# features["disagree_intersect_len_op:", disagreement_intersection_length(op_text)] += 1.0
	features["agree_intersect_len_op:", agreement_intersection_length(op_text)] += 1.0
	# features["disagree_intersect_sum_op:", disagreement_intersection_sum(op_text)] += 1.0
	# features["agree_intersect_sum_op:", agreement_intersection_sum(op_text)] += 1.0

	#root
	#features["disagree_intersect_len_root:", disagreement_intersection_length(root_reply)] += 1.0
	#features["agree_intersect_len_root:", agreement_intersection_length(root_reply)] += 1.0
	# features["disagree_intersect_sum_op:", disagreement_intersection_sum(root_reply)] += 1.0
	# features["agree_intersect_sum_op:", agreement_intersection_sum(root_reply)] += 1.0

	#last comment
	#features["disagree_intersect_len_last:", disagreement_intersection_length(last_reply)] += 1.0
	#features["agree_intersect_len_last:", agreement_intersection_length(last_reply)] += 1.0
	# features["disagree_intersect_sum_op:", disagreement_intersection_sum(last_reply)] += 1.0
	# features["agree_intersect_sum_op:", agreement_intersection_sum(last_reply)] += 1.0



SECOND_PERSON_PRONOUNS = ["you", "yours", "you'll", "y'all", "yall", "you're"]
FIRST_PERSON_PRONOUNS_SNG = ["i", "im", "i'll", "me", "my", "mine"]
FIRST_PERSON_PRONOUNS_PLR = ["we", "we'll", "ours", "our", "us"]


def personal_pronouns_helper(comment, features, pronouns_list, feature_key_name):
	comment = comment.lower().split()
	prons = [word for word in comment if word in pronouns_list]
	prons_count = len(prons)

	# featurize the pronoun count into the value
	features[feature_key_name] += prons_count

	# featurize the pronoun count as part of the key
	# features["1st_person_sg_num", str(first_person_count)] += 1.0

	# featurize the pronoun set (i.e. which pronouns appear at least once)
	# features["1st_person_sg_num" + str(list(set(first_person_prons)))] += 1.0



def words_in_common_helper(features, op, reply, name):
    common_words = len(op.intersection(reply))
    total_words = len(op.union(reply))
    common_words_feat = common_words / 20
    features["common_words:" + str(common_words_feat) + " " + name] = 1
    if len(reply) != 0:
        reply_frac = float(common_words) / len(reply)
    else:
        reply_frac = 0
    if len(op) != 0:
        op_frac = float(common_words) / len(op)
    else:
        op_frac = 0
    jaccard = float(common_words) / total_words
    features["root reply frac " + name] = reply_frac
    features["op frac " + name] = op_frac
    features["Jaccard " + name] = jaccard

def add_words_in_common_features(data_point, features):
    op_text = data_point["op_text"]
    root_reply = data_point["content"]["comments"][0]["body"]
    stop_words = set(get_stop_words("en"))

    op_text_words = set(op_text.split(" ")) #Unique words
    root_reply_words = set(root_reply.split(" "))

    op_text_content_words = op_text_words.difference(stop_words)
    root_reply_content_words = root_reply_words.difference(stop_words)

    op_text_stop_words = op_text_words.intersection(stop_words)
    root_reply_stop_words = root_reply_words.intersection(stop_words)
    words_in_common_helper(features, op_text_words, root_reply_words, "all")
    #words_in_common_helper(features, op_text_stop_words, root_reply_stop_words, "stop words only")
    #words_in_common_helper(features, op_text_content_words, root_reply_content_words, "content words only")

DEFINITE_ARTICLES = ["the"]
INDEFINITE_ARTICLES = ["a", "an"]

def add_misc_features(data_point, features):
    root_reply = data_point["content"]["comments"][0]["body"]
    num_sentences = root_reply.count(". ") + root_reply.count(".\n")
    features["num sentences:"] = num_sentences
    num_paragraphs = root_reply.count("\n\n")
    features["num paragraphs:"] = num_paragraphs
    num_question_marks = root_reply.count("?")
    features["num question marks:"] = num_question_marks


def add_article_features(data_point, features):
    root_reply = data_point["content"]["comments"][0]["body"].split(" ")
    num_def, num_indef = (0, 0)
    for word in root_reply:
        if word in DEFINITE_ARTICLES:
            num_def += 1
        if word in INDEFINITE_ARTICLES:
            num_indef += 1
    features["Definite Articles"] = num_def
    features["Indefinite Articles"] = num_indef

def add_link_features(data_point, features):
    root_reply = data_point["content"]["comments"][0]["body"]
    num_com_links = root_reply.count(".com")
    num_links = root_reply.count("http")
    frac_links = float(num_links) / len(root_reply.split(" "))
    frac_com_links = float(num_com_links) / len(root_reply.split(" "))
    features[".com links"] = num_com_links
    features["total links"] = num_links
    features["fraction links"] = frac_links
    features["fraction .com links"] = frac_links

def add_markdown_features(data_point, features):
    root_reply = data_point["content"]["comments"][0]["body"]
    num_italics = len(re.findall("[^\*]\*[^\*]", root_reply))
    features["num italics"] = num_italics
    num_bold = len(re.findall("[^\*]\*\*[^\*]", root_reply))
    features["num bold"] = num_bold


from spacy.parts_of_speech import NOUN, ADJ, PRON, ADV, INTJ, CONJ

def add_pos_count_parse_features(features, doc):
    num_nouns = len([word for word in doc if word.pos == spacy.parts_of_speech.NOUN])
    num_adj = len([word for word in doc if word.pos == spacy.parts_of_speech.ADJ])
    num_pron = len([word for word in doc if word.pos == spacy.parts_of_speech.PRON])
    num_adv = len([word for word in doc if word.pos == spacy.parts_of_speech.ADV])
    num_intj = len([word for word in doc if word.pos == spacy.parts_of_speech.INTJ])
    num_conj = len([word for word in doc if word.pos == spacy.parts_of_speech.CONJ])
    num_words = len(doc)
    features["Nouns Count"] = float(num_nouns)
    features["Adjectives Count"] = float(num_adj)
    features["Pronouns Count"] = float(num_pron)
    features["Adverbs Count"] = float(num_adv)
    features["Interjections Count"] = float(num_intj)
    features["Conjunctions Count"] = float(num_conj)

def add_pos_similarity_parse_features(features, doc_rr, doc_op): 
    num_nouns_rr = len([word for word in doc_rr if word.pos == spacy.parts_of_speech.NOUN])
    num_adj_rr = len([word for word in doc_rr if word.pos == spacy.parts_of_speech.ADJ])
    num_pron_rr = len([word for word in doc_rr if word.pos == spacy.parts_of_speech.PRON])
    num_adv_rr = len([word for word in doc_rr if word.pos == spacy.parts_of_speech.ADV])
    num_intj_rr = len([word for word in doc_rr if word.pos == spacy.parts_of_speech.INTJ])
    num_conj_rr = len([word for word in doc_rr if word.pos == spacy.parts_of_speech.CONJ])
    num_words_rr = len(doc_rr)
    num_nouns_op = len([word for word in doc_op if word.pos == spacy.parts_of_speech.NOUN])
    num_adj_op = len([word for word in doc_op if word.pos == spacy.parts_of_speech.ADJ])
    num_pron_op = len([word for word in doc_op if word.pos == spacy.parts_of_speech.PRON])
    num_adv_op = len([word for word in doc_op if word.pos == spacy.parts_of_speech.ADV])
    num_intj_op = len([word for word in doc_op if word.pos == spacy.parts_of_speech.INTJ])
    num_conj_op = len([word for word in doc_op if word.pos == spacy.parts_of_speech.CONJ])
    num_words_op = len(doc_op)
    features["Nouns Difference"] = float(num_nouns_rr) - num_nouns_op
    features["Adjectives Difference"] = float(num_adj_rr) - num_adj_op 
    features["Pronouns Difference"] = float(num_pron_rr) - num_pron_op
    features["Adverbs Difference"] = float(num_adv_rr) - num_adv_op
    features["Interjections Difference"] = float(num_intj_rr) - num_adv_op
    features["Conjunctions Difference"] = float(num_conj_rr) - num_conj_op

def add_entity_parse_features(features, doc):
    for ent in doc.ents:
        features["Entity: " + ent.label_] += 1

def add_parse_features(data_point, features, nlp):
    root_reply = data_point["content"]["comments"][0]["body"]
    op_text = data_point["op_text"]
    doc_rr = nlp(root_reply)
    doc_op = nlp(op_text)
    #add_pos_count_parse_features(features, doc_rr)
    add_pos_similarity_parse_features(features, doc_rr, doc_op)
    #Won't work unless you re-enable the entity tagger in load_data
    #add_entity_parse_features(features, doc_rr)



def test_phi(data_point, nlp):
    features = Counter()
    comments = data_point["content"]["comments"]
    op_text = data_point["op_text"]
    root_reply_text = comments[0]["body"]
    length = min(4, len(comments))
    
    ##### IMPORTANT!!!! Two ## marks means the feature is good and we're using it

    ##features['len:' + str(length)] += 1.0
    ##features['num words'] += len(data_point["content"]["comments"][0]["body"].split(" "))
    
    # interplay
    ##add_words_in_common_features(data_point, features)

    # discourse markers
    ##add_discourse_markers_features(data_point, features)

    # personal pronouns
    # personal_pronouns_helper(comment=root_reply_text, features, pronouns_list=SECOND_PERSON_PRONOUNS, "2nd_person_root")
    ##personal_pronouns_helper(op_text, features, FIRST_PERSON_PRONOUNS_SNG, "1st_person_sg_op")
    # personal_pronouns_helper(root_reply_text, features, FIRST_PERSON_PRONOUNS_PLR, "1st_person_pl_root")

    # positive/negative words
    # positive_words_intersection_features(root_reply_text, features)
    # negative_words_intersection_features(root_reply_text, features)
    # formatting
    ##add_article_features(data_point, features)
    ##add_link_features(data_point, features)
    ##add_misc_features(data_point, features)
    #add_markdown_features(data_point, features)
    add_parse_features(data_point, features, nlp)
    return features


def ups_downs_oracle_phi(data_point):
    features = Counter()
    comments = data_point["content"]["comments"]
    total_num_ups = 0
    total_num_downs = 0
    for comment in comments:
    	total_num_downs += comment["downs"]
    	total_num_ups += comment["ups"]
    	features[str(total_num_ups - total_num_downs)] += 1.0
