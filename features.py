from stop_words import get_stop_words
from collections import Counter

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

disagreement_markers = {
	"really": 67,
	"no": 66,
	"actually": 60,
	"but": 58,
	"so": 58,
	"you mean": 57
}

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
	# features["agree_intersect_len_op:", agreement_intersection_length(op_text)] += 1.0
	features["disagree_intersect_sum_op:", disagreement_intersection_sum(op_text)] += 1.0
	features["agree_intersect_sum_op:", agreement_intersection_sum(op_text)] += 1.0


	#root
	#features["disagree_intersect_len_root:", disagreement_intersection_length(root_reply)] += 1.0
	#features["agree_intersect_len_root:", agreement_intersection_length(root_reply)] += 1.0

	#last comment
	#features["disagree_intersect_len_last:", disagreement_intersection_length(last_reply)] += 1.0
	#features["agree_intersect_len_last:", agreement_intersection_length(last_reply)] += 1.0


def words_in_common_helper(features, op, reply, name):
    common_words = len(op.intersection(reply))
    total_words = len(op.union(reply))
    common_words_feat = common_words / 20
    features["common_words:" + str(common_words_feat)] = 1
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

def add_article_features(data_point, features):
    root_reply = data_point["content"]["comments"][0]["body"].split(" ")
    for word in root_reply:
        if word in DEFINITE_ARTICLES:
            num_def += 1


def test_phi(data_point):
    features = Counter()
    comments = data_point["content"]["comments"]
    length = min(4, len(comments))
    features['len:' + str(length)] += 1.0
    add_words_in_common_features(data_point, features)
    add_discourse_markers_features(data_point, features)
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
