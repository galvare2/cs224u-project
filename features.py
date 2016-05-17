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
def disagreement_measure(comment):
	com_list = comment.lower().split()
	return len(set(com_list).intersection(disagreement_markers.keys()))


# returns the size of the intersection of the agreement markers keys and the comment
def agreement_measure(comment):
	com_list = comment.lower().split()
	return len(set(com_list).intersection(agreement_markers.keys()))


def add_discourse_markers_features(data_point, features):
	op_text = data_point["op_text"]
	root_reply = data_point["content"]["comments"][0]["body"]
	last_reply = data_point["content"]["comments"][-1]["body"]

	features["disagree_factor:", disagreement_measure(root_reply)] += 1.0
	features["agree_factor:", agreement_measure(last_reply)] += 1.0

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
    #words_in_common_helper(features, op_text_words, root_reply_words, "all")
    #words_in_common_helper(features, op_text_stop_words, root_reply_stop_words, "stop words only")
    words_in_common_helper(features, op_text_content_words, root_reply_content_words, "content words only")

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
