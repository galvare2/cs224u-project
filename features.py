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

def add_discourse_markers_features(data_point, features)):
	hi = "hi"


def add_words_in_common_features(data_point, features):
    op_text = data_point["op_text"]
    root_reply = data_point["content"]["comments"][0]["body"]
    op_text_words = set(op_text.split(" ")) #Unique words
    root_reply_words = set(root_reply.split(" "))
    common_words = len(op_text_words.intersection(root_reply_words)) / 20
    features["common_words:" + str(common_words)] = 1


def test_phi(data_point):
    features = Counter()
    comments = data_point["content"]["comments"]
    length = min(4, len(comments))
    features['len:' + str(length)] += 1.0
    add_words_in_common_features(data_point, features)
    add_discourse_markers_features(data_point, features))
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
