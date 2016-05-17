from collections import Counter

"""
    data_point - dict that has metadata and key "content"
    data_point["content"] - dict that has metadata and key "comments"
    data_point["content"]["comments"] - list of comments. Each comment is
    a dict with fields:

    [u'subreddit_id', u'banned_by', u'link_id', u'likes', u'replies', u'user_reports', u'saved', u'id', u'gilded', u'report_reasons', u'author', u'parent_id', u'score', u'approved_by', u'controversiality', u'body', u'edited', u'author_flair_css_class', u'downs', u'body_html', u'subreddit', u'score_hidden', u'name', u'created', u'author_flair_text', u'created_utc', u'distinguished', u'mod_reports', u'num_reports', u'ups']

    data_point["content"]["comments"][i]["body"] - contains the actual text of the ith comment


"""

def test_phi(data_point):
    features = Counter()
    comments = data_point["content"]["comments"]
    length = min(4, len(comments))
    features['len:' + str(len(comments))] = length
    return features