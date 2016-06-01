import json
from sklearn.feature_extraction import DictVectorizer
from spacy.en import English

TRAIN_PAIR_DATA = "cmv/pair_task/train_pair_data.jsonlist"
TRAIN_OP_DATA = "cmv/op_task/train_op_data.jsonlist"
DEV_CUTOFF_FRACTION = 0.8


'''
helper functions for sanitizing data

'''

def shave_footnote(body):
    footnote_begin = "\n_____\n\n&gt;"
    footnote_end = "*Happy CMVing!*"
    footnote_begin_ind = body.find(footnote_begin)
    if footnote_begin_ind > -1:
        return body[:footnote_begin_ind]
    return body


def shave_CMV(title):
    cmv_ind = title.find("CMV: ")
    if cmv_ind == 0:
        return title[5:]
    return title



'''
pair task

'''

def reform_pair_data(data_object, value):
    result = {}
    result["op_author"] = data_object["op_author"]
    result["op_text"] = shave_footnote(data_object["op_text"])
    result["op_name"] = data_object["op_name"]
    result["op_title"] = shave_CMV(data_object["op_title"])
    result["content"] = data_object[value]
    return result


def load_pair_data():
    f = open(TRAIN_PAIR_DATA, "r")
    data = []
    for line in f:
        data_object = json.loads(line)
        data_pos = reform_pair_data(data_object, "positive")
        data_neg = reform_pair_data(data_object, "negative")
        data.append((data_pos, True))
        data.append((data_neg, False))
    num_examples = len(data)
    cutoff = int(num_examples * DEV_CUTOFF_FRACTION)
    data_train = data[:cutoff]
    data_dev = data[cutoff:]
    return data_train, data_dev


'''
op task

'''


def load_op_data():
    f = open(TRAIN_OP_DATA, "r")
    data = []
    for line in f:
        data_object = json.loads(line)
        title = data_object["title"]
        body = data_object["selftext"]

        # shave the footnotes off the body text
        body = shave_footnote(body)

        # shave the "CMV: " off the title
        title = shave_CMV(title)
        
        # the title is basically the first sentence of the body
        if title[-1] != '.':
            title = title + '.'
        body = title + ' ' + body

        data.append((body, data_object["delta_label"]))
    num_examples = len(data)
    cutoff = int(num_examples * DEV_CUTOFF_FRACTION)
    data_train = data[:cutoff]
    data_dev = data[cutoff:]
    return data_train, data_dev


'''
build dataset
'''


def build_dataset(data, phi, vectorizer=None):
    """Core general function for building experimental datasets.
    
    Parameters
    ----------
    data: the dict with the data 
       
    phi : feature function
       Any function that takes an `nltk.Tree` instance as input 
       and returns a bool/int/float-valued dict as output.
       
    vectorizer : sklearn.feature_extraction.DictVectorizer    
       If this is None, then a new `DictVectorizer` is created and
       used to turn the list of dicts created by `phi` into a 
       feature matrix. This happens when we are training.
              
       If this is not None, then it's assumed to be a `DictVectorizer` 
       and used to transform the list of dicts. This happens in 
       assessment, when we take in new instances and need to 
       featurize them as we did in training.
       
    Returns
    -------
    dict
        A dict with keys 'X' (the feature matrix), 'y' (the list of
        labels), 'vectorizer' (the `DictVectorizer`), and 
        'raw_examples' (the `nltk.Tree` objects, for error analysis).
    
    """  
    print "Build Data: Begin"
    labels = []
    feat_dicts = []
    raw_examples = []
    nlp = English(parser=False)
    for obj, label in data:
        labels.append(label)
        feat_dicts.append(phi(obj, nlp))
        raw_examples.append(obj)
    feat_matrix = None
    # In training, we want a new vectorizer:    
    if vectorizer == None:
        vectorizer = DictVectorizer(sparse=True)
        feat_matrix = vectorizer.fit_transform(feat_dicts)
    # In assessment, we featurize using the existing vectorizer:
    else:
        feat_matrix = vectorizer.transform(feat_dicts)
    print "Build data: End"
    return {'X': feat_matrix, 
            'y': labels, 
            'vectorizer': vectorizer, 
            'raw_examples': raw_examples}