import json
from sklearn.feature_extraction import DictVectorizer

TRAIN_OP_DATA = "cmv/op_task/train_op_data.jsonlist"
DEV_CUTOFF_FRACTION = 0.8

def load_data():
	f = open(TRAIN_OP_DATA, "r")
	data = []
	for line in f:
		data_object = json.loads(line)
		data.append((data_object, data_object["delta_label"]))
	num_examples = len(data)
	cutoff = int(num_examples * DEV_CUTOFF_FRACTION)
	data_train = data[:cutoff]
	data_dev = data[cutoff:]
	return data_train, data_dev

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
    labels = []
    feat_dicts = []
    raw_examples = []
    for obj, label in data:
        labels.append(label)
        feat_dicts.append(phi(obj))
        raw_examples.append(obj)
    feat_matrix = None
    # In training, we want a new vectorizer:    
    if vectorizer == None:
        vectorizer = DictVectorizer(sparse=True)
        feat_matrix = vectorizer.fit_transform(feat_dicts)
    # In assessment, we featurize using the existing vectorizer:
    else:
        feat_matrix = vectorizer.transform(feat_dicts)
    return {'X': feat_matrix, 
            'y': labels, 
            'vectorizer': vectorizer, 
            'raw_examples': raw_examples}