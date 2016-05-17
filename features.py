from collections import Counter

def test_phi(data_point):
    features = Counter()
    features['len'] = len(str(data_point["content"]))
    print data_point["content"].keys()
    print features
    return features