import load_data

def train_model():
	data_train, data_dev = load_data.load_data()
	dataset = build_dataset(data_train, phi)

def experiment(
        phi,
        train_func=fit_maxent_classifier,
        score_func=utils.safe_macro_f1,
        verbose=True):
    """Generic experimental framework for SST. Either assesses with a 
    random train/test split of `train_reader` or with `assess_reader` if 
    it is given.
    
    Parameters
    ----------
       
    phi : feature function (default: `unigrams_phi`)
        Any function that takes an `nltk.Tree` instance as input 
        and returns a bool/int/float-valued dict as output.
       
    train_func : model wrapper (default: `fit_maxent_classifier`)
        Any function that takes a feature matrix and a label list
        as its values and returns a fitted model with a `predict`
        function that operates on feature matrices.
    
    score_metric : function name (default: `utils.safe_macro_f1`)
        This should be an `sklearn.metrics` scoring function. The 
        default is weighted average F1 (macro-averaged F1). For 
        comparison with the SST literature, `accuracy_score` might
        be used instead. For micro-averaged F1, use
        
        (lambda y, y_pred : f1_score(y, y_pred, average='micro', pos_label=None))
                
        For other metrics that can be used here, see
        see http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
        
    verbose : bool (default: True)
        Whether to print out the model assessment to standard output.
        Set to False for statistical testing via repeated runs.
       
    Prints
    -------    
    To standard output, if `verbose=True`
        Model accuracy and a model precision/recall/F1 report. Accuracy is 
        reported because many SST papers report that figure, but the 
        precision/recall/F1 is better given the class imbalances and the 
        fact that performance across the classes can be highly variable.
        
    Returns
    -------
    float
        The overall scoring metric as determined by `score_metric`.
    
    """
	data_train, data_dev = load_data.load_data()
    # Train dataset:
    train = load_data.build_dataset(data_train, phi) 
    # Manage the assessment set-up:
    X_train = train['X']
    y_train = train['y']
    X_assess = None 
    y_assess = None
    if assess_reader == None:
         X_train, X_assess, y_train, y_assess = train_test_split(
                X_train, y_train, train_size=train_size)
    else:
        # Assessment dataset using the training vectorizer:
        assess = build_dataset(
            assess_reader, 
            phi, 
            class_func, 
            vectorizer=train['vectorizer'])
        X_assess, y_assess = assess['X'], assess['y']
    # Train:      
    mod = train_func(X_train, y_train)    
    # Predictions:
    predictions = mod.predict(X_assess)
    # Report:
    if verbose:
        print('Accuracy: %0.03f' % accuracy_score(y_assess, predictions))
        print(classification_report(y_assess, predictions, digits=3))
    # Return the overall score:
    return score_func(y_assess, predictions)


if __name__ == "__main__":
	train_model()