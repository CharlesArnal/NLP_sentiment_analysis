import numpy, os
from subprocess import call
from gensim.models import Doc2Vec
from Classifiers import SVM


class SVMDoc2Vec(SVM):
    """ 
    class for baseline extension using SVM with Doc2Vec pre-trained vectors
    """
    def __init__(self,model,svm_dir,perf=False):
        """
        initialisation of parent SVM object and self.model attribute
        to initialise SVM parent use: SVM.__init_(self,svmlight_dir)

        @param model: pre-trained doc2vec model to use
        @type model: string (e.g. random_model.model)

        @param svmlight_dir: location of local binaries for svmlight
        @type svmlight_dir: string
        """
        # Q8
        SVM.__init__(self,svm_dir,perf)
        # pre-trained doc2vec model 
        self.model=model

    def normalize(self,vector):
        """
        normalise vector between -1 and 1 inclusive.

        @param vector: vector inferred from doc2vec
        @type vector: numpy array

        @return: normalised vector

        Not sure this is actually a good idea
        """
        # TODO Q8
        pass

    def getVectors(self,reviews):
        """
        infer document vector for each review. 

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)

        @return: list of (string, list) tuples where string is the label ("1"/"-1") and list
                 contains the features in svmlight format e.g. ("1",[(1, 0.04), (2, 4.0), ...])
                 svmlight feature format is: (id, value) and id must be > 0.
        """
        vectors=[]
        for sentiment,review in reviews:
            inferred_vector=self.model.infer_vector(review)
            vectors.append(("1" if sentiment=="POS" else "-1",[(index+1,value) for index, value in enumerate(inferred_vector)]))
            """
            if sentiment=="POS":            # to be sure that they appear in the right order
                vectors.append(("1",[(i+1,value) for i, value in enumerate(normalized_tokens_count) if value!=0]))
            if sentiment=="NEG":            # to be sure that they appear in the right order
                vectors.append(("-1",[(i+1,value) for i, value in enumerate(normalized_tokens_count) if value!=0]))
            """
            
        return vectors

    # since using pre-trained vectors don't need to determine features 
    def getFeatures(self,reviews):
        pass
