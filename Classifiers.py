import os
from subprocess import call
from nltk.util import ngrams
from Analysis import Evaluation
from math import log
from numpy import linalg, array

class NaiveBayesText(Evaluation):
    def __init__(self,smoothing,bigrams,trigrams,discard_closed_class,smoothing_constant=1,no_unigrams=False):
        """
        initialisation of NaiveBayesText classifier.

        @param smoothing: use smoothing?
        @type smoothing: booleanp

        @param bigrams: add bigrams?
        @type bigrams: boolean

        @param trigrams: add trigrams?
        @type trigrams: boolean

        @param discard_closed_class: restrict unigrams to nouns, adjectives, adverbs and verbs?
        @type discard_closed_class: boolean
        """
        # set of features for classifier
        self.vocabulary=set()
        # prior probability     - dictionary, of the form "POS": f(POS)
        self.prior={}           
        # conditional probablility - dictionary, of the form "myword": [f(myword|POS),f(myword|NEG)]
        self.condProb={}
        # use smoothing?
        self.smoothing=smoothing
        # add bigrams?
        self.bigrams=bigrams
        # add trigrams?
        self.trigrams=trigrams
        # keep unigrams?
        self.no_unigrams=no_unigrams
        # restrict unigrams to nouns, adjectives, adverbs and verbs?
        self.discard_closed_class=discard_closed_class
        # stored predictions from test instances
        self.predictions=[]

        self.smoothing_constant=smoothing_constant

    def extractVocabulary(self,reviews):
        """
        extract features from training data and store in self.vocabulary.

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)
        """
        for sentiment,review in reviews:
            for token in self.extractReviewTokens(review):
                self.vocabulary.add(token)                      # remarque: les entrées de vocabulary sont de la forme ("badness","JJ") ou "badness" ou ("bad","JJ") (stemmed) ou "bad" (stemmed) selon les paramètres du corpus

    def extractReviewTokens(self,review):
        """
        extract tokens from reviews.

        @param review: movie review, without POS or NEG
        @type review: list of ("word", "POStag")

        @return: list of strings
        """
        text=[]
        if self.no_unigrams==False:
            for token in review:
                # check if pos tags are included in review e.g. ("bad","JJ")
                if len(token)==2 and self.discard_closed_class:
                    if token[1][0:2] in ["NN","JJ","RB","VB"]: text.append(token)  
                else:
                    text.append(token)
        if self.bigrams:                
            for bigram in ngrams(review,2): text.append(bigram)         # les ngrams contiennent aussi les tags
        if self.trigrams:                                               # pas d'option d'enlever les closed classes?
            for trigram in ngrams(review,3): text.append(trigram)
        return text

    def train(self,reviews):
        """
        train NaiveBayesText classifier.

        1. reset self.vocabulary, self.prior and self.condProb
        2. extract vocabulary (i.e. get features for training)
        3. get prior and conditional probability for each label ("POS","NEG") and store in self.prior and self.condProb
           note: to get conditional concatenate all text from reviews in each class and calculate token frequencies
                 to speed this up simply do one run of the movie reviews and count token frequencies if the token is in the vocabulary,
                 then iterate the vocabulary and calculate conditional probability (i.e. don't read the movie reviews in their entirety 
                 each time you need to calculate a probability for each token in the vocabulary)

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)
        """
        self.vocabulary=set()
        self.prior={}
        self.condProb={}

        self.extractVocabulary(reviews)

        count_classes={"POS":0, "NEG":0}    # dictionary with number of POS and NEG
        # tokens can be "words", ("words", "tag"), bigrams of these, ...
        count_tokens=dict([[token,[0,0]] for token in  self.vocabulary])    # dictionary with "token": [count in POS, count in NEG]
        count_pos,count_neg=0,0   # total number of tokens in positive and negative reviews respectively


        for sentiment, review in reviews:
            if sentiment=="POS":
                count_classes["POS"]+=1
            elif sentiment=="NEG":
                count_classes["NEG"]+=1
            else:
                print("error")
            for token in self.extractReviewTokens(review):
                if sentiment=="POS":
                    count_tokens[token][0]+=1
                    count_pos+=1
                elif sentiment=="NEG":
                    count_tokens[token][1]+=1
                    count_neg+=1

        self.prior["POS"]=count_classes["POS"]/(count_classes["POS"]+count_classes["NEG"])
        self.prior["NEG"]=count_classes["NEG"]/(count_classes["POS"]+count_classes["NEG"])
        
    
        for token in count_tokens:
            if self.smoothing:
                self.condProb[token] = [(count_tokens[token][0]+self.smoothing_constant)/(count_pos+len(self.vocabulary)*self.smoothing_constant),(count_tokens[token][1]+self.smoothing_constant)/(count_neg+len(self.vocabulary)*self.smoothing_constant)]
            elif count_pos != 0 and count_neg !=0:
                self.condProb[token] = [count_tokens[token][0]/count_pos,count_tokens[token][1]/count_neg]
            


        
    def test(self,reviews):
        """

        test NaiveBayesText classifier and store predictions in self.predictions.
        self.predictions should contain a "+" if prediction was correct and "-" otherwise.
        
        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)

        Deletes all previous predictions
        """
        self.predictions=[]
        
        for sentiment, review in reviews:
            pos_log_likelihood= log(self.prior["POS"])
            neg_log_likelihood= log(self.prior["NEG"])
            Prob_0_Pos=False            # used when a word was never seen before
            Prob_0_Neg=False
            for token in self.extractReviewTokens(review):
                if token in self.condProb:                      # ignore all words we have never seen (even if smoothing, they don't change the comparison)
                    if self.condProb[token][0]==0:
                        Prob_0_Pos=True
                    if self.condProb[token][1]==0:
                        Prob_0_Neg=True
                    if self.condProb[token][0]!=0 and self.condProb[token][1]!=0:
                        pos_log_likelihood+=log(self.condProb[token][0])
                        neg_log_likelihood+=log(self.condProb[token][1])
            if Prob_0_Neg and Prob_0_Pos:
                # when both probabilities 0, the model failed
                self.predictions.append("-")
            elif (sentiment == "POS" and (pos_log_likelihood>=neg_log_likelihood or Prob_0_Neg)) or (sentiment== "NEG" and (pos_log_likelihood<neg_log_likelihood or Prob_0_Pos)):
                self.predictions.append("+")
            else:
                self.predictions.append("-")
            

        



class SVM(Evaluation):
    """
    general svm class to be extended by text-based classifiers.
    """
    def __init__(self,svm_dir,perf=False):
        self.predictions=[]
        self.svm_dir=svm_dir
        self.perf=perf

    def writeFeatureFile(self,data,filename):
        """
        write local file in svmlight data format.
        see http://svmlight.joachims.org/ for description of data format.

        @param data: input data
        @type data: list of (string, list) tuples where string is the label and list are features in (id, value) tuples (vectors)

        @param filename: name of file to write
        @type filename: string
        """

        f = open(self.svm_dir+'\\NLP_practical\\'+filename, "w")
        for label, feature in data:
            f.write(label)
            
            for id, value in feature:
                f.write(" "+str(id)+":"+str(value))
            f.write('\n')
        # potentially one too many line break, but seems to work

            
    def train(self,train_data):
        """
        train svm 

        @param train_data: training data        (in practice, reviews as before)
        @type train_data: list of (string, list) tuples corresponding to (label, content)
        """
        # function to determine features in training set. to be implemented by child 
        self.getFeatures(train_data)
        # function to find vectors (feature, value pairs). to be implemented by child
        train_vectors=self.getVectors(train_data)
        self.writeFeatureFile(train_vectors,"train.data")
        # train SVM model
        if self.perf:
            call([self.svm_dir+"svm_perf_learn","-c"," 1","-b","0",self.svm_dir+"NLP_practical\\train.data",self.svm_dir+"NLP_practical\\svm_model"],shell=True,stdout=open(os.devnull,'wb'))
        else:
            call([self.svm_dir+"svm_learn",self.svm_dir+"NLP_practical\\train.data",self.svm_dir+"NLP_practical\\svm_model"],shell=True,stdout=open(os.devnull,'wb'))
        
        

    def test(self,test_data):
        """
        test svm 

        @param test_data: test data         (in practice, reviews)
        @type test_data: list of (string, list) tuples corresponding to (label, content)    (in practice reviews)
        """
        # reset predictions
        self.predictions=[]

        # function to find vectors (feature, value pairs). to be implemented by child
        # rmk: only features appearing in the training data will be considered
        test_vectors=self.getVectors(test_data)
        self.writeFeatureFile(test_vectors,"test.data")
        # test SVM model
        if self.perf:
            call([self.svm_dir+"svm_perf_classify",self.svm_dir+"NLP_practical\\test.data",self.svm_dir+"NLP_practical\\svm_model",self.svm_dir+"NLP_practical\\svm_predictions.data"],shell=True,stdout=open(os.devnull,'wb'))
        else:
            call([self.svm_dir+"svm_classify",self.svm_dir+"NLP_practical\\test.data",self.svm_dir+"NLP_practical\\svm_model",self.svm_dir+"NLP_practical\\svm_predictions.data"],shell=True,stdout=open(os.devnull,'wb'))
        # read and store predictions

        for line_index, line in enumerate(open(self.svm_dir+"NLP_practical\\svm_predictions.data","r", encoding="utf-8")):    # does not read last empty line
            if (test_vectors[line_index][0]=="1" and float(line)>=0) or (test_vectors[line_index][0]=="-1" and float(line)<0):
                self.predictions.append("+")
            else:
                self.predictions.append("-")

        



class SVMText(SVM):
    def __init__(self,bigrams,trigrams,svm_dir,discard_closed_class,no_unigrams=False,perf=False):
        """ 
        initialisation of SVMText object

        @param bigrams: add bigrams?
        @type bigrams: boolean

        @param trigrams: add trigrams?
        @type trigrams: boolean

        @param svm_dir: location of smv binaries
        @type svm_dir: string

        @param svm_dir: location of smv binaries
        @type svm_dir: string

        @param discard_closed_class: restrict unigrams to nouns, adjectives, adverbs and verbs?
        @type discard_closed_class: boolean
        """
        SVM.__init__(self,svm_dir,perf)
        self.vocabulary=set()
        # add in bigrams?
        self.bigrams=bigrams
        # add in trigrams?
        self.trigrams=trigrams
        # keep unigrams?
        self.no_unigrams=no_unigrams
        # restrict to nouns, adjectives, adverbs and verbs?
        self.discard_closed_class=discard_closed_class

    def getFeatures(self,reviews):
        """
        determine features from training reviews and store in self.vocabulary.

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)
        """
        # reset for each training iteration
        self.vocabulary=set()
        for sentiment,review in reviews:
            for token in self.extractReviewTokens(review): 
                self.vocabulary.add(token)
        # using dictionary of vocabulary:index for constant order
        # features for SVM are stored as: (feature id, feature value)
        # using index+1 as a feature id cannot be 0 for SVM
        self.vocabulary={token:index+1 for index,token in enumerate(self.vocabulary)}

    def extractReviewTokens(self,review):
        """
        extract tokens from reviews.

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)

        @return: list of strings
        """
        text=[]

        if self.no_unigrams==False:
            for term in review:
                # check if pos tags are included in review e.g. ("bad","JJ")
                # les lignes séparant les phrases ne contiennent rien, d'où la première ligne
                if len(term)==2 and self.discard_closed_class:
                    if term[1][0:2] in ["NN","JJ","RB","VB"]: text.append(term)
                else:
                    text.append(term)
        if self.bigrams:
            for bigram in ngrams(review,2): text.append(bigram)
        if self.trigrams:
            for trigram in ngrams(review,3): text.append(trigram)
        return text

    def getVectors(self,reviews):
        """
        get vectors for svm from reviews.

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)

        @return: list of (string, list) tuples where string is the label ("1"/"-1") and list
                 contains the features in svm format e.g. ("1",[(1, 0.04), (2, 4.0), ...])
                 svm feature format is: (id, value) and id must be > 0.

                only considers features that are already in self.vocabulary

                value associated to a feature is the number of occurences in reviews

        """
        vectors=[]
        for sentiment,review in reviews:
            # temporary variable, counts the number of occurences in the considered review of each token in self.vocabulary, indexed using self.vocabulary (with a shift of -1)
            tokens_count=[0]*len(self.vocabulary)
            for token in self.extractReviewTokens(review):
                if token in self.vocabulary:
                    tokens_count[self.vocabulary[token]-1]=+1  
            # normalize to improve performance (according to Pang et al)
            normalized_tokens_count=array(tokens_count)
            normalized_tokens_count=normalized_tokens_count/linalg.norm(normalized_tokens_count)

            if sentiment=="POS":            # to be sure that they appear in the right order
                vectors.append(("1",[(i+1,value) for i, value in enumerate(normalized_tokens_count) if value!=0]))
            if sentiment=="NEG":            # to be sure that they appear in the right order
                vectors.append(("-1",[(i+1,value) for i, value in enumerate(normalized_tokens_count) if value!=0]))
            
        return vectors
        
