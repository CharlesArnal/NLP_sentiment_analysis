import math,sys

class Evaluation():
    """ 
    general evaluation class implemented by classifiers 
    """
    def __init__(self):

        self.predictions=[]

    def crossValidate(self,corpus):
        """
        Remarque: n'a probablement pas de sens pour le Lexicon classifier (qu'on n'entraîne pas)

        function to perform 10-fold cross-validation for a classifier. 
        each classifier will be inheriting from the evaluation class so you will have access
        to the classifier's train and test functions. 

        1. read reviews from corpus.folds and store 9 folds in train_files and 1 in test_files 
        2. pass data to self.train and self.test e.g., self.train(train_files)
        3. repeat for another 9 runs making sure to test on a different fold each time

        Reminder: self.test deletes all previous predictions

        @param corpus: corpus of movie reviews
        @type corpus: MovieReviewCorpus object
        """
        # reset predictions
        self.predictions=[]
        total_predictions=[]
        i=0
        for fold_test in corpus.folds:
            i+=1
            print(i)
            train_files=[]
            test_files=[]
            for fold in corpus.folds:
                if fold ==fold_test:
                    test_files=corpus.folds[fold]
                else:
                    train_files.extend(corpus.folds[fold])

            self.train(train_files)
            self.test(test_files)
            total_predictions.extend(self.predictions)
        
        self.predictions=total_predictions



        
        



    def getStdDeviation(self):
        """
        get standard deviation across folds in cross-validation.
        (c'est à dire la variance de la précision de prédiction d'un fold à l'autre (avec comme moyenne celle sur tous les folds))
        """
        # get the avg accuracy and initialize square deviations
        avgAccuracy,square_deviations=self.getAccuracy(),0
        # find the number of instances in each fold
        fold_size=int(len(self.predictions)/10)
        # calculate the sum of the square deviations from mean
        for fold in range(0,len(self.predictions),fold_size):
            square_deviations+=(self.predictions[fold:fold+fold_size].count("+")/float(fold_size) - avgAccuracy)**2
        # std deviation is the square root of the variance (mean of square deviations)
        return math.sqrt(square_deviations/10.0)

    def getAccuracy(self):
        """
        get accuracy of classifier. 

        @return: float containing percentage correct
        """
        # note: data set is balanced so just taking number of correctly classified over total
        # "+" = correctly classified and "-" = error
        return self.predictions.count("+")/float(len(self.predictions))
