from Analysis import Evaluation

class SentimentLexicon(Evaluation):
    def __init__(self):
        """
        read in lexicon database and store in self.lexicon
        """
        Evaluation.__init__(self)

        # if multiple entries take last entry by default
        self.lexicon=dict([[l.split()[2].split("=")[1],l.split()] for l in open("C:\\Users\\Charles Arnal\\Desktop\\NLP\\Practical\\Downloaded\\data\\sent_lexicon","r")])
                            #le mot lui-même           toutes les autres infos sous forme de liste             
    def classify(self,reviews,threshold,magnitude=False,weight=3):
        """
        classify movie reviews using self.lexicon.
        self.lexicon is a dictionary of word: [polarity_info, magnitude_info], e.g. "bad": ["priorpolarity=negative","type=strongsubj"].
        explore data/sent_lexicon to get a better understanding of the sentiment lexicon.
        store the predictions in self.predictions as a list of strings where "+" and "-" are correct/incorrect classifications respectively e.g. ["+","-","+",...]

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)

        @param threshold: threshold to center decisions on. instead of using 0, there may be a bias in the reviews themselves which could be accounted for. 
                          experiment for good threshold values.
        @type threshold: integer
        
        @type magnitude: use magnitude information from self.lexicon?
        @param magnitude: boolean
        """

        self.predictions=[]
        
        for review in reviews:
            TotalSentiment=0
            for wordPlusTag in review[1]:
                if isinstance(wordPlusTag, str):
                    word=wordPlusTag
                elif isinstance(wordPlusTag, list):
                    word=wordPlusTag[0]
                if word in self.lexicon:
                    if self.lexicon[word][5] == "priorpolarity=positive":
                        if magnitude:
                            if  self.lexicon[word][0] == "type=strongsubj":
                                TotalSentiment+=weight
                            elif self.lexicon[word][0] == "type=weaksubj":
                                TotalSentiment+=1
                        else:
                            TotalSentiment+=1
                    elif self.lexicon[word][5] == "priorpolarity=negative":
                        if magnitude:
                            if  self.lexicon[word][0] == "type=strongsubj":
                                TotalSentiment-=weight
                            elif self.lexicon[word][0] == "type=weaksubj":
                                TotalSentiment-=1
                        else:
                            TotalSentiment-=1       
            

                         

            
            if (review[0]=="POS" and TotalSentiment>=threshold) or (review[0]=="NEG" and TotalSentiment<threshold):
                self.predictions.append("+")
            else:
                self.predictions.append("-")

