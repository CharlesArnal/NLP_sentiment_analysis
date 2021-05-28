import os, codecs, sys
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
# nltk.download('stopwords')        # Unclear what supposed to do


class MovieReviewCorpus():
    def __init__(self,stemming,pos):
        """
        initialisation of movie review corpus.

        @param stemming: use porter's stemming?
        @type stemming: boolean

        @param pos: use pos tagging?
        @type pos: boolean
        """

        # raw movie reviews
        self.reviews=[]               
        # held-out train/test set
        self.train=[]
        self.test=[]
        # folds for cross-validation, using Round-Robin splitting
        self.folds={}
        # porter stemmer
        self.stemmer=PorterStemmer() if stemming else None
        # part-of-speech tags
        self.pos=pos
        # import movie reviews
        self.get_reviews()

    def get_reviews(self):
        """
        processing of movie reviews.

        1. parse reviews in data/reviews and store in self.reviews.

           the format expected for reviews is: [(string,list), ...] e.g. [("POS",["a","good","movie"]), ("NEG",["a","bad","movie"])].
           in data/reviews there are .tag and .txt files. The .txt files contain the raw reviews and .tag files contain tokenized and pos-tagged reviews.

           to save effort, we recommend you use the .tag files. you can disregard the pos tags to begin with and include them later.
           when storing the pos tags, please use the format for each review: ("POS/NEG", [(token, pos-tag), ...]) e.g. [("POS",[("a","DT"), ("good","JJ"), ...])]

           to use the stemmer the command is: self.stemmer.stem(token)
           
        2. store training and held-out reviews in self.train/test. files beginning with cv9 go in self.test and others in self.train

        3. store reviews in self.folds. self.folds is a dictionary with the format: self.folds[fold_number] where fold_number is an int 0-9.
           you can get the fold number from the review file name.
        """

        # Get reviews from files, store them in reviews, test and train, and folds
        self.reviews=[]
        # Keeps the POS tags or discards them depending on self.pos
        # Negative reviews
        self.folds.update([str(i),[]] for i in range(10))
        for entry in os.scandir("C:\\Users\\Charles Arnal\\Desktop\\NLP\\Practical\\Downloaded\\data\\reviews\\NEG"):
            if entry.path.endswith(".tag"): 
                if self.pos:
                    if self.stemmer!=None:
                        review=("NEG", [(self.stemmer.stem(l.split()[0]),l.split()[1]) for l in open(entry.path,"r", encoding="utf-8") if len(l.split())==2])
                    else:
                        review=("NEG", [(l.split()[0],l.split()[1]) for l in open(entry.path,"r", encoding="utf-8") if len(l.split())==2])
                else:
                    if self.stemmer!=None:
                        review=("NEG", [self.stemmer.stem(l.split()[0]) for l in open(entry.path,"r", encoding="utf-8") if len(l.split())==2])
                    else:
                        review=("NEG", [l.split()[0] for l in open(entry.path,"r", encoding="utf-8") if len(l.split())==2])
                self.reviews.append(review)
                # Training set and testing set
                if entry.path.startswith(r"C:\Users\Charles Arnal\Desktop\NLP\Practical\Downloaded\data\reviews\NEG\cv9"):
                    self.test.append(review)
                else:
                    self.train.append(review)
                # Folds, with Round-robin splitting
                self.folds[entry.path[entry.path.find("cv")+4]].append(review)


        # Positive reviews
        for entry in os.scandir("C:\\Users\\Charles Arnal\\Desktop\\NLP\\Practical\\Downloaded\\data\\reviews\\POS"):
            if entry.path.endswith(".tag"): 
                if self.pos:
                    if self.stemmer!=None:
                        review=("POS", [(self.stemmer.stem(l.split()[0]),l.split()[1]) for l in open(entry.path,"r", encoding="utf-8") if len(l.split())==2])
                    else:
                        review=("POS", [(l.split()[0],l.split()[1]) for l in open(entry.path,"r", encoding="utf-8") if len(l.split())==2])
                else:
                    if self.stemmer!=None:
                        review=("POS", [self.stemmer.stem(l.split()[0]) for l in open(entry.path,"r", encoding="utf-8") if len(l.split())==2])
                    else:
                        review=("POS", [l.split()[0] for l in open(entry.path,"r", encoding="utf-8") if len(l.split())==2])
                self.reviews.append(review)
                # Training set and testing set
                if entry.path.startswith(r"C:\Users\Charles Arnal\Desktop\NLP\Practical\Downloaded\data\reviews\POS\cv9"):
                    self.test.append(review)
                else:
                    self.train.append(review)
                # Folds
                self.folds[entry.path[entry.path.find("cv")+2]].append(review)
        


"""
def tagged_document(list_of_list_of_words):
   for i, list_of_words in enumerate(list_of_list_of_words):
      yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])
      """

# Reads documents from the IMBd, stores reviews in an appropriate way
class IMBd_Doc2Vec_Corpus():
    def __init__(self,stemming=False):
        """
        initialisation of movie review corpus.

        Not using stemming and pos for now

        @param stemming: use porter's stemming?
        @type stemming: boolean

        @param pos: use pos tagging?
        @type pos: boolean
        """

        # raw movie reviews
        self.reviews=[]               
        # held-out train/test set
        self.train=[]
        self.test=[]
        # folds for cross-validation, using Round-Robin splitting
        self.folds={}
        # stemming
        self.stemming=stemming
        # porter stemmer
        self.stemmer=PorterStemmer() if stemming else None
        # import movie reviews
        self.get_reviews()
    
    # Modified get_reviews function adapted to the IMBd
    def get_reviews(self):
        """
        processing of movie reviews.

        1. parse reviews in data/reviews and store in self.reviews.

           the format expected for reviews is: [(string,list), ...] e.g. [("POS",["a","good","movie"]), ("NEG",["a","bad","movie"])].
           
        2. store training and test reviews in self.train/test

        
        3. store reviews in self.folds. self.folds is a dictionary with the format: self.folds[fold_number] where fold_number is an int 0-9.
           you can get the fold number from the review file name.
        """
        
        # Get reviews from files, store them in reviews, test and train, and folds
        self.folds.update([str(i),[]] for i in range(10))
        main_dir="C:\\Users\\Charles Arnal\\Desktop\\NLP\\Practical\\svm_light_windows64\\Extension"
        directories_to_visit=[main_dir+"\\train\\pos",main_dir+"\\train\\neg",main_dir+"\\test\\pos",main_dir+"\\test\\neg"]
        for c_dir in directories_to_visit:
            i=0
            for entry in os.scandir(c_dir):
                i+=1
                if i in [5000,10000]:
                    print(i)
                pos_review=c_dir.endswith("pos")
                review_text=[]
                for l in open(entry.path,"r", encoding="utf-8"):
                    if self.stemming:
                        review_text.append(self.stemmer.stem(l.split()[0]))        # to avoid getting \n
                    else:
                        review_text.append(l.split()[0])
                review=("POS" if pos_review else "NEG", review_text)
                self.reviews.append(review)
                # Training set and testing set (for the SVM, not the Doc2Vec model that trains on every review)
                if c_dir in [main_dir+"\\train\\pos",main_dir+"\\train\\neg"]:
                    
                    self.train.append(review)
                else:
                    self.test.append(review)
                # Folds, with Round-robin splitting 
                self.folds[entry.path[-1]].append(review)
            print(c_dir+" has been treated")


class text_curator():
    def __init__(self,keep_stopwords=False,keep_punctuation=False):
        self.keep_stopwords=keep_stopwords
        self.keep_punctuation=keep_punctuation

    def curate(self,dir,output_dir):
        my_stopwords=stopwords.words("english")
        i=0
        for entry in os.scandir(dir):
            i+=1
            if i in [5000,10000]:
                print(i)
            
            review=[]
            for l in open(entry.path,"r", encoding="utf-8"):
                review+=word_tokenize(l)
            
            f = open(output_dir+"\\"+str(i), "w",encoding="utf-8")
            for word in review:
                valid=True
                if self.keep_punctuation == False and word.isalpha() ==False:
                    valid=False
                if self.keep_stopwords==False and word in my_stopwords:
                    valid=False
                if valid:
                    f.write(word+"\n")

