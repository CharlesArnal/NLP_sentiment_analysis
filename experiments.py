import subprocess
import os
from Corpora import MovieReviewCorpus , IMBd_Doc2Vec_Corpus, text_curator
from Lexicon import SentimentLexicon
from Statistics import SignTest
from Classifiers import NaiveBayesText, SVMText
from Extensions import SVMDoc2Vec
#from Visualization import TF_visualizer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from gensim.models import doc2vec
from random import shuffle
import time
#assert doc2vec.FAST_VERSION > -1    # Does not work for some reason (apparently gensim version)
import multiprocessing                           #good idea?
import csv
from math import floor

#import tensorflow as tf
#import tensorflow_datasets as tfds
#from tensorboard.plugins import projector
"""
print ("--- Readying corpora ---")
# retrieve corpus
corpus=MovieReviewCorpus(stemming=False,pos=False)
# retrieve corpus with tokenized text and stemming (using porter)
stemmed_corpus=MovieReviewCorpus(stemming=True,pos=False)
# retrieve corpus with POS and no stemming
POS_corpus=MovieReviewCorpus(stemming=False,pos=True)

"""
# use sign test for all significance testing
signTest=SignTest()

# location of svmlight binaries 
svmlight_dir="C:\\Users\\Charles Arnal\\Desktop\\NLP\\Practical\\svm_light_windows64\\"
#svmperf_dir="C:\\Users\\Charles Arnal\\Desktop\\NLP\\Practical\\svm_perf_windows\\"

"""
print("--- classifying reviews using sentiment lexicon  ---")


# read in lexicon
lexicon=SentimentLexicon()

# on average there are more positive than negative words per review (~7.13 more positive than negative per review)
# to take this bias into account will use threshold (roughly the bias itself) to make it harder to classify as positive
threshold=8

# question 0.1
lexicon.classify(corpus.reviews,threshold,magnitude=False)

token_preds=lexicon.predictions
print("token-only results: %.3f" % lexicon.getAccuracy())

lexicon.classify(corpus.reviews,threshold,magnitude=True,weight=3)
magnitude_preds=lexicon.predictions
print( "magnitude results: %.3f" % lexicon.getAccuracy())


# question 0.2
p_value=signTest.getSignificance(token_preds,magnitude_preds)
print ("magnitude lexicon results are",("significant" if p_value < 0.05 else "not significant"),"with respect to token-only","(p=%.8f)" % p_value)

"""

"""
# question 1.0
print ("--- classifying reviews using Naive Bayes on held-out test set ---")
NB=NaiveBayesText(smoothing=False,bigrams=False,trigrams=False,discard_closed_class=False)
NB.train(corpus.train)
NB.test(corpus.test)
# store predictions from classifier
non_smoothed_preds=NB.predictions
print ("Accuracy without smoothing: %.3f" % NB.getAccuracy())



# question 2.0
# use smoothing
NB=NaiveBayesText(smoothing=True,bigrams=False,trigrams=False,discard_closed_class=False,smoothing_constant=1)
NB.train(corpus.train)
NB.test(corpus.test)
smoothed_preds=NB.predictions
# saving this for use later
num_non_stemmed_features=len(NB.vocabulary)
print( "Accuracy using smoothing: %.3f" % NB.getAccuracy())



# question 2.1
# see if smoothing significantly improves results
p_value=signTest.getSignificance(non_smoothed_preds,smoothed_preds)
print ("results using smoothing are",("significant" if p_value < 0.05 else "not significant"),"with respect to no smoothing","(p=%.8f)" % p_value)



# question 3.0
NB=NaiveBayesText(smoothing=True,bigrams=False,trigrams=False,discard_closed_class=False,smoothing_constant=1)
bigrams_only_NB=NaiveBayesText(smoothing=True,bigrams=True,trigrams=False,discard_closed_class=False,smoothing_constant=1,no_unigrams=True)
trigrams_only_NB=NaiveBayesText(smoothing=True,bigrams=False,trigrams=True,discard_closed_class=False,smoothing_constant=1,no_unigrams=True)
one_to_three_grams_NB=NaiveBayesText(smoothing=True,bigrams=True,trigrams=True,discard_closed_class=False,smoothing_constant=1,no_unigrams=False)
OC_BN=NaiveBayesText(smoothing=True,bigrams=False,trigrams=False,discard_closed_class=True,smoothing_constant=1)


#  Q4.2
# Compare the number of features for different configurations
print ("--- determining the number of features for various configurations ---")
NB.train(corpus.test)
print(f"Words only: {len(NB.vocabulary)}")
NB.train(stemmed_corpus.test)
print(f"Stemmed words: {len(NB.vocabulary)}")
bigrams_only_NB.train(corpus.test)
print(f"Bigrams only: {len(bigrams_only_NB.vocabulary)}")
trigrams_only_NB.train(corpus.test)
print(f"Trigrams only: {len(trigrams_only_NB.vocabulary)}")
one_to_three_grams_NB.train(corpus.test)
print(f"One to three-grams: {len(one_to_three_grams_NB.vocabulary)}")
NB.train(POS_corpus.test)
print(f"Words+POS only: {len(NB.vocabulary)}")
OC_BN.train(POS_corpus.test)
print(f"Open classes words + POS: {len(OC_BN.vocabulary)}")


print("For whole training set")
NB.train(corpus.reviews)
print(f"Words only: {len(NB.vocabulary)}")
bigrams_only_NB.train(corpus.reviews)
print(f"Bigrams only: {len(bigrams_only_NB.vocabulary)}")
trigrams_only_NB.train(corpus.reviews)
print(f"Trigrams only: {len(trigrams_only_NB.vocabulary)}")
print("For whole training set from IMBd")
NB.train(MyIMBdCorpus.reviews)
print(f"Words only: {len(NB.vocabulary)}")
bigrams_only_NB.train(MyIMBdCorpus.reviews)
print(f"Bigrams only: {len(bigrams_only_NB.vocabulary)}")
trigrams_only_NB.train(MyIMBdCorpus.reviews)
print(f"Trigrams only: {len(trigrams_only_NB.vocabulary)}")

"""

# smoothed NB cross-validation for different types of features
"""


print ("--- cross-validating NB with words ---")
# using previous instantiated object
NB.crossValidate(corpus)
# using cross-eval for smoothed predictions from now on
smoothed_preds=NB.predictions
print ("Accuracy: %.3f" % NB.getAccuracy())
print ("Std. Dev: %.3f" % NB.getStdDeviation())



# question 4.0

print ("--- cross-validating NB using stemming ---")
NB.crossValidate(stemmed_corpus)
stemmed_preds=NB.predictions
print ("Accuracy: %.3f" % NB.getAccuracy())
print ("Std. Dev: %.3f" % NB.getStdDeviation())


# question Q5.0
# cross-validate model using smoothing and bigrams (and no stemming)
print ("--- cross-validating Naive Bayes using smoothing and bigrams only ---")
bigrams_only_NB.crossValidate(corpus)
smoothed_and_bigram_preds=bigrams_only_NB.predictions
print ("Accuracy: %.3f" % bigrams_only_NB.getAccuracy())
print ("Std. Dev: %.3f" % bigrams_only_NB.getStdDeviation())

# cross-validate model using smoothing and trigrams (and no stemming)
print ("--- cross-validating Naive Bayes using smoothing and trigrams only ---")
trigrams_only_NB.crossValidate(corpus)
smoothed_and_trigram_preds=trigrams_only_NB.predictions
print ("Accuracy: %.3f" % trigrams_only_NB.getAccuracy())
print ("Std. Dev: %.3f" % trigrams_only_NB.getStdDeviation())



# cross-validate model using smoothing and 1 to 3 grams (and no stemming)
print ("--- cross-validating Naive Bayes using smoothing and 1 to 3 grams ---")
one_to_three_grams_NB.crossValidate(corpus)
smoothed_and_all_grams_preds=one_to_three_grams_NB.predictions
print ("Accuracy: %.3f" % one_to_three_grams_NB.getAccuracy())
print ("Std. Dev: %.3f" % one_to_three_grams_NB.getStdDeviation())


# cross-validate model using smoothing and POS 
print ("--- cross-validating Naive Bayes using unigrams and POS ---")
NB.crossValidate(POS_corpus)
smoothed_and_POS_preds=NB.predictions
print ("Accuracy: %.3f" % NB.getAccuracy())
print ("Std. Dev: %.3f" % NB.getStdDeviation())

# cross-validate model using smoothing and POS 
print ("--- cross-validating Naive Bayes using open classes words only and POS ---")
OC_BN.crossValidate(POS_corpus)
smoothed_and_OC_classes_preds=OC_BN.predictions
print ("Accuracy: %.3f" % OC_BN.getAccuracy())
print ("Std. Dev: %.3f" % OC_BN.getStdDeviation())




# Initialize SVM with various parameters

SVM=SVMText(bigrams=False,trigrams=False,svm_dir=svmlight_dir,discard_closed_class=False,no_unigrams=False)
bigrams_only_SVM=SVMText(bigrams=True,trigrams=False,svm_dir=svmlight_dir,discard_closed_class=False,no_unigrams=True)
trigrams_only_SVM=SVMText(bigrams=False,trigrams=True,svm_dir=svmlight_dir,discard_closed_class=False,no_unigrams=True)
one_to_three_grams_SVM=SVMText(bigrams=True,trigrams=True,svm_dir=svmlight_dir,discard_closed_class=False,no_unigrams=False)
OC_SVM=SVMText(bigrams=False,trigrams=False,svm_dir=svmlight_dir,discard_closed_class=True,no_unigrams=False)


# SVM cross-validation for different types of features

# Q6 and 6.1
print ("--- cross-validating SVM ---")
SVM.crossValidate(corpus)
# store predictions from classifier
SVM_preds=SVM.predictions
print ("Accuracy: %.3f" % SVM.getAccuracy())
print ("Std. Dev: %.3f" % SVM.getStdDeviation())


print ("--- cross-validating SVM with stemming ---")
SVM.crossValidate(stemmed_corpus)
# store predictions from classifier
stemmed_SVM_preds=SVM.predictions
print ("Accuracy: %.3f" % SVM.getAccuracy())
print ("Std. Dev: %.3f" % SVM.getStdDeviation())


print ("--- cross-validating SVM with bigrams only ---")
bigrams_only_SVM.crossValidate(corpus)
# store predictions from classifier
bigrams_only_SVM_preds=bigrams_only_SVM.predictions
print ("Accuracy: %.3f" % bigrams_only_SVM.getAccuracy())
print ("Std. Dev: %.3f" % bigrams_only_SVM.getStdDeviation())

print ("--- cross-validating SVM with trigrams only ---")
trigrams_only_SVM.crossValidate(corpus)
# store predictions from classifier
trigrams_only_SVM_preds=trigrams_only_SVM.predictions
print ("Accuracy: %.3f" % trigrams_only_SVM.getAccuracy())
print ("Std. Dev: %.3f" % trigrams_only_SVM.getStdDeviation())

print ("--- cross-validating SVM with 1 to 3-grams ---")
one_to_three_grams_SVM.crossValidate(corpus)
# store predictions from classifier
one_to_three_grams_SVM_preds=one_to_three_grams_SVM.predictions
print ("Accuracy: %.3f" % one_to_three_grams_SVM.getAccuracy())
print ("Std. Dev: %.3f" % one_to_three_grams_SVM.getStdDeviation())

print ("--- cross-validating SVM with unigrams + POS ---")
SVM.crossValidate(POS_corpus)
# store predictions from classifier
POS_SVM_preds=SVM.predictions
print ("Accuracy: %.3f" % SVM.getAccuracy())
print ("Std. Dev: %.3f" % SVM.getStdDeviation())

print ("--- cross-validating SVM with open classes unigrams + POS ---")
OC_SVM.crossValidate(POS_corpus)
# store predictions from classifier
OC_SVM_preds=OC_SVM.predictions
print ("Accuracy: %.3f" % OC_SVM.getAccuracy())
print ("Std. Dev: %.3f" % OC_SVM.getStdDeviation())


"""
# COMPARISONS ----------


"""
# Compare NB and sentiment prediction
p_value=signTest.getSignificance(token_preds,smoothed_preds)
print ("results using smoothed NB are",("significant" if p_value < 0.05 else "not significant"),"with respect to using sentiment lexicon","(p=%.8f)" % p_value)


# Q4.1
# see if stemming significantly improves results on smoothed NB
p_value=signTest.getSignificance(smoothed_preds,stemmed_preds)
print ("results using stemming and smoothing are",("significant" if p_value < 0.05 else "not significant"),"with respect to simply smoothing","(p=%.8f)" % p_value)


# see if bigrams significantly improves results on smoothed NB only
p_value=signTest.getSignificance(smoothed_preds,smoothed_and_bigram_preds)
print ("results using smoothing and bigrams only are",("significant" if p_value < 0.05 else "not significant"),"with respect to smoothing only","(p=%.8f)" % p_value)

# see if SVM outperforms smoothed BN
p_value=signTest.getSignificance(smoothed_preds,SVM_preds)
print ("results using SVM (without POS) are",("significant" if p_value < 0.05 else "not significant"),"with respect to BN with smoothing","(p=%.8f)" % p_value)


"""


# question 8.0
print ("--- Using document embeddings ---")

#TODO check word learning method
"""
# 1st model
print("--- Model 1 ---")

t1=time.time()
print("--- Create corpus ---")
Doc2Vec_Corpus=IMBd_Doc2Vec_Corpus( stemming=True )
print("--- Initialize Doc2Vec model ---")
docu=[doc2vec.TaggedDocument(review,[index]) for index, (sentiment, review) in   enumerate(Doc2Vec_Corpus.reviews)]
shuffle(docu)
model_1=doc2vec.Doc2Vec(vector_size=30, dm=1, dm_concat=1, epochs=200, window=5, min_count=2, workers=8 ) 
print("--- Build vocabulary of Doc2Vec model ---")
model_1.build_vocab(documents=docu)
print("--- Train Doc2Vec model ---")
model_1.train(documents=docu, total_examples=model_1.corpus_count,epochs=model_1.epochs)


print("--- Train SVM ---")
My_SVMDoc2Vec=SVMDoc2Vec(model_1,svmlight_dir)
My_SVMDoc2Vec.train(Doc2Vec_Corpus.train)
print("--- Test SVM ---")
My_SVMDoc2Vec.test(Doc2Vec_Corpus.test)
print( "Accuracy using Doc2Vec: %.3f" % My_SVMDoc2Vec.getAccuracy())
print(time.time()-t1)
model_1_preds=My_SVMDoc2Vec.predictions


# Compare with smoothed NB

t2=time.time()
NB=NaiveBayesText(smoothing=True,bigrams=False,trigrams=False,discard_closed_class=False,smoothing_constant=1)
print("--- Training NB ---")
NB.train(Doc2Vec_Corpus.train)
print("--- Testing NB ---")
NB.test(Doc2Vec_Corpus.test)
print ("Accuracy with smoothing: %.3f" % NB.getAccuracy())
print(time.time()-t2)
NB_preds=NB.predictions


p_value=signTest.getSignificance(NB_preds,model_1_preds)
print ("results using Doc2Vec are",("significant" if p_value < 0.05 else "not significant"),"with respect to smoothed NB","(p=%.8f)" % p_value)


# Compare with SVM

t3=time.time()
SVM=SVMText(bigrams=False,trigrams=False,svm_dir=svmlight_dir,discard_closed_class=False,no_unigrams=False)
print("--- Training SVM ---")
SVM.train(Doc2Vec_Corpus.train)
print("--- Testing SVM ---")
SVM.test(Doc2Vec_Corpus.test)
print(time.time()-t1)
print ("Accuracy: %.3f" % SVM.getAccuracy())
print(time.time()-t3)
SVM_preds=SVM.predictions

p_value=signTest.getSignificance(SVM_preds,model_1_preds)
print ("results using Doc2Vec are",("significant" if p_value < 0.05 else "not significant"),"with respect to SVM","(p=%.8f)" % p_value)
"""





"""
# vector visualization


print("--- Allowing vector visualization ---")
chemin="C:\\Users\\Charles Arnal\\Desktop\\NLP\\Practical\\Tensorboard"


test_Words=["good","excellent","interesting","better","liked","entertaining","bad","hated","boring","worst","Tarentino"]



doc_vectors = [model_1.infer_vector(review) for sentiment,review in Doc2Vec_Corpus.reviews]
word_vectors = [model_1.wv[PorterStemmer().stem(word)] for word in test_Words]

vectors = word_vectors +doc_vectors      # load your embeddings
doc_metadata = [[sentiment,0] for sentiment,review in Doc2Vec_Corpus.reviews]  # load your metadata
word_metadata = [[PorterStemmer().stem(word),1] for word in test_Words]
metadata=[["sentiment/name","word?"]]+word_metadata+doc_metadata

out_file=open(chemin+'\\testData.tsv', 'wt')
tsv_writer = csv.writer(out_file, delimiter='\t')
for vector in vectors:
    tsv_writer.writerow(vector)

out_file=open(chemin+'\\testMetadata.tsv', 'wt')
tsv_writer = csv.writer(out_file, delimiter='\t')
for meta in metadata:
    tsv_writer.writerow(meta)

"""



"""
#curate text:
curator=text_curator(keep_stopwords=True,keep_punctuation=True)
main_dir="C:\\Users\\Charles Arnal\\Desktop\\NLP\\Practical\\Downloaded\\data\\Extension"
data_directories=[main_dir+"\\train\\pos",main_dir+"\\train\\neg",main_dir+"\\test\\pos",main_dir+"\\test\\neg"]
curator.curate(data_directories[0],svmlight_dir+"Extension\\train\\pos")
curator.curate(data_directories[1],svmlight_dir+"Extension\\train\\neg")
curator.curate(data_directories[2],svmlight_dir+"Extension\\test\\pos")
curator.curate(data_directories[3],svmlight_dir+"Extension\\test\\neg")
"""


