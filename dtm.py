import numpy as np
import pandas as pd
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
import nltk
from nltk.stem.snowball import SnowballStemmer


ob_regex = r"[^a-z\s]"

def tokenize_ob(text_series):
    '''
    Cleans, tokenizes + stems Pandas series of strings from
    the dataset.
    
    Returns pandas series of lists of tokens
    '''
    # Clean text with regex
    clean = text_series.str.lower() \
                       .str.replace(ob_regex,
                                    "",
                                    regex=True)

    # Anonymous tokenizer + stemmer functions
    stop = nltk.corpus.stopwords.words('english')
    tokenize = lambda text: [i for i in nltk.word_tokenize(text) if i not in stop]
    stemmer = lambda tokens: [SnowballStemmer('english').stem(token) for token in tokens]

    # Tokenize and stem clean text
    tokens = clean.apply(tokenize)
    stemmed_tokens = tokens.apply(stemmer)
    
    return stemmed_tokens

def prepare_data(tokens):
    '''
    Prepares Pandas series of lists of tokens for use within a Gensim topic model
    
    Returns an id2word dictionary + bag of words corpus
    '''
    
    dictionary = corpora.Dictionary([i for i in tokens])
s
    bow_corpus = [dictionary.doc2bow(text) for text in tokens]
    
    return dictionary, bow_corpus

def print_dtm(dtm, n_topics, n_time_slices):
    '''
    Prints out top words in each topic across time slices for visual comparison
    
    Input: Gensim LdaSeqModel, n_topics (int), n_time_slices (int)
    '''
    for topic in range(n_topics):
        for time in range(n_time_slices): 
            print("##### Topic {}, Time Slice {} #####".format(topic, time))    
            print(dtm.dtm_coherence(time)[topic][:10])    
        print("\n")

def dtm_coherence(list_dtms, bow_corpus, dictionary, n_time_slices):
    '''
    Computes UMass Coherence for each time slice in a list of DTMs
    
    Input: List of Gensim LdaSeqModels, number of time slices modeled (int)
    Returns: Dict of lists of coherence scores for each DTM
    '''
    coherence = {}
    for i, ldaseq in enumerate(list_dtms):
        coherence[i] = []
        for t in range(n_time_slices):
            topics_dtm = ldaseq.dtm_coherence(t)
            cm_DTM = CoherenceModel(topics=topics_dtm,
                                    corpus=bow_corpus,
                                    dictionary=dictionary,
                                    coherence='u_mass')

            coherence[i].append(cm_DTM.get_coherence())
            
        return coherence