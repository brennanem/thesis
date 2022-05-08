#file containing functions for saving and dowloading data

#import packages
import numpy as np
import pandas as pd
import os
#import functions
from my_modules.text_prep import get_tokens

#FUNCTION: get_scores_csv
#RETURNS: this function downloads the saved scores data for each speech and returns
#       a dict for which each key is the name of the speech, and the corresponding
#       value is another dict for which each key is a sentiment score and the corresponding
#       value is a list of words with this sentiment scores
def get_scores_csv():
    all_scores = {}
    for filename in os.listdir("Source/word_sentiments"): #for each speech, 
        if not filename.startswith("."):
            #download csv as pandas DataFrame
            df = pd.read_csv("Source/word_sentiments/" + filename)
            df = df.rename(columns={'-1.0':-1, '-0.5':-0.5, '0.0':0, '0.5':0.5, '1.0':1})
            #convert DataFrame into dictionary
            scores = {col: df[col].dropna().tolist() for col in df.columns}
            #put dictionary for this speech into overall dict for all speeches
            all_scores[filename] = scores
    return all_scores

#FUNCTION: get_all_embeddings_csv
#RETURNS: this function downloads the saved embeddings data for each speech and returns
#       a dict for which each key is the name of the speech, and the corresponding
#       value is another dict for which each key is a unique lemma and the corresponding
#       value is its embedding vector
def get_all_embeddings_csv():
    all_embeddings = {}
    for filename in os.listdir("Source/embeddings"): #for each speech
        if not filename.startswith("."):
            #download csv as pandas DataFrame
            df = pd.read_csv("Source/embeddings/" + filename, dtype=np.float64)
            #convert DataFrame into dictionary
            embeddings = {col: df[col].dropna().tolist() for col in df.columns}
            #convert embegginds into numpy array of type 64-bit float
            for lemma in embeddings:
                embeddings[lemma] = np.array(embeddings[lemma], dtype=np.float64)
            #put dictionary for this speech into overall dict for all speeches
            all_embeddings[filename] = embeddings
    return all_embeddings

#FUNCTION: get_lemmata_csv
#RETURNS: this function downloads the saved lemmata data for each speech and returns
#       a dict for which each key is a list of lemma for the speech
def get_lemmata_csv():
    all_lemmata = {}
    for filename in os.listdir("Source/lemmata"): #for each speech
        if not filename.startswith("."):
            #download csv as pandas DataFrame
            df = pd.read_csv("Source/lemmata/" + filename)
            #convert DataFrame into list
            lemmata = df["0"].tolist()
            #put list of lemma for this speech into dict for all speeches
            all_lemmata[filename] = lemmata
    return all_lemmata

#FUNCTION: save_scores_csv
#RETURNS: this function saves as csv file the sentiment scores and the
#       corresponding list of lemma with each score for the given speech
def save_scores_csv(scores, filename):
    df = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in scores.items()]))
    df.to_csv("Source/word_sentiments/" + filename, index=False)

#FUNCTION: save_lemmata_csv
#RETURNS: this function saves as csv file the list of lemmata for the given speech
def save_lemmata_csv(lemmata, filename):
    lemmata_series = pd.Series(lemmata)
    lemmata_series.to_csv("Source/lemmata/" + filename, index=False)  

#FUNCTION: save_embeddings_csv
#RETURNS: this function saves as csv file the embeddings vectors and the
#       for each lemma in the given speech
def save_embeddings_csv(lemmata, words, filename):
    embeddings = {}
    tokens = get_tokens(lemmata, words)
    for word, lemma in zip(words, lemmata):
        if lemma in tokens: #only save non-stop words
            if lemma in embeddings:
                embeddings[lemma].append(word.embedding)
            else:
                embeddings[lemma] = [word.embedding]

    for lemma in embeddings:
        assert(np.shape(embeddings[lemma][0]) == (300,)), "shape not (300,)!"
        #if multiple occurences of the save lemma, calculate average of all embeddigns
        if len(embeddings[lemma]) > 1:
            avge_emb = np.sum(embeddings[lemma], axis=0)/len(embeddings[lemma])
            embeddings[lemma] = avge_emb
        else:
            embeddings[lemma] = embeddings[lemma][0]

    df = pd.DataFrame.from_dict(embeddings, dtype=np.float64)
    df.to_csv("Source/embeddings_only_nouns/" + filename, index=False)