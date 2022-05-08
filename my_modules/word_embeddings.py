#File containing functions to find most highly associated words

#import packages
import numpy as np
import math
#import functions
from my_modules.sentiment_analysis import get_negative_count, sort_negative


#FUNCTION: cosine_similarity
#RETURNS: Calculates and returns the cosine similarity between two vectors of two words
def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    v1_length = np.sqrt(np.dot(v1, v1))
    v2_length = np.sqrt(np.dot(v2, v2))
    cosine_sim = dot_product / (v1_length * v2_length)
    return cosine_sim   

#FUNCTION: find_associated_words
#RETURNS: given dict of embeddings for specific speech, for each of top_neg
#       negative words, find top_assoc words that are most strongly associated 
#       with current negative word, and return dict of this data
def find_associated_words(embeddings, neg_words, top_neg, top_assoc):
    #get sorted list of top_neg negative words from given list of neg_words
    neg_words_count = get_negative_count(neg_words)
    sorted_neg = sort_negative(neg_words_count) 
    sorted_neg = [pair[0] for pair in sorted_neg if not pair[0].startswith("non ")]
    if top_neg > len(sorted_neg): top_neg = len(sorted_neg)
    all_assoc_words = {}
    for i in range(top_neg): #for each of top_neg negative words
        neg_word = sorted_neg[i]
        if neg_word in embeddings:
            emb = embeddings[neg_word] #get neg word's embedding
            top_words = []
            similarities, associated_words = {}, {}
            for lemma in embeddings: #for each word in speech
                if lemma not in neg_words: #if it is not a negative word
                    curr_emb = embeddings[lemma] #get embedding of current word
                    #find cosine similarity of current word and current neg word
                    sim = cosine_similarity(emb, curr_emb) 
                    #append similarity to list and add to dict
                    similarities[sim] = lemma
                    top_words.append(sim)
            
            #sort cosine similarities in descending order (most similar first)
            top_words.sort(reverse=True)
            #get dict of top_assoc most highly associated words and their corresponding 
            #cosine similarities
            curr, count = 0, 0
            while count < top_assoc and curr < len(top_words):
                if not math.isnan(top_words[curr]):
                    associated_words[similarities[top_words[curr]]] = top_words[curr]
                    count += 1
                curr += 1
            #add to overall dict
            all_assoc_words[neg_word] = associated_words
        
    return all_assoc_words
