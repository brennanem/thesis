#File containing functions to be used in sentiment analysis

#import packages
import pandas as pd
#import functions
from my_modules.text_prep import get_tokens

#list of negative particles to identify negation of positive word
negative_particles = ["non", "neque", "nec", "haude", "minime", "ne", "neue",
                                                    "neu", "nedum", "quin", "nego"]

#list of stop words as identified by CLTK
stops = ["ab","ac","ad","adhic","aliqui","aliquis","an","ante","apud","at","atque","aut","autem","cum","cur","de",
    "deinde","dum","ego","enim","ergo","es","est","et","etiam","etsi","ex","fio","haud","hic","iam","idem","igitur",
    "ille","in","infra","inter","interim","ipse","is","ita","magis","modo","mox","nam","ne","nec","necque","neque",
    "nisi","non","nos","o","ob","per","possum","post","pro","quae","quam","quare","qui","quia","quicumque","quidem",
    "quilibet","quis","quisnam","quisquam","quisque","quisquis","quo","quoniam","sed","si","sic","sive","sub","sui",
    "sum","super","suus","tam","tamen","trans","tu","tum","ubi","uel","uero","unus","ut"]

#FUNCTION: get_sentiment_lex
#RETURNS: download and returnLatinAffectusv2 sentiment lexicon as dictionary
def get_sentiment_lex():
    df = pd.read_csv("Source/sentiment_lexicon/LatinAffectusv2.tsv", sep="\t")
    lex = {}
    for i in range(len(df)):
        lex[df.loc[i,"lemma"]] = [df.loc[i,"pos"], df.loc[i,"polarity_score"]]
    return lex

#FUNCTION: lemma_in_lexicon
#RETURNS: determine if given lemma is in sentiment lexicon by seeing if given lemma
#       matches lemma in lexicon AND given lemma's POS matches lemma's POS in lexicon,
#       return True if mathces, False if not
def lemma_in_lexicon(lemma, pos, lex):
    if lemma in lex:
        if str(pos) == "adjective" and lex[lemma][0] == "adj":
            return True
        if str(pos) == "noun" and lex[lemma][0] == "noun":
            return True
    return False

#FUNCTION: negated_verb
#RETURNS: determine if word at given index i has been negated by checking if it's governing
#       verb has been negated, return True if negated, False if not
def negated_verb(i, words, doc):
    sentence = doc.sentences[words[i].index_sentence]
    governor = words[i].governor
    negated = 1
    if governor != None:
        #if governor of verb is negative
        if sentence[governor].string in negative_particles:
            negated *= -1
        #if preceeding word of governor of verb is negative
        if governor != 0 and sentence[governor-1].string in negative_particles:
            negated *= -1
        #specific case for non modo
        if governor > 1 and sentence[governor-1].string == "modo" and sentence[governor-2].string == "non":
            negated *= -1
    return negated < 0

#FUNCTION: check_negation
#RETURNS: determine if word at given index i has been negated, return True if negated, False if not
def check_negation(score, i, words, doc):
    if i > 1 and words[i-1].string == "modo" and words[i-2].string == "non":
        score *= -1
    #check if there is negation directly before word
    if i != 0 and words[i-1].string in negative_particles:
        score *= -1
    #check if governing verb is negated
    if (negated_verb(i, words, doc)):
        score *= -1
    return score

#FUNCTION: analyze_sentiment
#RETURNS: find negative words dictated by lim (all neg words 0, or extreme neg words -0.5) 
#       from list of lemmata using sentiment lexicon
def analyze_sentiment(lexicon, lemmata, words, doc):
    tokens = get_tokens(lemmata, words)
    scores = {-1: [], -0.5: [], 0: [], 0.5: [], 1: []}
    for i, lemma in enumerate(lemmata):
        if lemma in tokens and lemma_in_lexicon(lemma, words[i].pos, lexicon):
            info = lexicon[lemma]
            score = info[1]
            score_before = score
            if score != 0:
                score = check_negation(score, i, words, doc)
            #if negated, add 'non' before word
            if score_before != score: 
                lemma = "non " + lemma
            #edge case, do not count non modus
            if lemma != "non modus" and words[i].string != "modo":
                scores[score].append(lemma)
    return scores

#FUNCTION: sort_negative
#RETURNS: from dictionary of negative words, make list of negative words according 
#       to frequency, in descending order
def sort_negative(neg):
    sorted_neg = sorted(neg.items(), key=lambda x: x[1], reverse=True)
    return sorted_neg

#FUNCTION: get_negative_count
#RETURNS: get number of occurences of each negative word
def get_negative_count(negative_words):
    neg_count = {}
    for word in negative_words:
        if word in neg_count:
            neg_count[word] += 1
        else:
            neg_count[word] = 1
    return neg_count


#FUNCTION: get_all_negative
#RETURNS: get lexical diversity given list of lemmata of text
def get_all_negative(neg_count):
    all_neg = []
    for neg in neg_count:
        for i in range(neg_count[neg]):
            all_neg.append(neg)
    return all_neg

#FUNCTION: get_overall_sentiment
#RETURNS: get overall sentiment, and count of positive, negative, and neutral words in given text
def get_overall_sentiment(lemmata, lexicon, words, doc):
    tokens = get_tokens(lemmata, words)
    total_neg = {-0.5: 0, -1: 0}
    total_pos = {0.5: 0, 1: 0}
    total_neut = {0: 0}
    overall = 0
    #get frequency count of positive, negative, and neutral words
    for i, lemma in enumerate(lemmata):
        if lemma in tokens and lemma_in_lexicon(lemma, words[i].pos, lexicon):
            info = lexicon[lemma]
            score = info[1]
            overall = overall + score
            if score == 0:
                total_neut[score] += 1
            else: 
                score = check_negation(score, i, words, doc)
                if score < 0:
                    total_neg[score] += 1
                elif score > 0:
                    total_pos[score] += 1
    return total_neg, total_pos, total_neut, overall

#FUNCTION: get_lexical_diversities
#RETURNS: get overall lexical diversit, negative lexical diversity, and proportion of 
#       negative words in text
def get_lexical_diversities(scores, lemmata):
    all_neg = scores[-1] + scores[-0.5]
    neg_div = len(set(all_neg)) / len(all_neg)
    lemmata = [lemma for lemma in lemmata if lemma != "" and lemma not in stops]
    all_div = len(set(lemmata)) / len(lemmata)
    proportion = len(all_neg)/(len(all_neg) + len(scores[0]) + len(scores[1]) + len(scores[0.5]))
    return [all_div, neg_div, proportion]
    


