#file containing functions for plotting graphs

#import packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from wordcloud import WordCloud
import pandas as pd
#import functions 
from my_modules.sentiment_analysis import sort_negative, get_negative_count


#FUNCTION: get_frequencies_for_plotting
#RETURNS: given a dictionary of scores, find the frequency of occurences of each word
#       with the given sentiment score and return lists of the words in order of decreasing
#       frequency, and then a list of their corresponding frequencies
def get_frequencies_for_plotting(scores, lim):
    #get list of all negative words with relevant score
    neg = scores[-1]
    if lim == "All":
        neg += scores[-0.5]
    #get dict of negative words by count
    neg_count = get_negative_count(neg)
    #sort negative words in order of decreasing frequency
    sorted_neg = sort_negative(neg_count)
    #assign x and y coordinates as negative lemma and its frequency, respectively
    x, y = [], []
    for word_freq in sorted_neg:
        x.append(word_freq[0])
        y.append(word_freq[1])
    return x, y

#FUNCTION: plot_negative_freq_line
#RETURNS: N.A. This function plots negative words in order of decreasing frequency, as line graph
def plot_negative_freq_line(scores, filename, lim):
    x, y = get_frequencies_for_plotting(scores, lim) #get frequencies
    #plot negative lemmata against frequency
    plt.figure(figsize=(20, 10))
    if len(x) > 115: #if too many negative words, only plot first 115 for legibility
        x = x[:115]
        y = y[:115]
    X = np.arange(len(x))
    Y = np.array(y)
    plt.xticks(ticks=X, labels=x, rotation=90, fontsize=7)
    plt.plot(X, Y)
    plt.xlabel("Words with negative sentiment")
    plt.ylabel("Word Frequency")
    plt.suptitle(lim + " Negative Word Frequency in " + filename)
    plt.savefig("Graphs/negation/" + lim + "negwords_" + filename + ".jpg")
    plt.show()

#FUNCTION: plot_negative_freq_bar
#RETURNS: N.A. This function plots negative words in order of decreasing frequency, as bar graph
#       with POS identification
def plot_negative_freq_bar(scores, filename, lim, lex):
    x, y = get_frequencies_for_plotting(scores, lim)
    #differentiate adj from noun
    colors = []
    for lemma in x:
        if lemma.startswith("non "):
            lemma = lemma[4:]
        if lex[lemma][0] == "noun":
            colors.append("tab:blue")
        elif lex[lemma][0] == "adj":
            colors.append("tab:orange")
    
    custom_lines = [Line2D([0], [0], color="tab:blue", lw=4),
                Line2D([0], [0], color="tab:orange", lw=4)]
            
    #uncomment two lines if want to print only top 10 neg words
    # x = x[:10]
    # y = y[:10]
    if len(x) > 115: #if too many negative words, only plot first 115 for legibility
        x = x[:115]
        y = y[:115]
    plt.figure(figsize=(20, 10))
    X = np.arange(len(x))
    Y = np.array(y)
    plt.bar(X, Y, color=colors)
    #uncomment line if want to print only top 10 neg words
    #plt.xticks(ticks=X, labels=x, fontsize=12)
    plt.xticks(ticks=X, labels=x, rotation=90, fontsize=7)
    plt.legend(handles=custom_lines, labels=["Noun", "Adjective"], loc="upper right")
    plt.xlabel("Words with negative sentiment")
    plt.ylabel("Word Frequency")
    plt.suptitle(lim + " Negative Word Frequency in " + filename + "\nWith POS Identification")
    plt.savefig("Graphs/top10neg/POS" + lim + "negwords_" + filename + ".jpg")
    plt.show()
    
    
#FUNCTION: plot_overall_sentiment
#RETURNS: N.A. This function plots total number of negative words, positive words, and 
#       neutral words in given speech as bar graph
def plot_overall_sentiment(scores, filename): 
    X = [-1, -0.5, 0, 0.5, 1]
    X = np.array(X)
    Y = [len(scores[-1]), len(scores[-0.5]), len(scores[0]), len(scores[0.5]), len(scores[1])]
    Y = np.array(Y)

    plt.figure(figsize=(10, 10))
    plt.plot(X, Y)
    plt.xlabel("Sentiment value, <0 negative, 0 neutral, >0 positive")
    plt.ylabel("Frequency")
    plt.suptitle("Total sentiment of all words in " + filename)
    plt.savefig("Graphs/negation/totalsentiment_" + filename + ".jpg")
    plt.show()

#FUNCTION: plot_sentiment_proportion
#RETURNS: N.A. This fcuntion plots total number of extreme negative words, mild negative words,
#       neutral words, mild positive words, extreme positive words, as line graph 
def plot_sentiment_proportion(scores, filename):
    x = ["Negative", "Neutral", "Positive"]
    X = np.arange(len(x))
    Y = [len(scores[-1]) + len(scores[-0.5]), len(scores[0]), len(scores[0.5]) + len(scores[1])]
    Y = np.array(Y)

    plt.figure(figsize=(10, 10))
    plt.bar(X, Y)
    plt.xticks(ticks=X, labels=x,)
    plt.xlabel("Sentiment")
    plt.ylabel("Frequency")
    plt.suptitle("Proportion of sentiments of all words in " + filename)
    plt.savefig("Graphs/negation/negproportion_" + filename + ".jpg")
    plt.show()

#FUNCTION: plot_lexical_diversities
#RETURNS: N.A. This fcuntion plots both the overall and negative lexical diversity for each speech
#       against speeches listed in chronological order as line graph
def plot_lexical_diversities(lexical_diversities):
    #list of titles in chronological order
    chron_titles = ["ProQuinctio", "ProRoscioAmerino", "ProRoscioComodeo", "InVerrem", "InCaecilium", "ProFonteio", "ProCaecina",
        "DeImperioCnPompei", "ProCluentio", "InCatilinam", "ProMurena", "DeLegeAgrarioContraRullum", "ProRabirioPerduellionis",
        "ProArchia", "ProSulla", "ProFlacco", "ProBalbo", "InVatinium", "ProSestio", "ProCaelio", "InPisonem", "ProScauro", 
        "ProPlancio", "ProRabirioPostumo", "ProMilone", "ProLigario", "ProMarcello", "ProRegeDeiotaro", "Philippicae", "All Speeches"]
    
    overall_divs, negative_divs = [], []
    for title in chron_titles:
        overall_divs.append(lexical_diversities[title][0])
        negative_divs.append(lexical_diversities[title][1])

    plt.figure(figsize=(20, 10))
    X = np.arange(len(chron_titles))
    plt.xticks(ticks=X, labels=chron_titles, rotation=45, fontsize=7)
    plt.plot(X, np.array(overall_divs), label="Overall Lexical Diversities", color="tab:blue")
    plt.plot(X, np.array(negative_divs), label="Negative Lexical Diversities", color="tab:orange")
    plt.xlabel("Speeches in chronological order")
    plt.suptitle("Lexical Diversities of All Speeches")
    plt.legend()
    plt.savefig("Graphs/negation/lex_divs.jpg")
    plt.show()

#FUNCTION: plot_assoc_words
#RETURNS: N.A. This fcuntion plots the top_k most strongly associated words for each negative
#       word in given dictionary for given speech, as a line graph and a wordcloud
def plot_assoc_words(assoc_words, speech, top_k):
    for neg_word in assoc_words:
        #print wordcloud
        if len(assoc_words[neg_word]) > 0:
            wordcloud = WordCloud(max_font_size=50, background_color="white").generate_from_frequencies(assoc_words[neg_word])
            plt.figure()
            plt.title("Top " + str(len(assoc_words[neg_word])) + " Words Most Strongly Associated with \'" + neg_word + "\' in " + speech)
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.savefig("Graphs/wordclouds/" + neg_word + "_in_" + speech + ".jpg")
            plt.show()
        #print line graph
        plt.figure(figsize=(15, 10))
        assoc_words_sim = assoc_words[neg_word]
        sim_assoc_words = {}
        sims = []
        for word in assoc_words_sim:
            sims.append(assoc_words_sim[word])
            sim_assoc_words[assoc_words_sim[word]] = word
        sims.sort(reverse=True)
        x,y = [], []
        for sim in sims:
            x.append(sim_assoc_words[sim])
            y.append(sim)
        X = np.arange(len(x))
        Y = np.array(y)
        plt.xticks(ticks=X, labels=x, rotation=45, fontsize=7)
        plt.plot(X, Y)
        plt.xlabel("Associated Words")
        plt.ylabel("Cosine Similarity")
        plt.suptitle("Top " + str(len(assoc_words[neg_word])) + " Words Most Strongly Associated with \'" + neg_word + "\' in " + speech)
        plt.savefig("Graphs/assoc_words/" + neg_word + "_in_" + speech + ".jpg")
        plt.show()
