#File contains functions for loading, opening, printing texts

#Import packages
import os
import re
from cltk.text import lat

#FUNCTION: prep_text
#RETURNS: prep latin text by setting all to lowercase, replacing all v's with u's and all 
#       j's with i's, return updated text
def prep_text(text):
    #replace j with i and v with u
    text = lat.replace_jv(text.lower())
    #strip perentheses and numbers //DO this with regex
    text = re.sub("[\d\[\]@#$%^&*\(\)_~<>-]", "", text)
    #strip white spaces that are not spaces between words
    text = re.sub("[\t\n\r\f\v]", "", text)
    return text

#FUNCTION: read_texts
#RETURNS: open and read files of Cicero's legal speeches, then return dictionary of all speeches
def read_texts():
    all_texts = {}
    #loop through all files and open and read them, add each file to dictionary, format: 'filename': text(as string)
    for filename in os.listdir("Source/cicero_legal_texts"):
        #if filename == "ProBalbo.txt":
        if filename.endswith(".txt"):
            with open("/Users/Brennan/cltk_data/lat/text/lat_text_latin_library/cicero/" + filename, "r") as f:
                full_txt = f.read()
            full_txt = prep_text(full_txt)
            all_texts[filename] = full_txt
    return all_texts

#FUNCTION: get_tokens
#RETURNS: given list of lemma, strip of punctuation and return non-stop words
def get_tokens(lemmata, words):
    return [lemma for word, lemma in zip(words, lemmata) if lemma != "" and word.pos != "punctuation" and word.stop == False]

#FUNCTION: get_lemmata
#RETURNS: given list of words, strip of punctuation and return
def get_lemmata(words):
    return [re.sub("[\.:;,!?\'\"]", "", word.lemma) for word in words]


