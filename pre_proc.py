import sys 
import string
import numpy as np
import time
import spacy
from spacy.lang.de.stop_words import STOP_WORDS
import csv
#import os.path # check file existence
from os import path,listdir
import os
import math
import pandas as pd
import logging
from collections import Counter
# print version of libs


# Ideas, testing and other notes and stuff, TODO: delete before release
#np.set_printoptions(threshold=sys.maxsize) # just for tests

# CMD-args -> input check -> *cut topic and text* -> cut words -> preproc -> lexikon ->  -> one hot oder so -> ??? -> profit

# output design: 
# header: position category, size cat, size lex || body: text (v2w format, final transformation in last step) 

# Logger
logger = logging.getLogger(__name__)

# Errormessage
USE_INFO = """USE: pre_proc.py [define Input] [option] [option] ...
-i 'direktory\\filename.txt'    -> Input needs to be defined
-o 'direktory\\filename.txt'    -> Define Output
-train                          -> input is training data (has def. category)
-sw 'threshold'                 -> use stopwords, threshold optional(10 default) 
-t                              -> show time needed"""

# default termfrequency treshold + boundaries
#DEF_IDF = 10
#LOW_BOUND = 0
#HIGH_BOUND = 100
#TEXT_SIZE = 500 #number of words, text can't be longer than that
TEXT_CT = 20 # number of texts in input
cleanup = "'#$%&*+=/<>{|@}[]^_-~"  #string.punctuation.replace('!"#$%&'()*+, -./:;<=>?','-')+'“'

# Variables
train = False

lex_size = 0
kat_size = 0

# Load necessary stuff
nlp = spacy.load('de')
#nlp = spacy.load('de_core_news_sm')
 
def failInput():
    print(USE_INFO)

def pathExists(path):
    if not os.path.isfile(path):
        return 'w+'
    else: 
        return 'r+'

def getAllFiles(directory):
    all_files = [(directory + f) for f in listdir(directory) if (f.endswith(".csv") or f.endswith(".txt"))]
    logger.debug("files found: "+str(len(all_files)))
    return all_files

def getDictionaryLength(path):
    return len(pd.read_table(path, header=None))

def fillText(num_arr_total, text_l):
    # shortens/extends ordinal encoded text
    # num_arr: text in indexform, wird auf größe aufgepummt
    # text_l: größe die der Text am ende haben soll
    output = []
    for num_arr in num_arr_total:

        output.append(num_arr[:text_l]+[0] * (text_l - len(num_arr)))
    '''
    if text_l > len(num_arr):
        out_arr=np.hstack((num_arr,np.zeros(text_l-len(num_arr))))
    elif text_l < len(num_arr):
        out_arr = num_arr[:text_l]
    else: 
        return np.asarray(num_arr)
    '''
    
    return output

def fillTextRepeat(num_arr_total, text_l):
    output = []
    try:
        for num_arr in num_arr_total:      
            i = len(num_arr)
            j = (text_l % i)
            output.append(num_arr * (int(text_l / i)) +num_arr[:j])
    except ZeroDivisionError:
        logger.info(ZeroDivisionError)
    
    return output

# used for categorizing as well

def buildDictCat(texts, cats_of_texts, cat_len, diction,dict_file, train):
    iterator = 0
    out_texts = []
    text_trans= {x:i for i,x in enumerate(diction)}
    if train:
        for text in texts:
            temp_text = []
            for word in text:
                if diction.get(word, None) == None:
                    # create new element
                    diction[word] = [0]*cat_len
                    text_trans[word] = len(text_trans)
                    
                # update 
                diction[word][cats_of_texts[iterator]] += 1
                temp_text.append(text_trans.get(word))

            out_texts.append(temp_text)
            iterator += 1
        saveDictCat(diction, dict_file)
    else:
        for text in texts:
            temp_text = []
            for word in text:
                temp_text.append(text_trans.get(word,1))

            out_texts.append(temp_text)

    return {i:x for i,x in enumerate(diction.values())}, out_texts

def encodeDictCat(texts, diction, cat_len):
    encoded_text = []
    for text in texts:
        single_text = []
        for word in text:

            dict_word = diction.get(word, None)
            if dict_word == None:
                # add neutral element
                dict_word = [0]*cat_len

            single_text.append(dict_word)
        encoded_text.append(single_text)
    
    return encoded_text

def saveOutFile(texts,classes, dic_file):
    with open(dic_file, mode="w",newline='',encoding="utf8") as dictFile:
        writer = csv.writer(dictFile, delimiter = "\t")
        iterator = 0
        for cat in classes:
            
            writer.writerow([cat]+[item for sublist in texts[iterator] for item in sublist])
            iterator += 1

    return True

def loadOutFile(dic_file,len_class):
    classes = []
    texts = []
    with open(dic_file, mode="r",newline='',encoding="utf8") as dictFile:
        reader = csv.reader(dictFile,delimiter= "\t")
        for row in reader:
            classes.append(row[0])
            #texts.append([int(el) for el in row[1:]])
            texts.append([list(map(int,row[idx : idx+len_class])) for idx in range(1,len(row),len_class)])
    return texts , classes

def saveDictCat(diction, dic_file):
    with open(dic_file, mode="w",newline='',encoding="utf8") as dictFile:
        writer = csv.writer(dictFile, delimiter = "\t")
        for key in diction.keys():
            writer.writerow([key]+[item for sublist in [diction[key]] for item in sublist])
    return True

def saveOutFile_test(tot_text, out_file):
    with open(out_file, mode="w",newline='',encoding="utf8") as dictFile:
        writer = csv.writer(dictFile, delimiter = "\t")
        for text in tot_text:
            writer.writerow([item for sublist in text for item in sublist])
    return True

def loadDictCat(dic_file,cat_size):
    with open(dic_file, mode='r',encoding= "utf8") as inp:
        reader = csv.reader(inp,delimiter= "\t")
        dict_from_csv = {rows[0]:list(map(int, rows[1:cat_size+1])) for rows in reader}   
    return dict_from_csv

def encodeDictCat_test(texts, diction, cat_len):
    encoded_text = []
    for text in texts:
        single_text = []
        for word in text:
            dict_word = diction.get(word, None)
            if dict_word == None:
                # add neutral element
                dict_word = [0]*cat_len

            single_text.append(dict_word)
        encoded_text.append(single_text)
    
    return encoded_text


def encodeDictCat_old(texts, diction, cat_len):
    encoded_text = []
    for text in texts:
        single_text = []
        for word in text:
            dict_word = diction.get(word, None)
            if dict_word == None:
                # add neutral element
                dict_word = [0]*cat_len

            single_text.append(dict_word)
        encoded_text.append(single_text)
    
    return encoded_text

def buildDictCat_old(texts, cats_of_texts, cat_len, diction):
    iterator = 0
    for text in texts:
        for word in text:
            if diction.get(word, None) == None:
                # create new element
                diction[word] = [0]*cat_len
            
            # update 
            diction[word][cats_of_texts[iterator]] += 1
        iterator += 1
    return diction

def dictionary(path, tot_text, training, text_length):
    # returns fixedsized numberarray
    word_dict=[[],[]]
    num_arr=[]
    
    # Read dictionary
    ## check if file exists and if its empty 
    
    if os.path.exists(path):
        if os.path.getsize(path)> 2:
            word_dict[0] = pd.read_table(path,usecols=[0], encoding="utf8", header = None).stack().tolist() # engine 
            word_dict[1] = pd.read_table(path,usecols=[1], encoding="utf8", header = None).stack().tolist()
    else:
        print("File doesn't exist")
        exit()
    
    # convert to dictionary
    dictionary = { word_dict[0][i]:i for i in range(0, len(word_dict[0]) ) }
    dictionary_length = len(dictionary)

    for word_arr in tot_text:
        
        encoded_text = []
    ## write info in word_dict
        if training:
            for word in word_arr:
                number = dictionary.get(word, None)
                if number == None:
                        number = dictionary_length
                        dictionary[word] = number
                        dictionary_length += 1
                        word_dict[1].append(0)
                encoded_text.append(number)

        else: 
            # new word in Testingdata
            for word in word_arr:
                number = dictionary.get(word, 1) # if wort doesnt exist in testdata write 1 
                encoded_text.append(number)

        ## calculating doc freq
        for word in set(encoded_text):
            word_dict[1][word] += 1    

        num_arr.append(encoded_text)

        # update TEXT_CT
        global TEXT_CT
        TEXT_CT += 1 

    # lexsize has to be saved regardless wether training data or not    
    global lex_size 
    lex_size = dictionary_length

    ## override dictionary
    pd.DataFrame([list(dictionary.keys()),word_dict[1]]).T.to_csv(path, sep= "\t", header = None, index = False)
    #print(pd.DataFrame(word_dict))
    
    '''
    # bring Text to standart size
    num_arr_fixsize = fillText(num_arr,text_length)

    return num_arr_fixsize
    '''

    # fixed size throws problems with tfidf
    return num_arr

def cutWord(total_text,mode):
    # text: newsarticle 
    # modus: preprocessing-typ: 
        # 0: default
        # 1: wordtyp
        # 2: grammer
        # 3: tfidf -> save output of dictionary
    
    total_tokens = []
    total_output = []
    total_text_len = []

    for text in total_text:

        # remove unnecessary punctuation
        for letter in cleanup:
            text = text.replace(letter, '')
        text = text.replace('.\\','. ')

        doc = nlp(text)

        filtered_text = []

        for token in doc:
            if not nlp.vocab[token.text].is_stop:
                filtered_text.append(token)
        '''
        for word in doc:
            if word.lemma_ not in STOP_WORDS:
                filtered_text.append(word)
        '''
        total_tokens.append(filtered_text)
    
    if mode == 1:
        return wordTyp(total_tokens)

    elif mode == 2:
        return grammar(total_tokens)

    else:
        # both default and TF IDF need entire textbody
        logger.info("pre_proc started")
        for text in total_tokens:
            y = []
            for token in text:
                if token.pos_ != "PUNCT" and token.pos != "NUM" and token.pos_ != "SYM" and token.lemma_ != '\n' and token.lemma_ != ' ':
                    y.append(token.lemma_.lower())
            total_output.append(y)
            total_text_len.append(len(y))
        logger.info("pre_proc finished")
    
    return total_output, total_text_len

def bagOfWords(text_lists,l_size):
    # create Output list
    bow_lists = []
    for text in text_lists:
        counted_word = Counter({x:0 for x in range(l_size)})
        counted_word.update(Counter(text))
        x = np.asarray(list(dict(sorted(counted_word.items())).values()))
        x = x/len(text)
        bow_lists.append(list(x))
    
    return bow_lists

def bagOfWords_old(text_lists,l_size):
    # create Output list
    bow_lists = []
    for text in text_lists:
        counted_word = Counter({x:0 for x in range(l_size)})
        counted_word.update(Counter(text))

        bow_lists.append(list(dict(sorted(counted_word.items())).values()))
    
    return bow_lists


# transforms ordinal encoded text to one-hot-encoded text, used as input for NN 
def oneHot(num_arr, l_size, o_size):
    # num_arr: Text transformed with ordinal encoding
    # l_size: Wörterbuchlänge, sollte lex_size betragen
    # o_size: length  of output, default should be TEXT_SIZE

    # create first entry of output, can't use vstack on empty Matrix cause of dimensions
    word_to_vec = np.zeros(l_size)
    word_to_vec[int(num_arr[0])] = 1
    
    # encode Text 
    i = 1
    while i < len(num_arr):
        w_vec = np.zeros(l_size)
        w_vec[int(num_arr[i])] = 1
        word_to_vec = np.vstack((word_to_vec, w_vec))
        i += 1

    # fill rest with expletive
    '''
    while i < o_size:
        w_vec = np.zeros(l_size)
        
        # expletive should be first in lexikon 
        w_vec[0] = 1
        word_to_vec = np.vstack((word_to_vec, w_vec))
        i += 1
'''
    #print(np.shape(word_to_vec))

    # returns array of arrays like: ([0,1,0,0],[1,0,0,0],...)
    # see: https://numpy.org/doc/stable/reference/generated/numpy.vstack.html
    return word_to_vec

def topic(categories, path):
    # category: string, transformed into number
    # path: path, where dictionary with categories lies

    # check if categories are in list form. 
    # Error could accure when there is one Text -> give string 
    if type(categories) is not list:
        categories = [categories]

    word_dict= []
    # Idea for different output, might be an option later
    out_arr=[]
    ## check if file exists
    with open(path, mode=pathExists(path), newline='', encoding='utf8') as dictFile:

        ## check if empty, read if not (empty csv-file -> 2 bytes)
        #if os.path.getsize(path)>2:
        try:
            word_dict = pd.read_table(dictFile,header=None).stack().tolist() 
        except pd.errors.EmptyDataError:
            print("No categories in file.")

    for category in categories:
        if not category in word_dict: 
            word_dict.append(category)
        out_arr.append(word_dict.index(category))
    global kat_size
    kat_size = len(word_dict)
    
    ## override dictionary
        ## charset NOT UTF-16 (ASCII works for both)
    with open(path, mode=pathExists(path), newline='',encoding= "utf8") as dictFile:
        pd.DataFrame(word_dict).to_csv(dictFile, sep= "\t", header = None, index = False)

    # create vector
    # out_arr = [0]*cat_lex_size
    # out_arr[word_dict.index(category)] = 1

    #return word_dict.index(category)
    return out_arr

# only put in header
def headerTransform(header):
    # create Topic-vector
    # header design: headerlength, size of topicvector, topicID, size of dictionary
    i = np.zeros(header[1])
    i[header[2]] = 1
    return i


### function which change the Inputsize
# Termfrequency-inversedocument frequency

def tfIdf(input_lists, dict_path, bound, doc_count):
    output_lists = []
    # read doc frequency
    doc_freq = pd.read_table(dict_path,usecols=[1], engine="python", encoding="utf8", header = None).stack().tolist()
    #reader_size = len(doc_freq)
    for input_arr in input_lists:
        j = len(input_arr)
        counted_words = Counter(input_arr)
        
        i = 0       
        # output: array of arrays, arr1: idf, arr2: tf, arr3: tf idf -> will be returned
        calc_arr = np.zeros([3,j]) 
        transf_arr = []
        # calculate TF-IDF

        for number in input_arr:
            
            calc_arr[0][i] = math.log(doc_count/doc_freq[int(number)])
            calc_arr[1][i] = counted_words.get(int(number))/j
            calc_arr[2][i] = calc_arr[0][i] * calc_arr[1][i]
            
            if calc_arr[2][i] > bound:
                transf_arr.append(input_arr[i])

            i += 1

        output_lists.append(transf_arr)

    return output_lists 

def wordTyp(total_text):
    output_text = []
    output_len = []

    logger.info("pre_proc started")

    for text in total_text:
        word_typ_preproc= []
        for token in text: 
            if token.pos_ == "NOUN" or token.pos_ == "VERB":
                word_typ_preproc.append(token.lemma_.lower())
        output_text.append(word_typ_preproc)
        output_len.append(len(word_typ_preproc))

    logger.info("pre_proc finished")
    return output_text, output_len

def grammar(total_text):
    output_text = []
    output_len = []

    logger.info("pre_proc started")
    
    for text in total_text:
        grammar_preproc= []
        for token in text: 
            if (token.pos_ == "NOUN" or token.pos_ == "VERB") and (token.dep_ == "sb" or token.dep_ == "pd" or token.dep_ == "ROOT"):
                    grammar_preproc.append(token.lemma_.lower())
        output_text.append(grammar_preproc)
        output_len.append(len(grammar_preproc))

    logger.info("pre_proc finished")
    return output_text, output_len

def loadStopword(path):
    # TODO: check if valid directory 
    logger.info("start adding stopworts")
    files = getAllFiles(path)
    logger.debug(str(len(files))+" stopwordfiles found")

    for document in files:
        read_stopwords = pd.read_table(document, header = None).stack().tolist()
        logger.debug(str(len(read_stopwords))+" stopwords found")
        # convert to lemma
        stop_word_string = " "
        stop_word_string = stop_word_string.join(read_stopwords)
        doc = nlp(stop_word_string)
        for word in doc:
            nlp.Defaults.stop_words.add(word.lemma_)  

    logger.info("Stopwords added")

    return True

def smallerDict(dict_path, treshold):
    out_list = []
    new_dict = {}

    out_list = dict(pd.read_table(dict_path, encoding="utf-8", header = None).itertuples(index=False, name=None))

    iterator = 0 # iterates over every word in dictionary
    counter = 0 # counts every word, that is over the treshold 
    second_list = []
    for el in out_list.keys():
        val = int(out_list.get(el))
        if val > treshold or val < 1:
            second_list.append((el,val,iterator,counter))
            new_dict[counter] = (el,val)
            counter += 1
        else:
            # if word doesn't meet treshold -> not in training
            second_list.append((el,val,iterator,1))
        iterator += 1
    # create new dict_path
    z = path.splitext(path.basename(dict_path))
    output_file = path.dirname(dict_path) + "/" + z[0] + "_small" + z[1]
    with open(output_file,"w",newline='', encoding= "utf-8") as new_dict_file:
        writer = csv.writer(new_dict_file, delimiter = "\t")
        writer.writerows(new_dict.values())

    return_list = {old_index:(word, doc_freq, new_index) for word, doc_freq, old_index ,new_index in second_list}

    return return_list, output_file

def smallerText(texts, new_dict):

    out_texts = []
    for text in texts:
        temp_text_list = []
        for word in text:
            temp_text_list.append(new_dict.get(word)[2])
        out_texts.append(temp_text_list)

    return out_texts