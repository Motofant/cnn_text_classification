import sys 
import string
import numpy as np
import time
import spacy
from spacy.lang.de.stop_words import STOP_WORDS
import csv
import os.path # check file existence
import math
import pandas as pd


# Ideas, testing and other notes and stuff, TODO: delete before release
#np.set_printoptions(threshold=sys.maxsize) # just for tests

# CMD-args -> input check -> *cut topic and text* -> cut words -> preproc -> lexikon ->  -> one hot oder so -> ??? -> profit

# output design: 
# header: position category, size cat, size lex || body: text (v2w format, final transformation in last step) 


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
cleanup = string.punctuation.replace(',-.','-')+'“'

# Variables
#input_file = ""
#output_file = ""
train = False
#tf_idf = DEF_IDF
#t_use = False
#t_start = 0
#t_end = 0
lex_size = 0
kat_size = 0

# Load necessary stuff
nlp = spacy.load('de')


# 
def failInput():
    print(USE_INFO)

def pathExists(path):
    if not os.path.isfile(path):
        return 'w+'
    else: 
        return 'r+'

def getDictionaryLength(path):
    return len(pd.read_table(path, header=None))

def fillText(num_arr, text_l):
    # shortens/extends ordinal encoded text
    # num_arr: text in indexform, wird auf größe aufgepummt
    # text_l: größe die der Text am ende haben soll
    out_arr = num_arr[:text_l]+[0] * (text_l - len(num_arr))
    '''
    if text_l > len(num_arr):
        out_arr=np.hstack((num_arr,np.zeros(text_l-len(num_arr))))
    elif text_l < len(num_arr):
        out_arr = num_arr[:text_l]
    else: 
        return np.asarray(num_arr)
    '''
    
    return out_arr

def fillTextRepeat(num_arr, text_l):
    i = len(num_arr)
    j = (text_l % i)
    output = num_arr * (int(text_l / i)) +num_arr[:j]
    return output

# used for categorizing as well

def dictionary_old(path,word_arr):
    word_dict=[]
    num_arr=[]

    ## check if file exists
    with open(path, mode=pathExists(path), newline='') as dictFile:

        ## check if empty, read if not
        if os.path.getsize(path)>2:
            print("not empty")
            reader= csv.reader(dictFile, delimiter='\t', quotechar=',', quoting=csv.QUOTE_MINIMAL)
            try:
                for row in reader:
                    #print(row)
                    
                    word_dict.append(row[0])
            except:
                pass   
            print("Completed reading")
    ## write info in word_dict
    for word in word_arr:
        if not word in word_dict: 
            word_dict.append(word)
        num_arr.append(word_dict.index(word))
    
    ## update lexsize
    global lex_size 
    lex_size = len(word_dict)

    ## override dictionary
        ## truncate not final
        ## charset NOT UTF-16 (ASCII works for both)
    with open(path, mode=pathExists(path), newline='',errors='ignore') as dictFile:
        dictFile.truncate()
        writer = csv.writer(dictFile)
        for word in word_dict:
            writer.writerow([word])  
    return num_arr

def dictionary_bef_pd(path, tot_text, training, text_length):
    # returns fixedsized numberarray
    word_dict=[[],[]]
    num_arr=[]
    
    # Read dictionary
    ## check if file exists
    with open(path, mode=pathExists(path), newline='') as dictFile:
        
        ## check if empty, read if not
        if os.path.getsize(path)>2:
            #print("not empty")
            reader= csv.reader(dictFile, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            try:
                for row in reader:                    
                    word_dict[0].append(row[0])
                    word_dict[1].append(row[1])
            except:
                print("invalid dictionary-line")
                print(row)
                #pass   
            #print("Completed reading")

    for word_arr in tot_text:
        
        encoded_text = []
    ## write info in word_dict
        if training:
            for word in word_arr:
                if not word in word_dict[0]: 
                    word_dict[0].append(word)
                    word_dict[1].append(0)          # get's counted later       
                encoded_text.append(word_dict[0].index(word))

        else: 
            # new word in Testingdata
            for word in word_arr:
                if not word in word_dict[0]: 
                    # 1 = doesn't exist in dictionary
                    encoded_text.append(1)
                    continue
                encoded_text.append(word_dict[0].index(word))

        # lexsize has to be saved regardless wether training data or not    
        global lex_size 
        lex_size = len(word_dict[0])

        ## calculating doc freq
        i = 0
        while i < len(word_dict[0]):
            if i in encoded_text:
                word_dict[1][i] = int(word_dict[1][i])+1                
            i +=1

        num_arr.append(encoded_text)

        # update TEXT_CT
        global TEXT_CT
        TEXT_CT += 1 

    ## override dictionary
        ## truncate not final
        ## charset NOT UTF-16 (ASCII works for both)
    with open(path, mode=pathExists(path), newline='',errors='ignore') as dictFile:
        dictFile.truncate()
        writer = csv.writer(dictFile,delimiter = '\t')
        i = int(0)
        for word in word_dict[0]:
            #w = word#.encode('utf-8',errors='ignore')
            writer.writerow([word,word_dict[1][i]])
            i += 1 
 
    '''
    # bring Text to standart size
    num_arr_fixsize = fillText(num_arr,text_length)

    return num_arr_fixsize
    '''
    # fixed size throws problems with tfidf
    return num_arr

def dictionary(path, tot_text, training, text_length):

    # returns fixedsized numberarray
    word_dict=[[],[]]
    num_arr=[]
    
    # Read dictionary
    ## check if file exists and if its empty 
    
    if os.path.exists(path):
        if os.path.getsize(path)> 2:
            word_dict[0] = pd.read_table(path,usecols=[0], header = None).stack().tolist() # engine 
            word_dict[1] = pd.read_table(path,usecols=[1], header = None).stack().tolist()
    else:
        print("File doesn't exist")
        exit()

    for word_arr in tot_text:
        
        encoded_text = []
    ## write info in word_dict
        if training:
            for word in word_arr:
                if not word in word_dict[0]: 
                    word_dict[0].append(word)
                    word_dict[1].append(0)          # get's counted later       
                encoded_text.append(word_dict[0].index(word))

        else: 
            # new word in Testingdata
            for word in word_arr:
                if not word in word_dict[0]: 
                    # 1 = doesn't exist in dictionary
                    encoded_text.append(1)
                    continue
                encoded_text.append(word_dict[0].index(word))

        ## calculating doc freq
        i = 0
        while i < len(word_dict[0]):
            if i in encoded_text:
                word_dict[1][i] = int(word_dict[1][i])+1                
            i +=1

        num_arr.append(encoded_text)

        # update TEXT_CT
        global TEXT_CT
        TEXT_CT += 1 

    # lexsize has to be saved regardless wether training data or not    
    global lex_size 
    lex_size = len(word_dict[0])
    pd.DataFrame([word_dict[0],word_dict[1]]).T.to_csv(path, sep= "\t", header = None, index = False)
    #print(pd.DataFrame(word_dict))
    
    ## override dictionary
        ## truncate not final
        ## charset NOT UTF-16 (ASCII works for both)

    '''
    # bring Text to standart size
    num_arr_fixsize = fillText(num_arr,text_length)

    return num_arr_fixsize
    '''
    # fixed size throws problems with tfidf
    return num_arr

def cutWord(text,modus): 
    
    # text: newsarticle 
    # modus: preprocessing-typ: 
        # 0: default
        # 1: wordtyp
        # 2: grammer
        # 3: tfidf -> save output of dictionary
    # remove punctuation 
    for letter in cleanup:
        text = text.replace(letter, '')
    text = text.replace('.\\','. ')

    doc = nlp(text)
    # cut up text in words
    # remove stopwords to improve speed
    # TODO: different when custom TF IDF
    filtered_text = []

    for el in doc:
        if nlp.vocab[el.text].is_stop == False:
            filtered_text.append(el)

    if modus == 1:
        return wordTyp(filtered_text)
    elif modus == 2:
        return grammer(filtered_text)
    else: 
        y = []
        for token in filtered_text:
            if token.pos_ != "PUNCT" and token.pos_ != "SYM" and token.lemma_ != '\n':
                x = token.lemma_.replace('. . . ',' ')

                # remove capitalized beginning
                y.append(x.lower())
        
        return y, len(y)

# TODO: change l_size to TEXT_SIZE
def bagOfWords(num_arr, l_size):
    # num_arr: Text transformed with ordinal encoding
    # l_size: Wörterbuchlänge, sollte lex_size betragen
    # no output size needed because length is allways constant
    word_to_vec = np.zeros(l_size)
    i=0
    while i < len(num_arr):
        word_to_vec[int(num_arr[i])] += 1
        i += 1
    return word_to_vec


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
    while i < o_size:
        w_vec = np.zeros(l_size)
        
        # expletive should be first in lexikon 
        w_vec[0] = 1
        word_to_vec = np.vstack((word_to_vec, w_vec))
        i += 1

    #print(np.shape(word_to_vec))

    # returns array of arrays like: ([0,1,0,0],[1,0,0,0],...)
    # see: https://numpy.org/doc/stable/reference/generated/numpy.vstack.html
    return word_to_vec


# TODO: only takes in single word, more efficient when list gets transformed in one go -> problems when single word
    # DONE: see topic
    # UPDATE: texts are currently filtered before running this programm
def topic_old(category, path):
    # category: string, transformed into number
    # path: path, where dictionary with categories lies

    word_dict= []
    # Idea for different output, might be an option later
    # out_arr=[]
    ## check if file exists
    with open(path, mode=pathExists(path), newline='', encoding='utf8') as dictFile:

        ## check if empty, read if not (empty csv-file -> 2 bytes)
        if os.path.getsize(path)>2:
            #print("not empty")
            reader= csv.reader(dictFile, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for row in reader:
                #print("2")
                word_dict.append(row[0])
            #print("Completed reading") 

    if not category in word_dict: 
        word_dict.append(category)
    global kat_size
    kat_size = len(word_dict)
    #print(len(word_dict))
    ## override dictionary
        ## truncate not final
        ## charset NOT UTF-16 (ASCII works for both)
    with open(path, mode=pathExists(path), newline='') as dictFile:
        dictFile.truncate()
        writer = csv.writer(dictFile)
        for word in word_dict:
            writer.writerow([word]) 

    # create vector
    # out_arr = [0]*cat_lex_size
    # out_arr[word_dict.index(category)] = 1

    return word_dict.index(category)

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
def tfIdf(input_arr, path, bound, doc_count):
    # input_arr: array out of dictionary
    reader_size = 0
    j = len(input_arr)
    doc_freq = np.empty(0)
    
    # output: array of arrays, arr1: idf, arr2: tf, arr3: tf idf -> will be returned
    output = np.zeros([3,j])

    # inverse document frequency
    # read doc freq
    with open(path, mode="r+", newline='') as dictFile:
        reader = csv.reader(dictFile, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            doc_freq = np.append(doc_freq,int(row[1]))
        reader_size = len(doc_freq)

    term_freq = np.zeros(reader_size)
    i = 0
    for number in input_arr:
        output[0][i] = math.log(doc_count/doc_freq[number])
        # counting term frequency
        term_freq[number] += 1
        i += 1 

    # term frequency
    i = 0
    for number in input_arr:
        output[1][i] = term_freq[number]/j
        i += 1
    
    # tf-idf
    transf_arr = np.empty(0)
    i = 0
    for number in input_arr:
        output[2][i] = output[0][i] * output[1][i]
        if output[2][i] > bound:
            transf_arr = np.append(transf_arr,input_arr[i])
        i += 1  
    #breakpoint()
    # if deleting of values is outsourced 
    #return np.vstack((input_arr,output[2]))
    return transf_arr.astype(int)


def wordTyp(text):
    output = []
    for token in text: 
        if token.pos_ == "NOUN" or token.pos_ == "VERB":
            output.append(token.lemma_.lower())

    return output, len(output)

def grammer(text):
    output = []
    for token in text: 
        if (token.pos_ == "NOUN" or token.pos_ == "VERB") and (token.dep_ == "sb" or token.dep_ == "pd" or token.dep_ == "ROOT"):
                output.append(token.lemma_.lower())
    return output, len(output)


"""
# analyse cmdlineargs
for arg in sys.argv:
    if arg == "-i":
        input_file = sys.argv[sys.argv.index(arg)+1]
        #print(input_file)
        continue

    if arg == "-o":
        output_file = sys.argv[sys.argv.index(arg)+1]
        continue

    if arg == "-sw":
        try:
            tf_idf = int(sys.argv[sys.argv.index(arg)+1])
        except ValueError:
            tf_idf = DEF_IDF
            print("no tf cutoff defined")
        continue

    if arg == "-t":   
        t_use = True
        continue

    #if arg == "-train":
        # TODO: remember to take in category
    # TODO: bag of words



# check validity of Input
try:
    f = open(input_file)
except FileNotFoundError:
    print("Inputfile not found.")
    failInput()
    exit()
else:
    f.close()
    print("Inputfile accepted")

# check validity of output
try:
    f = open(output_file)
except FileNotFoundError:
    # if no output defined, use modififed input file
    output_file = input_file.replace(".txt", "_proc.txt")
    print("Outputfile not found. Defaultsetting used. Name: "+ output_file)
else:
    f.close()
    print("Outputfile accepted")

# check tf value
# TODO: define cutoff value for term frequency 
if tf_idf > HIGH_BOUND or tf_idf < LOW_BOUND:
    tf_idf = DEF_IDF
    print("Bad tf value, default is used.\nNeeds to be between " + str(LOW_BOUND) + " and " + str(HIGH_BOUND) + ".")

t_start = time.time()
# TODO: al the stuff








t_end = time.time()
print(t_end-t_start)
"""


# TODO: zwischenspeichern von num_arr aus wordcut für idf


#######
## 
#######
'''
out=[]
cat = []
#with open('C:\\Users\\Erik\\Documents\\Uni\\BA\\material\\test.csv', mode ='r',newline='\n',encoding='utf8') as ofile:
#with open('C:\\Users\\Erik\\Desktop\\test.csv', mode ='r',newline='\n') as ofile:
with open('D:\\Datenbank\\test_100k.csv', mode ='r',newline='\n', encoding= 'utf8') as ofile:
    oreader = csv.reader(ofile, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for row in oreader:
        if (len(row) == 2):
            #print(row[1])
            out.append(row[0])
            cat.append(row[1])
    #print(out)
print ("Finished reding file")
#######
## Update dictionaries
#######

print("start cutting")
pos = 0
txt_num = []
cat_num = []

# extra to keep size constant 
for x in out:
    txt_num.append(dictionary('C:\\Users\\Erik\\Desktop\\in.csv',cutWord(x)))
    cat_num.append(topic(cat[pos],'C:\\Users\\Erik\\Desktop\\kat.csv'))
    pos = pos + 1 


print("finished cutting\n start defining output")
pos = 0

with open('C:\\Users\\Erik\\Desktop\\littleout.csv', mode=pathExists('C:\\Users\\Erik\\Desktop\\littleout.csv'), newline='') as outFile:
#with open('C:\\Users\\Erik\\Desktop\\new.csv', mode=pathExists('C:\\Users\\Erik\\Desktop\\new.csv'), newline='') as outFile:
    for x in txt_num:

    
    #    y = dictionary('C:\\Users\\Erik\\Desktop\\in.csv',cutWord(x))
    #    category = topic(cat[pos],'C:\\Users\\Erik\\Desktop\\kat.csv')

        #create output
        output = []

        #
        output.append(4) #headersize
        output.append(kat_size) # number of categories
        output.append(cat_num[pos]) # index of category
        output.append(lex_size) # number of words in dictionary
        #print(output)
        # test header-transformation
        #print(headerTransform([1,5,3]))

        output.extend(txt_num[pos])

        #for word in txt_num[pos]:
        #    output.append(word)
        # check output
        #print(output)
        pos += 1
#        outFile.truncate()
        writer = csv.writer(outFile)
        writer.writerow(output)
# only if on hot used
#        for word in y:
#            writer.writerow(word)
        #writer.writerows([y])
        #print(y[0])


#print(topic(cat[0],'C:\\Users\\Erik\\Desktop\\kat.csv'))

#print(lex_size)

###y = dictionary('C:\\Users\\Erik\\Desktop\\in.csv',cutWord(out))
#print(bagOfWords(y))
#print(y)
#print(oneHot(y))


'''

newarr = []
with open('C:\\Users\\Erik\\Desktop\\new.csv', mode=pathExists('C:\\Users\\Erik\\Desktop\\new.csv'), newline='') as outFile:
    
    reader= csv.reader(outFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for row in reader:
        inarr = []
#        for num in row:
#        print(num)
        i = 4
        for i in range(4,len(row)-1):
            inarr.append(int(row[i]))
            i += 1
        newarr.append(inarr)


#for row in newarr
i = 0
#for row in newarr:
#    with open('C:\\Users\\Erik\\Desktop\\new__'+str(i)+'.csv', mode='w+', newline='') as outFile:
#        writer = csv.writer(outFile)
#        writer.writerow(oneHot(newarr[0], 200)) #573252))
#        y = oneHot(row, 300)
#        for x in y:
            
#            pass
#            writer.writerow(x)
#        i += 1
#np.savetxt('C:\\Users\\Erik\\Desktop\\np.csv', oneHot(newarr[0], 573), delimiter=',', fmt='%d')
#print(bagOfWords(newarr[0], 573))

