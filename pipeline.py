import pre_proc as p
import configtry as c
import categorisation_cnn as cc
import numpy as np
import keras
import csv
import sys
from os import path,listdir,remove
import logging
import pandas as pd
from re import search
import spacy
import tensorflow as tf
from data_gen import DataGenerator
np.set_printoptions(threshold=sys.maxsize) # just for tests
# TODO: variable configlocation

## Init logging 
logging.basicConfig(filename='./Pipeline_time.log',format= "%(asctime)s :: %(relativeCreated)d ms :: %(levelname)s :: %(module)s.%(funcName)s :: %(message)s", level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.info("------------------------------------------")
logger.info("Starting Pipeline")


#### variables
#region
# inputtyp
training = False
config_load = False
just_encode = True
delete_saves = True
# TODO: change classes from read topics
classes = ["Politik", "Kultur", "Gesellschaft", "Leben", "Sport", "Reisen", "Wirtschaft", "Technik", "Wissenschaft"]
# networkvar TODO: add to ini
text_vec_l = 1200
word_vec_l = 1 
load_nn = False
class_number = 9
dict_size_treshold = 5
# number of all texts
text_count = 0
# max words in text
word_max = 0 # used for fill word 
bar = 0.01
#word in dictionary -> only for testingpurposes
#dict_size = 0
# fix textsize 
    # 0 = longest Text with new fillword
    # 1 = shortest Text
    # 2 = longest Text with repeating text
fix_size_param = 0

# Preprocessing
    # 0 = no preproc
    # 1 = wordtyp
    # 2 = grammar
    # 3 = tf-idf
preproc = 3

# Coding
    # 0 = no coding -> ordinal encoded
    # 1 = bag of words
    # 2 = one Hot
    # 3 = word 2 vec
coding = 1
final_set  = True
loaded_config = "def"

batch_size = 1

# files
stop_word_dir = './stopword/'
dic_file = './dictionary/def_dictionary.csv'
topic_dic_file = './dictionary/kat.csv'
topic_file = './save/'
input_file = 'C:/Users/Erik/Desktop/check.csv'
dict_file_dir = './dictionary/'
save_file_dir = './save/'
out_file_dir = './output/'
input_directory = './input/'

# NNfiles
weight_save = "./weight/weight.h5"

# file name (used for creating saves and outputs)
file_name = ""

#endregion

# create new stuff
def newProfil(config, name, data):
    # check wether profil  allready exits
    if c.existProf(config,name):
        logging.warning("Profilname already used")
        return "Cant create Profil, because it allready exists."
    else:
        # set data to False, if settings not closer defiend
        if data:
            # TODO: try
            c.newProfMauell(name, data[0],data[1],data[2],data[3],data[4],data[5],data[6])
        else:
            c.newProf(name)
    return "Profil created succesfully."

def newDictionary(path, name):
    file_name = path +"/"+name+ "_dictionary.csv"
    with open(file_name, mode='w+', newline='', encoding= 'utf8') as dictFile:
        writer = csv.writer(dictFile, delimiter = "\t")
        writer.writerow(["BLANK",0])
        writer.writerow(["NOT IN TRAINING",0])
    return file_name

# helping Function
def dictLen(path):
    dict_len  = 0 
    with open(path, mode=p.pathExists(path), newline='',encoding= "utf8") as dictFile:
        reader = csv.reader(dictFile, delimiter='\t', quotechar=',', quoting=csv.QUOTE_MINIMAL)        
        for row in reader:
            dict_len += 1
    return dict_len

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

def loadDictCat(dic_file,cat_size):
    with open(dic_file, mode='r',encoding= "utf8") as inp:
        reader = csv.reader(inp,delimiter= "\t")
        dict_from_csv = {rows[0]:list(map(int, rows[1:cat_size+1])) for rows in reader}   
    return dict_from_csv

def catTransform(cat):
    return [el for l in cat for el in l[1:]]

def getFilename(input_path):
    return path.splitext(path.basename(input_path))[0]


def deleteSaves(save_dir, preproc, encoding):
    for i in listdir(save_dir):
        if search('t.*_'+str(preproc)+'_'+str(encoding)+'.*.csv',i):
            remove(save_dir + i)
    return True
    
def resetVar(training):
    
    # reset global variables
    global word_max, text_count
    if not training:
        word_max = 0
    text_count = 0
    return True

def calcUnknownWords():
    pass

# actually relevant for Pipeline
def readFile(in_file, train):
    file_in = []
    category = []
    logger.info("Start reading Inputfile")
    with open(in_file, mode=p.pathExists(in_file), newline='', encoding= 'utf8') as dictFile:
        reader = csv.reader(dictFile, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # TODO: Nicht schön 
        if train:
            for row in reader: 
                try:
                    file_in.append(row[0])
                    category.append(row[1])
                except IndexError:
                    print("File doesn't contain category for all texts. Current text is skipped.")      
 
        else:
            for row in reader:
                file_in.append(row[0])
    # TODO: maybe change to array instead of keeping extra variables
    global text_count
    text_count += len(file_in)
    logger.info("Inputfile reading concluded")
    return file_in,category#, len(file_in)

def textAna(text_in, prep_mode, fix_size,dictio, train, text_len, categories,cat_len):
    logger.info("general textanalysis starting")
    preproc_out = []
    total_text = []
    length_list = []

    # TODO -> read dictionary in before
    # Cut texts up in Lists of words
    total_text,length_list = p.cutWord(text_in,prep_mode)

        #if (prep_mode == 1 or prep_mode == 2) and num > text_len:

    # update word_max in needed
    
    longest_text = max(length_list)
    shortest_text = min(length_list)
    # if text > 1200 exists  -> gets shorted

    global word_max

    if train:
        if fix_size == 1:
            logger.debug("fize_size == 1")
            if shortest_text < 64:
                shortest_text = 64
            word_max = shortest_text
        else:
            logger.debug("fize_size != 1")
            if longest_text > 1200:
                longest_text = 1200             
            word_max = longest_text
        logger.debug("word_max changed to "+ str(word_max))
    # encode text with dictionary

    # TODO check output 
    if coding == 3:
        # TODO: do fill -> encode to ordinal -> change encoding later
        
        dictionary = p.buildDictCat(total_text,categories,cat_len,loadDictCat(dictio, cat_len))
        preproc_out = p.encodeDictCat(total_text,dictionary,cat_len)
        #preproc_out = p.fillText(preproc_out,1200)
        saveDictCat(dictionary, dictio)
        saveOutFile(preproc_out, categories, out_file_dir)
        exit()
    else:
        preproc_out = p.dictionary(dictio,total_text, train, text_len)
    
    logger.info("general textanalysis concluded")
    return preproc_out

def texAnaTfIdf(text_in,dictio,border,text_number):
    # TODO: Test if working for single text

    logger.info("TF IDF started")


    preproc_out = p.tfIdf(text_in,dictio, border, text_number)

    logger.info("TF IDF concluded")
    return preproc_out

def encodingTyp(arr_in, code, fill_param, vec_l, word_l):
    ## Bag of words
    # static size -> fillword not needed
    
    logger.info("encoding started")
    arr_out = []
    if code == 1:
        arr_out = p.bagOfWords(arr_in,vec_l)
        logger.info("encoding concluded")
        return arr_out

    else:
        # fill text to same size, necessary fro input
        if fill_param == 2:
            #arr_out = p.fillTextRepeat(arr_in, text_l)
            arr_out = p.fillTextRepeat(arr_in, vec_l)
            logger.info("encoding concluded")
        else: 
            # no modification to encoding ->  ordinal encoding
            # One Hot only directly before input in NN to avoid big saves
            arr_out = p.fillText(arr_in, vec_l)
            #arr_out = p.fillText(arr_in, text_l)
            logger.info("encoding concluded")
        
        # encode in OneHot if asked
        if code == 2:
            output = []
            
            for arr in arr_out:
                output.append(list(keras.utils.to_categorical(arr, word_l)))
            return output

        return arr_out
        
        '''
        # not Bag of word -> fix size manualy
        y = []
        for text in arr_in:
            y.append(p.fillText(text,text_l))
        if code == 2:
            #breakpoint()
            coding_out = np.array([p.oneHot(y[0],dict_len,text_l)])
            for text in y[1:]:
                coding_out = np.vstack((coding_out, [p.oneHot(text,dict_len,text_l)]))# wordmax hier nicht notwendig, da txt auf länge gebracht 
        else:
        '''
        
    logger.info("encoding concluded")
    #return coding_out.tolist()

# not longer needed
def category(cat_in, topic_file):
    y =[]
    for cat in cat_in:
        
        y.append(p.topic_old(cat,topic_file))
    return y

# Saving and loading
def saveCat(path, data, prep, code, name):
    file_name = path + "topic_"+str(prep)+"_"+str(code)+".csv"
    
    # including represented file
    representing = "train_"+str(prep)+"_"+str(code)+"_"+str(name)
    data.insert(0,representing)
    
    with open(file_name, mode='a', newline='', encoding= 'utf8') as dictFile:
        writer = csv.writer(dictFile, delimiter = "\t")
        writer.writerow(data)
    
    return True

def saveDataSplit(ord_enc_data, cats, train, pre_proc, encoding, directory, batchsize, vec_l, word_l, filler):
    # ord_enc_data: Data without encoding
    # directory: directory to save output files.
    # batchsize: number of texts in one file
    # vec_l: size of dictionary(BOW) or text (other encodings)

    ## split list into multiple lists of batchsize
    split_lists = [ord_enc_data[i:i+batchsize] for i in range(0,len(ord_enc_data), batchsize)]
    split_cats = [cats[i:i+batchsize] for i in range(0,len(cats), batchsize)]
    ## Save splited lists in different files.
    ## Save names to ID List

    file_IDs = []
    
    file_name = ""
    if train:
        file_name = "train_" + str(pre_proc) + "_" + str(encoding) + "_"
        
        iter = 0
        for p_o_list in split_lists:
            # encode part of total texts        
            encoded_list = encodingTyp(p_o_list, encoding, filler, vec_l, word_l)

            # push category to first value
            part_iter = 0
            for single_text in encoded_list:
                single_text.insert(0,split_cats[iter][part_iter])
                part_iter += 1

            # Generate ID
            path = directory + file_name + str(iter) + ".csv"
            file_IDs.append(file_name + str(iter))
            iter += 1

            pd.DataFrame(encoded_list).to_csv(path, sep = "\t", header= None, index=False)        

    
    else: 
        file_name = "test_" + str(pre_proc) + "_" + str(encoding) + "_"
    
        iter = 0
        for p_o_list in split_lists:
            # encode part of total texts        
            encoded_list = encodingTyp(p_o_list, encoding, filler, vec_l, word_l)

            # Generate ID
            path = directory + file_name + str(iter) + ".csv"
            file_IDs.append(file_name + str(iter))
            iter += 1

            pd.DataFrame(encoded_list).to_csv(path, sep = "\t", header= None, index=False)        

    return file_IDs

def loadCat(path, prep, code):
    file_name = path + "topic_"+str(prep)+"_"+str(code)+".csv"
    out = []

    with open(file_name, mode=p.pathExists(file_name), newline='', encoding= 'utf8') as dictFile:
        reader = csv.reader(dictFile, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            transfer =[row[0]]
            for element in row[1:]:
                transfer.append(int(element))
            out.append(transfer)
    return out

def saveData(path, data, train, prep, code, name):
    # Used when data modified via tf-idf is includeed in multiple sittings 
    # define savefile
    if train: 
        file_name = path + "train_"+str(prep)+"_"+str(code)+"_"+str(name)+".csv"
    else: 
        file_name = path + "test_"+str(prep)+"_"+str(code)+"_"+str(name)+".csv"

    with open(file_name, mode='a+', newline='', encoding= 'utf8') as dictFile:
        writer = csv.writer(dictFile, delimiter = "\t")
        for line in data:
            writer.writerow(line)
    return 1

def saveShutdown(config_name, saved_states):
    c.saveProf(config_name,saved_states[0],saved_states[1],saved_states[2],saved_states[3],saved_states[4],saved_states[5],saved_states[6])

def loadValues(path):
    file_name = path + "start_config.txt"
    
    with open(file_name, 'r') as savefile:
        saved_values = list(map(int,savefile.readlines()))
    return saved_values

def loadData(path, train, prep, code, name ):
    # get all saved files with same preprocessing and encoding
    if train:
        file_name = path+"train_"+str(prep)+"_"+str(code)+"_"+str(name)+".csv"
    else: 
        file_name = path+"test_"+str(prep)+"_"+str(code)+"_"+str(name)+".csv"
    output = []
    #breakpoint()

    with open(file_name, mode=p.pathExists(file_name), newline='', encoding= 'utf8') as dictFile:
        reader = csv.reader(dictFile, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            i = 0
            for nr in row:
                row[i] = int(float(nr))
                i += 1
            output.append(row)

    ## output like [[dat1text1],[dat2text1],[dat2text2]...] -> no difference 
    return output

def loadConfig(config_name):
    global text_count,preproc,coding,word_max,dic_file,save_file_dir,out_file_dir
    x = c.getProf(config_name)
    if len(x) != 7:
        print("config doesn't match variables, default config is used")
        x = c.getProf("def")
    try:
        text_count = int(x[0])
        preproc = int(x[1])
        coding = int(x[2])
        word_max = int(x[3])
        dic_file = x[4]
        save_file_dir = x[5]
        out_file_dir = x[6]
    except Exception:
        print("Error occured. Unexpected design of loaded config.")
    
    return x


#####################
### Pipeline
#####################
if __name__ == "__main__":
    '''print(spacy.__version__)
    print(np.__version__)
    print(tf.__version__)'''
    # Commandline
    #region
    # shortest input: pipeline.py inputfile (default config, final input, testingset)
    # longest input: pipeline.py config -t -n inputfile (Input not final, trainingset)
    length =len(sys.argv)
    # check validity of input
    if length < 2 or length > 7:
        print("invalid length")
        exit()

    # TODO change to enumerate?
    skip_iteration = False

    iterator = 0 
    for arg in sys.argv:
        # extra functions
        if skip_iteration:
            skip_iteration = False
            continue

        if arg == "-dict":
            try:
                newDictionary(sys.argv[2],sys.argv[3])
                print("Dictionary created succesfully")
            except Exception:
                print("wrong input")
            exit()
        
        # 
        if arg == "-fill":
            try:
                print(sys.argv[iterator+2])
                fix_size_param = int(sys.argv[iterator+2])

            except ValueError:
                logger.debug("-fill got bad value, continue with 2")
                fix_size_param = 2
            continue

        # Inputfile is not the final input
        if arg == "-n":
            final_set = False
            continue
        # Input is a trainingset
        if arg == "-f":
            training = True
            continue
        # Last Element is inputfile
        if arg== sys.argv[-1]:
            input_directory = arg
            continue
        else:
            loaded_config = arg
        iterator += 1

    #endregion
    ## Config stuff 
    # check validity of File and config
    if not c.existProf("test.ini",loaded_config):
        print("No valid config, continue with default.")
        loaded_config = "def"
    if not path.isdir(input_directory):
        print("Input directory doesn't exist, shutting down.")
        exit()

    # load Config, if not defiend use default
    config_input = loadConfig(loaded_config)
    logger.info("loaded config: " + loaded_config)
    # load stopwords
    p.loadStopword(stop_word_dir)

    # prepare fix size process
    if training:
        if fix_size_param == 1:
            logger.debug("fize size to smallest text")
            if word_max == 0:
                word_max = 1200
                logger.info("changed word_max to 1200")
        elif fix_size_param == 2:
            logger.debug("fize size to biggest text (repeat text)")
            pass
        else: 
            # in case input is random int
            logger.debug("fize size to biggest text (repeat fillword)")
            fix_size_param = 0

    # get all documents in inputdirectory
    input_files = p.getAllFiles(input_directory)
    n_o_files = len(input_files)
    if n_o_files == 0:
        print("No files in directory")
        exit()

    # do things with all documents
    for input_file in input_files:
        
        # get Filename to create output
        file_name = getFilename(input_file)

        # read input file 
        text, cat = readFile(input_file, training)

        # define category if trainingsdata
        if training:
            # Old
            # cat = category(cat, topic_dic_file)
            cat = p.topic(cat,topic_dic_file)

        # deleted cause right now save is needed
        #    if not final_set:
            saveCat(topic_file, cat,  preproc,coding, file_name)

        # call function wordcut + preprocessing
        analysed_text = textAna(text,preproc,fix_size_param,dic_file,training, word_max, cat[1:], len(classes))


        # TODO: change that in final set no save needed 
        saveData(save_file_dir,list(analysed_text), training ,preproc, coding, file_name)

        logger.info(file_name + " finished.")
        print(file_name + " finished")
        del analysed_text
        del text
    # temp exclusion
    if final_set:
        # define dictionary length
        dict_length = p.getDictionaryLength(dic_file)

        # def output list
        final_output = []
        # define params

        vec_l = dict_length if coding == 1 else word_max
        word_l = dict_length if coding == 2 else 1
        number_of_texts = 0
        cats = []
        texts_list = []

        # define file_parameter
        

        if training:
            file_parameter = "train_"+str(preproc)+"_"+str(coding)    
            # adding categories
            #check how many files will be transformed
            # TODO: maybe count in config
            # load categoryfile
            cats = loadCat(topic_file,preproc,coding)


            for f in listdir(save_file_dir):
                g = getFilename(f)
                
                if file_parameter in g:
                    # load textfile
                    texts = loadData(save_file_dir,training, preproc,coding,g.replace("train_"+str(preproc)+"_"+str(coding)+"_",""))
                    if preproc == 3:
                        texts = texAnaTfIdf(texts,dic_file,bar,text_count)
                        #breakpoint()
                    texts_list.extend(texts)
                    '''
                    # encoding will happen later
                    f_o= encodingTyp(texts, coding, fix_size_param, dictLen(dic_file),word_max)
                    
                    # add category, TODO: can be better
                    for row in cats:
                        if row[0] in g:
                            
                            i = 0
                            for element in row[1:]:                      
                                f_o[i].insert(0, element)
                                i += 1
                    
                    final_output += f_o
                    '''
            
            if dict_size_treshold > 0:
                logger.info("treshold != 0, dictionarymod started")
                # decrease dictionary size by deleting
                mod_dict, dic_file = p.smallerDict(dic_file, dict_size_treshold)
                
                # modify pre_processed text
                texts_list = p.smallerText(texts_list, mod_dict)
                logger.info("treshold != 0, dictionarymod ended")
                del mod_dict
        else:
            # testing data
            # get all files with same encoding
            file_parameter = "test_"+str(preproc)+"_"+str(coding)
            for f in listdir(save_file_dir):
                g = getFilename(f)

                if file_parameter in g:
                    # load textfile
                    texts = loadData(save_file_dir, training, preproc,coding,g.replace("test_"+str(preproc)+"_"+str(coding)+"_",""))
                    
                    if preproc == 3:
                        texts = texAnaTfIdf(texts,dic_file,bar,text_count)
                        #breakpoint()
                    texts_list.extend(texts)
                    '''
                    f_o= encodingTyp(texts, coding, fix_size_param, dictLen(dic_file),word_max)

                    final_output += f_o
                    '''
        text_count = len(texts_list)
        # save before next step
        #saveData(out_file_dir,final_output, training, preproc, coding, "")
        transformed_cats = catTransform(cats)
        # 
        file_ID_list = saveDataSplit(texts_list, transformed_cats,training, preproc, coding, out_file_dir, batch_size,vec_l,word_l,fix_size_param)
        logger.info("All preparations (Textanalysis and Preprocessing) concluded.")

        if just_encode:
            # TODO: save file_ID
            config_input[0] = text_count
            config_input[4] = dic_file
            resetVar(training)
            saveShutdown(loaded_config,config_input)
            if delete_saves:
                deleteSaves(save_file_dir,preproc,coding)
            print("done")
            
            exit()

        # prepare data
        '''
        # define n_o_texts
        n_o_texts = len(final_outpu#t)
        '''
        # text_vec_l
        '''        
        if coding == 1:
            text_vec_l = vec_l
        else:
            text_vec_l = vec_l
            if training:
                text_vec_l -= 1'''
        text_vec_l = vec_l
        word_vec_l = word_l

        # word_vec_l 
        '''        
        if coding == 2:
            word_vec_l = dict_length
        else:
            word_vec_l = 1
        '''
        


        # starting neural Network -> no save needed, watch for correct inputtype
        model = cc.newNetwork((text_vec_l,word_vec_l))
        if load_nn:
            model.load_weights(weight_save)

        model.summary()
        #final_output = np.asarray(final_output)
        # adjust input
        # TODO: make better
        cats = []
        #breakpoint()
        '''if training:
            cats = [sublist[0] for sublist in final_output]
            long_list = [item for sublist in final_output for item in sublist[1:]]
        else:
            long_list = [item for sublist in final_output for item in sublist]


        in_text = np.reshape(long_list,(n_o_texts,text_vec_l,word_vec_l)) 
        '''
        if training:
            # TODO: define trainings- and validationdata
            trainings_train_gen = DataGenerator(file_ID_list, out_file_dir,training, text_vec_l, word_vec_l, len(classes),coding, batch_size,1)
            
            #valid_class = keras.utils.to_categorical(cats,class_number)

            logger.info("start training")
            history = model.fit_generator(generator= trainings_train_gen, epochs =20, workers=4)
            #history =model.fit(x = in_text,y =valid_class,shuffle = True,epochs=10, batch_size=10)
            logger.info("training done")
            model.save_weights(weight_save)

            #accuracy = model.evaluate(in_text,valid_class)
            #print(accuracy)
            
            cc.visualHist(history)    
        else:
            logger.info("start predictions")
            prediction = model.predict(in_text)    
            logger.info("predictions done")
            categories = cc.setCats(topic_dic_file)
            # TODO: change
            for i in prediction:
                print(cc.showResult(i,classes).draw())
                i += 1
        resetVar(training)
        if delete_saves:
            deleteSaves(save_file_dir,preproc,coding)


    # ending, saves necessary data for next launch
    # updating values
    config_input[3] = word_max
    config_input[0] = text_count
    config_input[4] = dic_file
    saveShutdown(loaded_config,config_input)
