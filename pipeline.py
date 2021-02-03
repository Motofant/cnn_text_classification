import pre_proc as p
import configtry as c
import categorisation_cnn as cc
import numpy as np
import keras
import csv
import sys
from os import path,listdir
#np.set_printoptions(threshold=sys.maxsize) # just for tests
# TODO: variable configlocation

#### variables
#region
# inputtyp
training = False
config_load = False
just_encode = True

# networkvar TODO: add to ini
text_vec_l = 1200
word_vec_l = 1 
load_nn = True
class_number = 9
# number of all texts
text_count = 0
# max words in text
word_max = 1200 # used for one hot -> fill word 
bar = .50
#word in dictionary -> only for testingpurposes
#dict_size = 0
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
coding = 1
final_set  = True
loaded_config = "def"

# files

dic_file = 'C:/Users/Erik/Desktop/dictionary.csv'
topic_dic_file = './dictionary/kat.csv'
topic_file = './save/'
input_file = 'C:/Users/Erik/Desktop/check.csv'
dict_file_dir = './dictionary/'
save_file_dir = './save/'
out_file_dir = './output/'

# NNfiles
weight_save = "./save/weight/weight.h5"

# file name (used for creating saves and outputs)
file_name = ""

#endregion

# create new stuff
def newProfil(config, name, data):
    # check wether profil  allready exits
    if c.existProf(config,name):
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
    with open(path, mode=p.pathExists(path), newline='') as dictFile:
        reader = csv.reader(dictFile, delimiter='\t', quotechar=',', quoting=csv.QUOTE_MINIMAL)        
        for row in reader:
            dict_len += 1
    return dict_len

def getFilename(input_path):
    return path.splitext(path.basename(input_path))[0]

# actually relevant to Pipeline
def readFile(in_file, train):
    file_in = []
    category = []
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
    return file_in,category#, len(file_in)

def textAna(text_in, prep_mode, dictio, train, text_len):
    preproc_out = []

    if prep_mode == 1 or prep_mode == 2:
        for text in text_in:
            txt, num = p.cutWord(text,prep_mode)
            preproc_out.append(p.dictionary(dictio,txt, train, text_len))
            if num > text_len: 
                    global word_max
                    word_max = num
    else:
        for text in text_in:
            txt, num = p.cutWord(text,prep_mode)
            preproc_out.append(p.dictionary(dictio,txt, train, text_len))
        
            
    return preproc_out

def texAnaTfIdf(text_in,dictio,border):
    # TODO: Test if working for single text
    preproc_out = []  

    for text in text_in:
        preproc_out.append(list(p.tfIdf(text,dictio, border, 500)))

    return preproc_out

def encodingTyp(arr_in, code, dict_len, text_l):
    ## Bag of words
    # static size -> fillword not needed
    if code == 1:
        # NOT DRY (see https://en.wikipedia.org/wiki/Don%27t_repeat_yourself)

        coding_out = np.array([p.bagOfWords(arr_in[0],dict_len)])
        for text in arr_in[1:]:
            coding_out = np.vstack((coding_out,p.bagOfWords(text, dict_len)))
    else:
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
        # no modification to encoding ->  ordinal encoding
            return y
    
    return coding_out.tolist()

def category(cat_in, topic_file):
    y =[]
    for cat in cat_in:
        
        y.append(p.topic(cat,topic_file))
    return y

# Saving and loading
def saveCat(path, data, prep, code, name):
    file_name = path + "topic_"+str(prep)+"_"+str(code)+".csv"
    
    # including represented file
    representing = "text_"+str(prep)+"_"+str(code)+"_"+str(name)
    data.insert(0,representing)
    
    with open(file_name, mode='a', newline='', encoding= 'utf8') as dictFile:
        writer = csv.writer(dictFile, delimiter = "\t")
        writer.writerow(data)

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

def saveData(path, data, prep, code, name):
    # Used when data modified via tf-idf is includeed in multiple sittings 
    # define savefile
    file_name = path + "text_"+str(prep)+"_"+str(code)+"_"+str(name)+".csv"
    with open(file_name, mode='a+', newline='', encoding= 'utf8') as dictFile:
        writer = csv.writer(dictFile, delimiter = "\t")
        for line in data:
            writer.writerow(line)

def saveShutdown(config_name, saved_states):
    c.saveProf(config_name,saved_states[0],saved_states[1],saved_states[2],saved_states[3],saved_states[4],saved_states[5],saved_states[6])

def loadValues(path):
    file_name = path + "start_config.txt"
    
    with open(file_name, 'r') as savefile:
        saved_values = list(map(int,savefile.readlines()))
    return saved_values

def loadData(path, prep, code, name ):
    # get all saved files with same preprocessing and encoding
    file_name = path+"text_"+str(prep)+"_"+str(code)+"_"+str(name)+".csv"
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
    except:
        print("Error occured. Unexpected design of loaded config.")
    
    return x


#####################
### Pipeline
#####################

# Commandline
#region
# shortest input: pipeline.py inputfile (default config, final input, testingset)
# longest input: pipeline.py config -t -n inputfile (Input not final, trainingset)
length =len(sys.argv)
# check validity of input
if length < 2 or length > 7:
    print("invalid length")
    exit()


for arg in sys.argv:
    # extra functions
    if arg == "-dict":
        try:
            newDictionary(sys.argv[2],sys.argv[3])
            print("Dictionary created succesfully")
        except:
            print("wrong input")
        exit()



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
        input_file = arg
        continue
    else:
        loaded_config = arg

#endregion

# check validity of File and config
if not c.existProf("test.ini",loaded_config):
    print("No valid config, continue with default.")
    loaded_config = "def"
if not path.isfile(input_file):
    print("Input file doesn't exist, shutting down.")
    exit()

# get Filename to create output
file_name = getFilename(input_file)

# load Config, if not defiend use default

config_input = loadConfig(loaded_config)

# read input file 
text, cat = readFile(input_file, training)

# define category if trainingsdata
if training:
    cat = category(cat, topic_dic_file)
# deleted cause right now save is needed
#    if not final_set:
    saveCat(topic_file, cat,  preproc,coding, file_name)

# call function wordcut + preprocessing
analysed_text = textAna(text,preproc,dic_file,training, word_max)


# TODO: change that in final set no save needed 
saveData(save_file_dir,list(analysed_text), preproc, coding, file_name)

if final_set: 
    final_output = []
    # adding categories
    #check how many files will be transformed
    # TODO: maybe count in config
    # load categoryfile
    cats = loadCat(topic_file,preproc,coding)
    # define file_parameter
    file_parameter = "text_"+str(preproc)+"_"+str(coding)

    for f in listdir(save_file_dir):
        g = getFilename(f)

        if file_parameter in g:
            # load textfile
            texts = loadData(save_file_dir,preproc,coding,g.replace("text_"+str(preproc)+"_"+str(coding)+"_",""))
            if preproc == 3:
                texts = texAnaTfIdf(analysed_text,dic_file,bar)
            f_o= encodingTyp(texts, coding, dictLen(dic_file),word_max)
            # add category, TODO: can be better
            for row in cats:
                if row[0] in g:
                    
                    i = 0
                    for element in row[1:]:                      
                        f_o[i].insert(0, element)
                        i += 1

            final_output += f_o
    if just_encode:
        saveData(out_file_dir,final_output, preproc, coding, "")
        config_input[0] = text_count
        saveShutdown(loaded_config,config_input)
        exit()
                
    #start neural network
    model = cc.newNetwork((text_vec_l,word_vec_l))
    if load_nn:
        model.load_weights(weight_save)

    model.summary()

    # adjust input
    in_text = np.array(final_output).reshape((int(final_output.shape[0]/text_vec_l),text_vec_l,word_vec_l)) 
    
    if training:
        valid_class = keras.utils.to_categorical(cats,class_number)
        history =model.fit(x = in_text,y =valid_class,shuffle = True,epochs=10, batch_size=10)
        model.save_weights(weight_save)

        #accuracy = model.evaluate(in_text,valid_class)
        #print(accuracy)
        
        cc.visualHist(history)    
    else:
        prediction = model.predict(in_text)    

        # TODO: change
        for i in range(5):
            print(valid_class[i])
            print(prediction[i])
            i += 1

# ending, saves necessary data for next launch
# updating values
config_input[3] = word_max
config_input[0] = text_count
saveShutdown(loaded_config,config_input)


## starting neural Network -> no save needed, watch for correct inputtype
