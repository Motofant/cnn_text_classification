import pre_proc as p
import configtry as c
import numpy as np
import csv
import sys
#np.set_printoptions(threshold=sys.maxsize) # just for tests
# TODO: variable configlocation
#### variables
# inputtyp
training = False
config_load = False
# number of all texts
text_count = 0
# max words in text
word_max = 50 # used for one hot -> fill word 
bar = 0
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
topic_dic_file = 'C:/Users/Erik/Desktop/kat.csv'
topic_file = 'C:/Users/Erik/Desktop/save/topic/'
input_file = 'C:/Users/Erik/Desktop/check.csv'
dict_file_dir = './dictionary/'
save_file_dir = './save/'
out_file_dir = './output/'

# TODO: insert cmd args

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


    for text in text_in:
        preproc_out.append(p.dictionary(dictio,p.cutWord(text,prep_mode), train, text_len))
    
    return preproc_out

def texAnaTfIdf(text_in,dictio,border):
    # TODO: Test if working for single text
    preproc_out = []  

    for text in text_in:
        preproc_out.append(list(p.tfIdf(text,dictio, border, 500)))

    return preproc_out

def encodingTyp(arr_in, code, dict_len):
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
            y.append(text)
        if code == 2:
            coding_out = np.array([p.oneHot(y,dict_len,word_max)])
            for text in arr_in[1:]:
                coding_out = np.vstack((coding_out, [p.oneHot(text,dict_len,word_max)]))# wordmax hier nicht notwendig, da txt auf länge gebracht 
        else:
        # no modification to encoding ->  ordinal encoding
            return arr_in
    
    return coding_out

def category(cat_in, topic_file):
    y =[]
    for cat in cat_in:
        
        y.append(p.topic(cat,topic_file))
    return y

# Saving and loading
def saveCat(path, data, prep, code):
    file_name = path + "topic_"+str(prep)+"_"+str(code)+".csv"
    with open(file_name, mode='w+', newline='', encoding= 'utf8') as dictFile:
        writer = csv.writer(dictFile, delimiter = "\t")
        writer.writerow(data)

def loadCat(path, prep, code):
    file_name = path + "topic_"+str(prep)+"_"+str(code)+".csv"
    out = []
    with open(file_name, mode=p.pathExists(file_name), newline='', encoding= 'utf8') as dictFile:
        reader = csv.reader(dictFile, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            for element in row:
                out.append(element)
    return out

def saveData(path, data, prep, code):
    # Used when data modified via tf-idf is includeed in multiple sittings 
    # define savefile
    file_name = path + "text_"+str(prep)+"_"+str(code)+".csv"
    with open(file_name, mode='a', newline='', encoding= 'utf8') as dictFile:
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

def loadData(path, prep, code):
    file_name = path + "text_"+ "_"+str(prep)+"_"+str(code)+".csv"
    output = []
    with open(file_name, mode=p.pathExists(file_name), newline='', encoding= 'utf8') as dictFile:
        reader = csv.reader(dictFile, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            i = 0
            for nr in row:
                row[i] = int(nr)
                i += 1
            output.append(row)
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

# check validity of File and config
if not c.existProf("C:/Users/Erik/Desktop/test.ini",loaded_config):
    print("No valid config, continue with default.")
    loaded_config = "def"
if not os.path.isfile(input_file):
    print("Input file doesn't exist, shutting down.")
    exit()


# load Config, if not defiend use default

config_input = loadConfig(loaded_config)

# read input file 
text, cat = readFile(input_file, training)

# define category if trainingsdata
if training:
    cat = category(cat, topic_dic_file)
    cat = loadCat(topic_file, preproc,coding)+ cat
    if not final_set:
        saveCat(topic_file, cat,  preproc,coding)

# call function wordcut + preprocessing
analysed_text = textAna(text,preproc,dic_file,training, word_max)


if final_set:
    if preproc == 3:
        analysed_text = texAnaTfIdf(analysed_text,dic_file,bar)
    
    analysed_text = loadData(save_file_dir, preproc, coding) + analysed_text
    final_output = encodingTyp(analysed_text, coding, dictLen(dic_file))
    saveData(out_file_dir,final_output, preproc, coding)
else:
    saveData(save_file_dir,list(analysed_text), preproc, coding)

# ending, saves necessary data for next launch
# updating values
config_input[0] = text_count
saveShutdown(loaded_config,config_input)


## Testing
