#import keras
import numpy as np
import keras
from keras.models import Sequential
#import keras.models
from keras.layers import MaxPooling1D, Dense, Dropout, Flatten, Conv2D, Conv1D, MaxPool1D,GlobalMaxPooling1D
import pandas as pd 
import keras.optimizers as ko
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import logging # TODO
from texttable import Texttable
import pre_proc as p 
import pipeline as pl
from data_gen import DataGenerator
import os
#import sklearn.model_selection as sms


tf.compat.v1.disable_eager_execution() # to prevent tf-bug with inputdata (anugrahasinha, https://github.com/tensorflow/tensorflow/issues/38503)
#np.set_printoptions(threshold=sys.maxsize)


#### Variables ####
#region
# Pipelinevariables
## already in pipeline-> not needed when porting

## new variables
data_from_file = True
load_nn = True

classes = ["Politik", "Kultur", "Gesellschaft", "Leben", "Sport", "Reisen", "Wirtschaft", "Technik", "Wissenschaft"]

# necessary for current version

model_save = "./model/tw2v_10.h5"
weight_save = "./weight/weight.h5"

con_name = "nw2v" 
## variables to include in config 
network_id = 2
batch_training = False
no_epoch = 10
batch_size = 50
# input
training = True
load_nn = not training
#input_file = "C:/Users/Erik/Documents/Uni/BA/Repo/cnn_text_classification/output/tb/test_3_1_0.csv"
fp = "C:/Users/Erik/Documents/Uni/BA/Repo/cnn_text_classification/output/tb/"
if training:
    if batch_training:
        input_files = [f for f in os.listdir(fp) if "train" in f]
    else:
        input_file = "C:/Users/Erik/Documents/Uni/BA/Repo/cnn_text_classification/output/ww2v/oute.csv"        
    #print(input_files)
    #exit()
else:
    input_file = "C:/Users/Erik/Documents/Uni/BA/Repo/cnn_text_classification/output/ww2v/out.csv"
    pass
    #exit()


# inputparams
Text_length = 1200 #also bowlength
word_vec_length =  9# only not 1 in oneHot

cat_size = 9

# data
input_categories = np.array([]) # only when trainingsdata
input_text = np.array([])

#endregion

#### Functions ####
#region

def netParams(encoding, cat_len, text_len, dict_loc):
    ## defines network params 

    # encoding: int, basis to determain used variables
    # cat_len: int, n.o. categorsies
    # text_len: int, n.o. words in text
    # dict_loc: string, path to dictionary

    if encoding == 0:
        return text_len, 1
    elif encoding == 1:
        return p.getDictionaryLength(dict_loc), 1
    elif encoding == 3:
        return text_len, cat_len

def usedNetwork(network_index, input_shape):
    ## defines network used

    # network_index: int, basis to determain network
    # input_shape: shape of datainput to design inputlayer  

    if network_index == 0:
        return newNetwork_ord(input_shape)
    elif network_index == 1:
        return newNetwork_bow(input_shape)
    elif network_index == 2:
        return newNetwork_w2v(input_shape)
    elif network_index == 3:
        return newNetwork_bow_for_w2v(input_shape)
    elif network_index == 4:
        return newNetwork_w2v_for_bow(input_shape)

def setCats(path):
    return pd.read_csv(path,header = None)[0].to_numpy()

def showResult(prediction, classes):
    ## presentation of classification of one text

    # prediction: array of floats, network output
    # classes: list of strings, rowname in table

    table = Texttable()
    table.set_cols_dtype(["t","f"])
    table.add_row(["Class", "Percantage\nin %"])
    i = 0
    for element in classes:
        table.add_row([element, prediction[i]*100])
        i += 1 

    return table 

def readFile(input_file, train, text_l, word_l,cat_size):
    ## reads file and returns corretly transformed inputdata
    ## not used for datagenerators

    # input_file: string, path to inputfile
    # train: boolean, non training data does not return categories per text 
    # text_l: int, parameter to transform input correctly
    # word_l: int, parameter to transform input correctly
    # cat_size: int, parameter to transform input correctly

    if train:
        in_cat = pd.read_table(input_file,usecols=[0],header = None).to_numpy()
        in_text = pd.read_table(input_file,usecols=list(range((text_l*word_l)+1))[1:],header = None).to_numpy()
        print(len(in_text[0]))
    else:
        in_cat = []
        in_text = pd.read_table(input_file,header = None).to_numpy()
    
    return  in_text.reshape((in_text.shape[0],text_l,word_l)),keras.utils.to_categorical(in_cat,cat_size)

def visualHist(history):
    ## shows visualization of accuracy and loss of network after training is completed 
    
    # history: network parameters progress over time 
    
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])

    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['dense_accuracy'], loc='lower right')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['dense_loss'], loc='upper right')
    plt.show()

## neural networks
def newNetwork_bow_for_w2v(in_shape):

    model = Sequential()

    model.add(Conv1D(128,20,input_shape=in_shape,activation="relu",padding="same"))
    model.add(MaxPooling1D(3,data_format="channels_first"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Flatten())
    model.add(Dense(9,activation="softmax"))

    optimizer = keras.optimizers.Adam(lr=0.00001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'],experimental_run_tf_function=False)

    print("compiled succesfully")
    
    return model

def newNetwork_bow(in_shape):

    model = Sequential()

    model.add(Conv1D(128,20,input_shape=in_shape,activation="sigmoid",padding="same"))
    model.add(MaxPooling1D(3,data_format="channels_first"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Flatten())
    model.add(Dense(9,activation="softmax"))

    optimizer = keras.optimizers.Adam(lr=0.00001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'],experimental_run_tf_function=False)
 
    print("compiled succesfully")
    
    return model

def newNetwork_w2v_for_bow(in_shape):
    ## create network (save)
    # https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
    # https://keras.io/examples/nlp/pretrained_word_embeddings/


    model = Sequential()

    model.add(Conv1D(64,36,dilation_rate=9,input_shape=in_shape,activation="sigmoid",padding="valid"))
    model.add(MaxPooling1D(2,data_format="channels_first"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Flatten())
    model.add(Dense(9, activation="softmax"))
    
    optimizer = keras.optimizers.Adam(lr=0.00001)
    model.compile( loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'],experimental_run_tf_function=False)

    print("compiled succesfully")

    return model

def newNetwork_w2v(in_shape):
    ## create network (save)
    # https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
    # https://keras.io/examples/nlp/pretrained_word_embeddings/

    model = Sequential()

    model.add(Conv1D(64,36,dilation_rate=9,input_shape=in_shape,activation="relu",padding="valid"))
    model.add(MaxPooling1D(2,data_format="channels_first"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Flatten())
    model.add(Dense(9, activation="softmax"))
    
    optimizer = keras.optimizers.Adam(lr=0.0001)
    model.compile( loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'],experimental_run_tf_function=False)

    print("compiled succesfully")

    return model

def newNetwork_ord(in_shape):
    ## create network (save)
    # https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
    # https://keras.io/examples/nlp/pretrained_word_embeddings/


    model = Sequential()

    model.add(Conv1D(64,4,input_shape=in_shape,activation="relu",padding="valid"))
    model.add(MaxPooling1D(2,data_format="channels_first"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Flatten())
    model.add(Dense(9, activation="softmax"))
    
    optimizer = keras.optimizers.Adam(lr=0.0001)
    model.compile( loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'],experimental_run_tf_function=False)

    print("compiled succesfully")

    return model
#endregion

#### "Pipeline" ####
if __name__ == "__main__":
    # define vectors
    
    input_text = []
    valid_class = []
    # define variables
        # get config
    config = pl.loadConfig(con_name)

    classes = p.getCategories(config[7])
    Text_length,word_vec_length = netParams(int(config[2]),len(classes),int(config[3]),config[4])
    nn_input_size = (Text_length,word_vec_length)
    network_id = int(config[8])
    batch_training = config[9]
    batch_size = int(config[10])
    no_epoch = int(config[11])
    
    '''
    print(str(batch_training))
    print(str(batch_size))
    print(str(no_epoch))
    print(str(nn_input_size))
    
    breakpoint()
    '''

    # TODO: check whether single file or datagenerator 
    if data_from_file:
        input_text, valid_class = readFile(input_file, training, Text_length, word_vec_length,cat_size)
        
        pass
    else:
        ## use Vector
        pass 
        #print(len(input_text[0]))

    # create CNN
    model = usedNetwork(network_id, nn_input_size)

    ## if already used -> use weights
    if load_nn:# or True:
        model.load_weights(weight_save)

        ## show model
    model.summary()



    # do stuff
    if training:
        #train_gen = DataGenerator()
        #model.fit_generator(generator= train_gen)
        
        ## training -> save weights in the end -> non result needed
            ### TODO: change epoches/batchsize ? 
        if batch_training:
            trainings_train_gen = DataGenerator(input_files, fp,training, Text_length, word_vec_length, len(classes),1, batch_size,1)
            print("train data gen done")
            history = model.fit_generator(generator= trainings_train_gen, epochs =no_epoch)      
        else:
            history = model.fit(x = input_text,y =valid_class,shuffle = True,epochs=no_epoch, batch_size=batch_size)

            ### update weights
        model.save_weights(weight_save)

        visualHist(history)
    else:
        ## test -> no save requiered (no weight updates) -> Show result

        predictions = model.predict(input_text)
            #print(valid_class[)
        print(predictions)
        # TODO: check if multiple texts as input
        j = 1
        print(type(predictions[0][0]))
        

        out = []
        for row in predictions:
            row = list(row)
            i = max(row)
            out.append(row.index(i))
        #np.savetxt("./result.csv",out,delimiter=",")
        '''import csv
        with open("./result.txt", mode="w+", newline='') as dictFile:
            writer=csv.writer(dictFile)
            for row in predictions:
                i = []
                for el in row:
                    i.append(float(el))
                writer.writerow(i)'''
        
        pd.DataFrame(out).to_csv("./result.csv", header = None, index = False)
        '''for i in predictions:
            print(j)
            print(showResult(i, classes).draw())
            j += 1
            ### TODO: Show results
'''