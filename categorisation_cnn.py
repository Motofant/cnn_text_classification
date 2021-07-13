# libraries
import pandas as pd 
import keras.optimizers as ko
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import MaxPooling1D, Dense, Conv1D
from texttable import Texttable
from datetime import datetime

# custom files
import pre_proc as p 
import pipeline as pl
from data_gen import DataGenerator
import cnn_config as cc

# to prevent tf-bug with inputdata (anugrahasinha, https://github.com/tensorflow/tensorflow/issues/38503)
tf.compat.v1.disable_eager_execution() 

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

def usedNetwork(network_index, input_shape, output_dim):
    ## defines network used

    # network_index: int, basis to determain network
    # input_shape: shape of datainput to design inputlayer  

    if network_index == 0:
        return newNetwork_ord(input_shape, output_dim)
    elif network_index == 1:
        return newNetwork_bow(input_shape, output_dim)
    elif network_index == 2:
        return newNetwork_w2v(input_shape, output_dim)
    elif network_index == 3:
        return newNetwork_bow_for_w2v(input_shape, output_dim)
    elif network_index == 4:
        return newNetwork_w2v_for_bow(input_shape, output_dim)

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
    else:
        in_cat = []
        in_text = pd.read_table(input_file,header = None).to_numpy()

    print("shape of inputdata: ")
    print((in_text.shape[0],text_l,word_l))
    print()
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
def newNetwork_bow_for_w2v(in_shape, output_dim):

    model = Sequential()

    model.add(Conv1D(128,20,input_shape=in_shape,activation="relu",padding="same"))
    model.add(MaxPooling1D(3,data_format="channels_first"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Flatten())
    model.add(Dense(output_dim,activation="softmax"))

    optimizer = keras.optimizers.Adam(lr=0.00001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'],experimental_run_tf_function=False)

    print("compiled succesfully")
    
    return model

def newNetwork_bow(in_shape, output_dim):

    model = Sequential()

    model.add(Conv1D(128,20,input_shape=in_shape,activation="sigmoid",padding="same"))
    model.add(MaxPooling1D(3,data_format="channels_first"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Flatten())
    model.add(Dense(output_dim,activation="softmax"))

    optimizer = keras.optimizers.Adam(lr=0.00001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'],experimental_run_tf_function=False)
 
    print("compiled succesfully")
    
    return model

def newNetwork_w2v_for_bow(in_shape, output_dim):
    ## create network (save)
    # https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
    # https://keras.io/examples/nlp/pretrained_word_embeddings/


    model = Sequential()

    model.add(Conv1D(64,36,dilation_rate=9,input_shape=in_shape,activation="sigmoid",padding="valid"))
    model.add(MaxPooling1D(2,data_format="channels_first"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Flatten())
    model.add(Dense(output_dim, activation="softmax"))
    
    optimizer = keras.optimizers.Adam(lr=0.00001)
    model.compile( loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'],experimental_run_tf_function=False)

    print("compiled succesfully")

    return model

def newNetwork_w2v(in_shape, output_dim):
    ## create network (save)
    # https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
    # https://keras.io/examples/nlp/pretrained_word_embeddings/

    model = Sequential()

    model.add(Conv1D(64,36,dilation_rate=9,input_shape=in_shape,activation="relu",padding="valid"))
    model.add(MaxPooling1D(2,data_format="channels_first"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Flatten())
    model.add(Dense(output_dim, activation="softmax"))
    
    optimizer = keras.optimizers.Adam(lr=0.0001)
    model.compile( loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'],experimental_run_tf_function=False)

    print("compiled succesfully")

    return model

def newNetwork_ord(in_shape, output_dim):
    ## create network (save)
    # https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
    # https://keras.io/examples/nlp/pretrained_word_embeddings/


    model = Sequential()

    model.add(Conv1D(64,4,input_shape=in_shape,activation="relu",padding="valid"))
    model.add(MaxPooling1D(2,data_format="channels_first"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Flatten())
    model.add(Dense(output_dim, activation="softmax"))
    
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

    # read configs
    # get variables
    config_cnn = cc.getProf("cnn")
    con_name = config_cnn[0]
    input_dir = config_cnn[1]
    weight_save = config_cnn[2]
    training = True if config_cnn[3] == "True" else False
    load_nn = True if config_cnn[4] == "True" else False

    config = pl.loadConfig(con_name)
    classes = p.getCategories(config[7])
    cat_size = len(classes)
    Text_length,word_vec_length = netParams(int(config[2]),cat_size,int(config[3]),config[4])
    nn_input_size = (Text_length,word_vec_length)
    network_id = int(config[8])
    batch_training = True if config[9] == "True" else False
    batch_size = int(config[10])
    no_epoch = int(config[11])

    # read inputdata
    if batch_training:
        # get filelist from inputdir
        input_files = [f for f in os.listdir(input_dir) if "train" in f]
    else:
        input_text, valid_class = readFile(input_dir, training, Text_length, word_vec_length,cat_size)
    print("Finished: Read input")

    # create CNN
    model = usedNetwork(network_id, nn_input_size, cat_size)

    ## if already used -> use weights
    ## for testing weights have to be used
    if load_nn or not training:
        model.load_weights(weight_save)

    ## show model
    model.summary()

    # start process of training
    if training:
        if batch_training:
            # for batchtraining start datagenerator 
            trainings_train_gen = DataGenerator(input_files, input_dir,training, Text_length, word_vec_length, cat_size,1, batch_size,1)
            print("train data gen done")

            # train model
            history = model.fit_generator(generator= trainings_train_gen, epochs =no_epoch)      
        
        else:
            # train model
            history = model.fit(x = input_text,y =valid_class,shuffle = True,epochs=no_epoch, batch_size=batch_size)
        print("Finished: Train Network")

        # update weights after training 
        model.save_weights(weight_save)
    
        # show trainaccuracy
        visualHist(history)

    else:
        # Prediction for all texts in input
        predictions = model.predict(input_text)        
        print("Finished: Prediction")

        # find category with highest probability
        out = []
        for row in predictions:
            row = list(row)
            i = max(row)
            out.append(row.index(i))
        print("Finished: Calculating category with highest probability")     

        # save calcualted categories 
        name = "result_"+con_name+"_"+str(datetime.now().strftime("%H-%M-%S"))+".csv"
        pd.DataFrame(out).to_csv("./results/"+name, header = None, index = False)
        print("Finished: Saving Results")