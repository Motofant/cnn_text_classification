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
input_file = "C:/Users/Erik/Documents/Uni/BA/Repo/cnn_text_classification/output/defaulttwo_t.csv"# ordinal encoded input
#input_file = "C:/Users/Erik/Documents/Uni/BA/Repo/cnn_text_classification/output/defaulttest.csv"# ordinal encoded input
input_file = "C:/Users/Erik/Desktop/test1.csv"
input_file = "C:/Users/Erik/Documents/Uni/BA/Repo/cnn_text_classification/output/test_0_1_.csv"
#input_file = "C:/Users/Erik/Documents/Uni/BA/Repo/cnn_text_classification/test/test_0_0_fill_2.csv"
input_file = "C:/Users/Erik/Documents/Uni/BA/Repo/cnn_text_classification/cnn_input/train_0_0_0.csv"
input_file = "C:/Users/Erik/Documents/Uni/BA/Repo/cnn_text_classification/cnn_input/test_0_0_0.csv"

model_save = "./model/model.h5"
weight_save = "./weight/weight.h5"
weight_save = "C:/Users/Erik/Documents/Uni/BA/Repo/cnn_text_classification/weight/weight.h5"

# input
training = False
load_nn = not training
input_file = ""
fp = "C:/Users/Erik/Documents/Uni/BA/Repo/cnn_text_classification/output/gb/"
if training:
    input_file = "C:/Users/Erik/Documents/Uni/BA/Repo/cnn_text_classification/output/ww2v/oute.csv"
    
    input_files = [f for f in os.listdir(fp) if "train" in f]
    #print(input_files)
    #exit()
else:
    input_file = "C:/Users/Erik/Documents/Uni/BA/Repo/cnn_text_classification/output/gb/test_2_1_0.csv"
    pass
    #exit()




# inputtype
one_hot = False

# inputparams
Text_length = 8487 #also bowlength

word_vec_length = 1 # only not 1 in oneHot

cat_size = 9

# data
input_categories = np.array([]) # only when trainingsdata
input_text = np.array([])

# Neural Network
#nn_input_size = (581, 1)
nn_input_size = (1200, 9)
nn_input_size = (8487, 1)
#endregion

#### Functions ####
#region
def setShape(vec_length, word_rep_length):
    # vec_lec: length of vector representing the entire text (bow -> lex_size/ other -> text_size) 
    # word_rep_length: length of vector representing a single word (if not one-Hot -> 1)
    return(vec_length,word_rep_length)

def setCats(path):
    return pd.read_csv(path,header = None)[0].to_numpy()

def showResult(prediction, classes):
    table = Texttable()
    table.set_cols_dtype(["t","f"])
    table.add_row(["Class", "Percantage\nin %"])
    i = 0
    for element in classes:
        table.add_row([element, prediction[i]*100])
        i += 1 

    return table 

def readFile(input_file, train, text_l, word_l,cat_size):
    if train:
        in_cat = pd.read_table(input_file,usecols=[0],header = None).to_numpy()
        in_text = pd.read_table(input_file,usecols=list(range(text_l+1))[1:],header = None).to_numpy()
        print(len(in_text[0]))
    else:
        in_cat = []
        in_text = pd.read_table(input_file,header = None).to_numpy()
        
    
    return  in_text.reshape((in_text.shape[0],text_l,word_l)),keras.utils.to_categorical(in_cat,cat_size)

def visualHist(history):
    
    # test visual
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

def newNetwork_old(in_shape):
    ## create network (save)
    # https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
    # https://keras.io/examples/nlp/pretrained_word_embeddings/


    model = Sequential()
    
    model.add(Conv1D(64,3,input_shape=in_shape,activation="relu",padding="valid"))
    model.add(MaxPooling1D(3,data_format="channels_first"))
    model.add(keras.layers.BatchNormalization())
    #model.add(Dropout(0.3))
    
    
    model.add(Conv1D(64,3, activation="relu",padding="valid"))
    model.add(MaxPooling1D(pool_size = 3))
    model.add(keras.layers.BatchNormalization())
    #model.add(Dropout(0.3))
    
    model.add(Conv1D(64,3, activation="relu",padding="valid"))
    model.add(MaxPooling1D(2,data_format="channels_first"))
    model.add(keras.layers.BatchNormalization())
    #model.add(Dropout(0.3))
     
    model.add(Conv1D(64,3, activation="relu",padding="valid"))
    model.add(MaxPooling1D(2,data_format="channels_first"))
    model.add(keras.layers.BatchNormalization())
    #model.add(Dropout(0.3))
            

    model.add(Conv1D(64,5, activation="relu",padding="valid"))
    model.add(MaxPooling1D(2,data_format="channels_first"))
    model.add(keras.layers.BatchNormalization())
    #model.add(Dropout(0.3))
    
    model.add(Conv1D(64,5, activation="relu",padding="valid"))
    model.add(MaxPooling1D(2,data_format="channels_first"))
    model.add(keras.layers.BatchNormalization())
    #model.add(Dropout(0.3))
       
    model.add(Conv1D(64,5, activation="relu",padding="valid"))
    model.add(MaxPooling1D(3,data_format="channels_first"))
    model.add(keras.layers.BatchNormalization()) # nötig damit nicht 0,1111
    #model.add(Dropout(0.3))
        
    #model.add(GlobalMaxPooling1D())
    model.add(keras.layers.Flatten())

    model.add(Dense(9,activation="softmax"))

    #optimizer = keras.optimizers.Adam(lr=0.001)
    optimizer = keras.optimizers.Adam()

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'],experimental_run_tf_function=False)
    print("compiled succesfully")

    return model

def newNetwork(in_shape):

    model = Sequential()

    model.add(Conv1D(512,10,10, input_shape=in_shape,activation="sigmoid",padding="same"))
    #model.add(MaxPooling1D(10,data_format="channels_first"))
    model.add(keras.layers.BatchNormalization())
    #model.add(Dropout(0.3))

    model.add(Conv1D(128,3,activation="sigmoid",padding="same"))
    model.add(MaxPooling1D(3,data_format="channels_first"))
    model.add(keras.layers.BatchNormalization())
    #model.add(Dropout(0.3))    

    model.add(Conv1D(128,3,activation="sigmoid",padding="same"))
    model.add(MaxPooling1D(3,data_format="channels_first"))
    model.add(keras.layers.BatchNormalization())
    #model.add(Dropout(0.3))     
    '''
    model.add(Conv1D(512,10,activation="relu",padding="same"))
    model.add(MaxPooling1D(5,data_format="channels_first"))
    model.add(keras.layers.BatchNormalization())
    #model.add(Dropout(0.3))
    
    model.add(Conv1D(512,10,activation="relu",padding="same"))
    model.add(MaxPooling1D(5,data_format="channels_first"))
    model.add(keras.layers.BatchNormalization())
    #model.add(Dropout(0.3))
    
    model.add(Conv1D(512,10,activation="relu",padding="same"))
    model.add(MaxPooling1D(5,data_format="channels_first"))
    model.add(keras.layers.BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Conv1D(512,10, input_shape=in_shape,activation="relu",padding="same"))
    model.add(MaxPooling1D(10,data_format="channels_first"))
    model.add(keras.layers.BatchNormalization())
    #model.add(Dropout(0.3))
    
    model.add(Conv1D(512,10,activation="relu",padding="same"))
    model.add(MaxPooling1D(5,data_format="channels_first"))
    model.add(keras.layers.BatchNormalization())
    #model.add(Dropout(0.3))

    model.add(Conv1D(512,10,activation="relu",padding="same"))
    model.add(MaxPooling1D(5,data_format="channels_first"))
    model.add(keras.layers.BatchNormalization())
    #model.add(Dropout(0.3))
    
    model.add(Conv1D(512,10,activation="relu",padding="same"))
    model.add(MaxPooling1D(5,data_format="channels_first"))
    model.add(keras.layers.BatchNormalization())
    #model.add(Dropout(0.3))
    '''
    '''
    model.add(Conv1D(64,20,5, activation="relu",padding="same"))
    model.add(MaxPooling1D(5,data_format="channels_first"))
    model.add(keras.layers.BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv1D(64,20,5, activation="relu",padding="same"))
    model.add(MaxPooling1D(5,data_format="channels_first"))
    model.add(keras.layers.BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv1D(64,20,5, activation="relu",padding="same"))
    model.add(MaxPooling1D(5,data_format="channels_first"))
    model.add(keras.layers.BatchNormalization())
    model.add(Dropout(0.3))
    '''
    model.add(keras.layers.Flatten())
    model.add(Dense(9,activation="softmax"))
    optimizer = keras.optimizers.Adam()
    optimizer = keras.optimizers.Adam(lr=0.0001)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'],experimental_run_tf_function=False)
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
    '''    
    model.add(Conv1D(64,3,activation="relu",padding="valid"))
    model.add(MaxPooling1D(3,data_format="channels_first"))
    model.add(keras.layers.BatchNormalization())
    '''
    '''
    model.add(Conv1D(64,9,activation="relu",padding="valid"))
    model.add(MaxPooling1D(3,data_format="channels_first"))
    model.add(keras.layers.BatchNormalization())


    model.add(Conv1D(64,9,activation="relu",padding="valid"))
    model.add(MaxPooling1D(3,data_format="channels_first"))
    model.add(keras.layers.BatchNormalization())
    '''
      
    #model.add(GlobalMaxPooling1D())
    
    model.add(keras.layers.Flatten())
    #model.add(Dropout(0.5))

    model.add(Dense(9, activation="softmax"))
    
    optimizer = keras.optimizers.Adam(lr=0.0001)
    
    model.compile( loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'],experimental_run_tf_function=False)
    #model.compile( loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'],experimental_run_tf_function=False)

    print("compiled succesfully")

    return model
#endregion

#### "Pipeline" ####
if __name__ == "__main__":
    # define vectors
    input_text = []
    valid_class = []

    # check where data is comming from
    if data_from_file:
        input_text, valid_class = readFile(input_file, training, Text_length, word_vec_length,cat_size)
        
        pass
    else:
        ## use Vector
        pass 
        #print(len(input_text[0]))
    if one_hot:
        input_text = keras.utils.to_categorical(input_text,word_vec_l)
        #input_text = keras.utils.to_categorical(input_text,278504,dtype="int8")
    
        
        #input_text = p.oneHot(input_text,278504,0)
    # create CNN
    model = newNetwork(nn_input_size)
    #print(input_text.shape)
        ## if already used -> use weights
    if load_nn:
        model.load_weights(weight_save)

        ## show model
    model.summary()



    # do stuff
    if training:
        #train_gen = DataGenerator()
        #model.fit_generator(generator= train_gen)
        
        ## training -> save weights in the end -> non result needed
            ### TODO: change epoches/batchsize ? 
        #history =model.fit(x = input_text,y =valid_class,shuffle = True,epochs=7, batch_size=50)
        
        
        trainings_train_gen = DataGenerator(input_files, fp,training, Text_length, word_vec_length, len(classes),1, 50,1)
        print("train data gen done")
        history = model.fit_generator(generator= trainings_train_gen, epochs =3)#, workers=4)        
            ### TODO: show accc improvement? 
            ### update weights
        model.save_weights(weight_save)
        #accuracy = model.evaluate(input_text,valid_class)
        #print(accuracy)
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