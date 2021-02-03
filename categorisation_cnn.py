#import keras
import numpy as np
import keras
from keras.models import Sequential
#import keras.models
from keras.layers import MaxPooling1D, Dense, Dropout, Flatten, Conv2D, Conv1D, MaxPool1D,GlobalMaxPooling1D
import csv
import pandas as pd 
import keras.optimizers as ko
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
#import sklearn.model_selection as sms


tf.compat.v1.disable_eager_execution() # to prevent tf-bug with inputdata (anugrahasinha, https://github.com/tensorflow/tensorflow/issues/38503)
#np.set_printoptions(threshold=sys.maxsize)


#### Variables ####
#region
# Pipelinevariables
## already in pipeline-> not needed when porting
training = True


## new variables
data_from_file = True
load_nn = False


# necessary for current version
input_file = "C:/Users/Erik/Documents/Uni/BA/Repo/cnn_text_classification/output/defaulttwo_t.csv"# ordinal encoded input
input_file = "C:/Users/Erik/Documents/Uni/BA/Repo/cnn_text_classification/output/defaulttest.csv"# ordinal encoded input

model_save = "./model/model.h5"
weight_save = "./weight/weight.h5"
weight_save = "C:/Users/Erik/Documents/Uni/BA/Repo/cnn_text_classification/weight/weight.h5"



# inputtype
one_hot = False

# inputparams
Text_length = 1200 # also bowlength
word_vec_length = 1 # only not 1 in oneHot

cat_size = 9

# data
input_categories = np.array([]) # only when trainingsdata
input_text = np.array([])

# Neural Network
nn_input_size = (1200, 1)

#endregion

#### Functions ####
#region
def setShape(vec_length, word_rep_length):
    # vec_lec: length of vector representing the entire text (bow -> lex_size/ other -> text_size) 
    # word_rep_length: length of vector representing a single word (if not one-Hot -> 1)
    return(vec_length,word_rep_length)

def readFile(infile, train, text_l, word_l,cat_size):
    input_text = []
    input_categories = []

    with open(input_file, mode = "r", newline = "\n") as text_file:
        reader = csv.reader(text_file, delimiter= "\t",  quotechar='"', quoting=csv.QUOTE_MINIMAL )
        i = 0
        if train:

            for row in reader:
                input_categories = np.append(input_categories,int(float(row[0])))
                numbers = []
                for item in row[1:]:
                    numbers.append(int(float(item)))
                input_text= np.append(input_text,numbers)
                i +=1
        else:
            for row in reader:
                numbers = []
                for item in row:
                    numbers.append(int(float(item)))
                input_text= np.append(input_text,numbers)
                i +=1
        
        return np.array(input_text.reshape((int(input_text.shape[0]/text_l),text_l,word_l))), keras.utils.to_categorical(input_categories,cat_size)

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

def newNetwork(in_shape):
    ## create network (save)
    # https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
    # https://keras.io/examples/nlp/pretrained_word_embeddings/
    model = Sequential()

    model.add(Conv1D(64,3,input_shape=in_shape,activation="relu",padding="valid"))

    model.add(MaxPooling1D(2,data_format="channels_first"))
    model.add(keras.layers.BatchNormalization())
    #model.add(Dropout(0.5))

    model.add(Conv1D(64,3, activation="relu",padding="valid"))
    model.add(MaxPooling1D(pool_size = 2))
    model.add(keras.layers.BatchNormalization())
    #model.add(Dropout(0.5))

    model.add(Conv1D(64,3, activation="relu",padding="valid"))
    model.add(MaxPooling1D(2,data_format="channels_first"))
    model.add(keras.layers.BatchNormalization())
    #model.add(Dropout(0.5))

    model.add(Conv1D(64,5, activation="relu",padding="valid"))
    model.add(MaxPooling1D(2,data_format="channels_first"))
    model.add(keras.layers.BatchNormalization())
    #model.add(Dropout(0.5))

    model.add(Conv1D(64,5, activation="relu",padding="valid"))
    model.add(MaxPooling1D(2,data_format="channels_first"))
    model.add(keras.layers.BatchNormalization())
    #model.add(Dropout(0.5))

    model.add(Conv1D(64,5, activation="relu",padding="valid"))
    model.add(MaxPooling1D(2,data_format="channels_first"))
    model.add(keras.layers.BatchNormalization()) # nötig, damit nicht 0,1111
    #model.add(Dropout(0.5))

    #model.add(GlobalMaxPooling1D())

    model.add(keras.layers.Flatten())
    model.add(Dropout(0.5)) #ony for testing needed -> for limited datasets
    model.add(Dense(9, activation="softmax"))

    model.compile( loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'],experimental_run_tf_function=False)
    print("compiled succesfully")

    return model
#endregion

#### "Pipeline" ####
'''
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

# create CNN
model = newNetwork(nn_input_size)

    ## if already used -> use weights
if load_nn:
    model.load_weights(weight_save)

    ## show model
model.summary()



# do stuff
if training:
    ## training -> save weights in the end -> non result needed
        ### TODO: change epoches/batchsize ? 
    history =model.fit(x = input_text,y =valid_class,shuffle = True,epochs=10, batch_size=10)
        ### TODO: show accc improvement? 
        ### update weights
    model.save_weights(weight_save)
    accuracy = model.evaluate(input_text,valid_class)
    print(accuracy)
    visualHist(history)
else:
    ## test -> no save requiered (no weight updates) -> Show result
    predictions = model.predict(input_text[:5])
    
    for i in range(5):
        print(valid_class[i])
        print(predictions[i])
        i += 1


        ### TODO: Show results
'''