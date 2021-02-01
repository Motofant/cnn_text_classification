#import keras
import numpy as np
import keras
from keras.models import Sequential
import keras.models
from keras.layers import MaxPooling1D, Dense, Dropout, Flatten, Conv2D, Conv1D, MaxPool1D,GlobalMaxPooling1D
import csv
import pandas as pd 
import pre_proc as p
import keras.optimizers as ko
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn.model_selection as sms


tf.compat.v1.disable_eager_execution() # thanks to anugrahasinha, https://github.com/tensorflow/tensorflow/issues/38503

#### Variables ####

# necessary for beta-version
input_file = "C:/Users/Erik/Documents/Uni/BA/Repo/cnn_text_classification/output/defaulttwo_t.csv"# ordinal encoded input
input_file = "C:/Users/Erik/Documents/Uni/BA/Repo/cnn_text_classification/output/defaulttest.csv"# ordinal encoded input

#cat_input = "C:/Users/Erik/Desktop/cnn_testinput_cat.csv"
#np.set_printoptions(threshold=sys.maxsize)

# inputtype
training = True
one_hot = False

# inputparams
Text_length = 1200 # also bowlength
word_vec_length = 1 # only not zero in oneHot

cat_size = 9

# data
input_categories = np.array([]) # only when trainingsdata
input_text = np.array([])

# Neural Network
nn_input_size = (1200, 1)

#### Function ####

def learnPastSetting():
    # TODO
    # loads weights
    # chacks if inpput is the same
    pass

def setShape(vec_length, word_rep_length):
    # vec_lec: length of vector representing the entire text (bow -> lex_size/ other -> text_size) 
    # word_rep_length: length of vector representing a single word (if not one-Hot -> 1)
    return(vec_length,word_rep_length)

def makeMore(in_arr, length):
    y = [ [] for _ in range(length) ]
    i = 0
    for e in in_arr:
        y[i].append(e)
        if i <length-1:
            i +=1
        else:
            i = 0
        
        #return np.array(y)
    return y

def readDataFile(infile, training, one_hot):
    pass
# TODO: encode to onehot when asked







## import data, size is constant -> pd or np can be used
## inputdesign: if trainingsdata -> row[0] = classencoding (needs to be transformed in oH)








'''
input_text = np.asarray([pd.DataFrame(pd.read_csv(input_file, header=None, index_col= False)).to_numpy()]).astype("int64")
if training:
    input_categories =np.asarray([pd.DataFrame(pd.read_csv(cat_input, header=None, index_col= False)).to_numpy()]).astype("int64")
'''
'''
input_text = pd.DataFrame(pd.read_csv(input_file, header=None, index_col= False)).to_numpy()
if training:
    input_categories =pd.DataFrame(pd.read_csv(cat_input, header=None, index_col= False)).to_numpy()

print(input_text)
'''

## alternative with csv
with open(input_file, mode = "r", newline = "\n") as text_file:
    reader = csv.reader(text_file, delimiter= "\t",  quotechar='"', quoting=csv.QUOTE_MINIMAL )
    i = 0
    for row in reader:
        #input_categories=np.append(input_categories,p.oneHot(row[0],cat_size, 1))
        input_categories = np.append(input_categories,int(float(row[0])))
        numbers = []
        for item in row[1:]:
            numbers.append(int(float(item)))
        input_text= np.append(input_text,numbers)
        i +=1

input_text = np.array(input_text.reshape((int(input_text.shape[0]/Text_length),Text_length,word_vec_length))) # 1 has to be varaible for one Hot 
#k = makeMore(input_categories, cat_size)
k = input_categories

## define input

## get needed dimensions from input (constant size )


## create network (save)
# https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
# https://keras.io/examples/nlp/pretrained_word_embeddings/

inp = keras.Input(nn_input_size)
outz = keras.utils.to_categorical(k,cat_size)
#print(input_text)
#print(outz)
model = Sequential()
#model = keras.Model(inp, [out_0,out_1,out_2,out_3,out_4,out_5,out_6,out_7,out_8])
#model = keras.Model(inp,np.array(k))
model.add(Conv1D(64,3,input_shape=(1200,1),activation="relu",padding="valid"))

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

model.add(Conv1D(64,3, activation="relu",padding="valid"))
model.add(MaxPooling1D(2,data_format="channels_first"))
model.add(keras.layers.BatchNormalization())
#model.add(Dropout(0.5))

model.add(Conv1D(64,3, activation="relu",padding="valid"))
model.add(MaxPooling1D(2,data_format="channels_first"))
model.add(keras.layers.BatchNormalization())
#model.add(Dropout(0.5))

model.add(Conv1D(64,3, activation="relu",padding="valid"))
model.add(MaxPooling1D(2,data_format="channels_first"))
model.add(keras.layers.BatchNormalization()) # nötig, damit nicht 0,1111
#model.add(Dropout(0.5))

#model.add(GlobalMaxPooling1D())

model.add(keras.layers.Flatten())
model.add(Dropout(0.5)) #ony for testing needed -> for limited datasets
model.add(Dense(9, activation="softmax"))



'''
inp = keras.Input(nn_input_size)

x = Conv1D(64,3,activation="sigmoid",padding="valid")(inp)
x = MaxPooling1D(2,data_format="channels_first")(x)
x = Conv1D(64,3,activation="sigmoid",padding="valid")(x)
x = MaxPooling1D(2,data_format="channels_first")(x)
x = Conv1D(64,3,activation="sigmoid",padding="valid")(x)
x = MaxPooling1D(2,data_format="channels_first")(x)
x = GlobalMaxPooling1D()(x)

x = Dropout(0.5)(x)

out_0 = Dense(1, activation="sigmoid")(x)
out_1 = Dense(1, activation="sigmoid")(x)
out_2 = Dense(1, activation="sigmoid")(x)
out_3 = Dense(1, activation="sigmoid")(x)
out_4 = Dense(1, activation="sigmoid")(x)
out_5 = Dense(1, activation="sigmoid")(x)
out_6 = Dense(1, activation="sigmoid")(x)
out_7 = Dense(1, activation="sigmoid")(x)
out_8 = Dense(1, activation="sigmoid")(x)
'''


#model = keras.Model(inp, [out_0,out_1,out_2,out_3,out_4,out_5,out_6,out_7,out_8])
#model = keras.Model(inp, k)




## Mach zeugs
model.compile( loss='mean_squared_error', optimizer="adam", metrics=['accuracy'],experimental_run_tf_function=False)
print("compiled succesfully")

#model.fit(x = input_text,y =[k[0],k[1],k[2],k[3],k[4],k[5],k[6],k[7],k[8]],epochs=50, batch_size=450)
history =model.fit(x = input_text,y =outz,shuffle = True,epochs=20, batch_size=10)
#history = model.fit(x = input_text,y =[np.array(k[0]),np.array(k[1]),np.array(k[2]),np.array(k[3]),np.array(k[4]),np.array(k[5]),np.array(k[6]),np.array(k[7]),np.array(k[8])],epochs=50, batch_size=450)

model.summary()
## output

accuracy = model.evaluate(input_text,outz)
#print('Accuracy: %.2f' % (accuracy*100))
print(accuracy)
predictions = model.predict(input_text)
for i in range(5):
#print(input_text.tolist())
     print("prediction: ")
     print(predictions[i])
     print("real")
     print(outz[i])

#print(history.history.keys())





# test visual
plt.plot(history.history['accuracy'])
# plt.plot(history.history['dense_1_accuracy'])
# plt.plot(history.history['dense_2_accuracy'])
# plt.plot(history.history['dense_3_accuracy'])
# plt.plot(history.history['dense_4_accuracy'])
# plt.plot(history.history['dense_5_accuracy'])
# plt.plot(history.history['dense_6_accuracy'])
# plt.plot(history.history['dense_7_accuracy'])
# plt.plot(history.history['dense_8_accuracy'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['dense_accuracy'], loc='lower right')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
# plt.plot(history.history['dense_1_loss'])
# plt.plot(history.history['dense_2_loss'])
# plt.plot(history.history['dense_3_loss'])
# plt.plot(history.history['dense_4_loss'])
# plt.plot(history.history['dense_5_loss'])
# plt.plot(history.history['dense_6_loss'])
# plt.plot(history.history['dense_7_loss'])
# plt.plot(history.history['dense_8_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['dense_loss'], loc='upper right')
plt.show()