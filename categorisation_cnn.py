#import keras
import numpy as np
from keras.models import Sequential
from keras.layers import MaxPooling1D, Dense, Dropout, Flatten, Conv2D, Conv1D, MaxPool1D,GlobalMaxPooling1D
import csv
import pandas as pd 


# TODO: encode to onehot when asked
# define variables
input_file = "C:/Users/Erik/Desktop/cnn_testinput.csv"# bow encoded input
cat_input = "C:/Users/Erik/Desktop/cnn_testinput_cat.csv"
training = True
Text_length = 50 # for testing purposes

input_categories = [] # only when trainingsdata

## import data, size is constant -> pd or np can be used


input_text = np.asarray([pd.DataFrame(pd.read_csv(input_file, header=None, index_col= False)).to_numpy()]).astype("int64")
if training:
    input_categories =np.asarray([pd.DataFrame(pd.read_csv(cat_input, header=None, index_col= False)).to_numpy()]).astype("int64")
'''
input_text = pd.DataFrame(pd.read_csv(input_file, header=None, index_col= False)).to_numpy()
if training:
    input_categories =pd.DataFrame(pd.read_csv(cat_input, header=None, index_col= False)).to_numpy()

print(input_text)
'''
'''
## alternative with csv
with open(input_file, mode = "r", newline = "\n") as text_file:
    reader = csv.reader(text_file, delimiter= ",",  quotechar='"', quoting=csv.QUOTE_MINIMAL )
    for row in reader:
        numbers = []
        for item in row:
            numbers.append(int(item))
        input_text.append(numbers)
'''
#print(input_text)
#print(input_categories)
#for i in input_text[1]:
#    pass
    #print(str(j))
#print(np.isnan(input_text))
## define input

## get needed dimensions from input (constant size )


## create network (save)
# https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
# https://keras.io/examples/nlp/pretrained_word_embeddings/

model = Sequential()
model.add(Conv1D(32,3,input_shape=(None,24), activation="relu",padding="same"))
model.add(MaxPooling1D(2,data_format="channels_first"))
model.add(Conv1D(16,3, activation="relu",padding="same"))
model.add(MaxPooling1D(2,data_format="channels_first"))
model.add(Conv1D(8,3, activation="relu"))
model.add(GlobalMaxPooling1D())
model.add(Dense(9, activation="relu"))
model.add(Dropout(0.5)) #ony for testing needed -> for limited datasets



## Mach zeugs
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print("compiled succesfully")
model.fit(input_text,input_categories,epochs=10, batch_size=9)

## output

model.summary()
