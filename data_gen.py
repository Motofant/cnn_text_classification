import keras
import numpy as np
from pre_proc import bagOfWords
import pandas as pd

class DataGenerator(keras.utils.Sequence):
    # datagen
    # Batchsize variable = 1. real batchsize defiend in length of output 
    def __init__(self, list_IDs, directory, training, dim, n_channels, n_classes, encoding, batch_size, n_o_files, shuffle = True):
        self.list_IDs = list_IDs
        self.directory = directory
        self.training = training
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.encoding = encoding
        self.batch_size = batch_size
        self.n_o_files = n_o_files

        self.shuffle = shuffle
        self.on_epoch_end()

    def readFile(self, input_file, directory, encoding, text_l, word_l ,n_classes):
        path = directory + input_file + ".csv"
        in_cat = pd.read_table(path,usecols=[0],header = None).to_numpy()
        in_text = pd.read_table(path,usecols=list(range(text_l+1))[1:],header = None).to_numpy()
        
        # Bag of Words
        if encoding == 1:
            in_text = np.reshape(np.array(bagOfWords(in_text, text_l)), (len(in_cat),self.dim, self.n_channels))
        # One Hot
        elif encoding == 2:
            # TODO: think about reading fct
            in_text = keras.utils.to_categorical(x,text_l)
        else:
            in_text = np.reshape(in_text, (len(in_cat),self.dim, self.n_channels))
        return in_text , keras.utils.to_categorical(in_cat, n_classes)


    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, list_IDs_temp):
        
        # Init
        #X = np.empty((self.batch_size, *self.dim, self.n_channels))
        X = np.empty((self.batch_size, self.dim, self.n_channels))
        
        y = np.empty((self.batch_size),dtype = int)

        # Full Input_data

        
        # generating Data
        for i, ID in enumerate(list_IDs_temp):
            # generate one Sample 
            X, y = self.readFile(ID, self.directory, self.encoding, self.dim, self.n_channels, self.n_classes)
        
        return X,y

    def __len__(self):
        return int(np.floor(len(self.list_IDs)/self.n_o_files))

    def __getitem__(self, index):
        #indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        indexes = self.indexes[index*self.n_o_files:(index+1)*self.n_o_files]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        
        X,y = self.__data_generation(list_IDs_temp)
        return X, y
