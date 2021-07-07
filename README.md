# Pipeline to categorize Newsarticles via a CNN
testet on Python 3.7.3

## Required Libraries:
- Pre-processing
	- spacy (v2.3.2)
	- numpy (v1.20.0)
	- pandas
	- keras (v2.4.3)

- Pipeline
	- configparser

- Neural Network
	- Tensorflow (v2.3.1)
	- keras (v2.4.3)


## First Start:
	
## Usage:
### CommandlineArgs
- use Pipeline
	- python ./pipeline.py [-f] [-n] config_name input_directory
		- -f -> 
		- -n -> not final set -> add more 
- create dictionaries

### Variables
Values can be accessed in configuration

#### Filelocations
- dictionary [dict_pos]
- ordinal encoded save [save_dir]
- output files [output_dir]
- file containing the categorynames [cat_pos]
- file containing stopwords [stop_word]

#### Preprocessing 
- word selection process [pre_proc]
	0. no wordelimination
	1. wordtyp
	2. part of sentence
	3. tf-idf
- Encoding [coding]
	0. Ordinal
	1. BOW
	2. ~~One Hot~~ (only partially implemented)
	3. W2V
- batchencoding [batch_enc]
- batchsize [batch_size]

#### Network
- Typ [net_typ]
	0. Ordinal
	1. BOW
	2. W2V
	3. BOW-network modified to be used with W2V-encoded data
	4. W2V-network modified to be used with BOW-encoded data
- epochs [epochs]

#### other saved Values
- number of encoded texts [text_count]
- maximum number of word in text [word_max]

### Inputdata
- .csv/txt files 