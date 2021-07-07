# configtry 
# works for python 3, sonst try gefordert

import configparser as cop
import os

def existProf(config_location, name):
    config = cop.ConfigParser()
    config.read(config_location)
    return config.has_section(name)


def newProf(name):
    config = cop.ConfigParser()
    config[name] = {
        "text_count" : "0",
        "pre_proc" : "0",
        "coding" : "0",
        "word_max" : "1200",
        "dict_pos" : "./dictionary/lex.csv", 
        "save_dir" : "./save/",
        "output_dir": "./output/",
        "cat_pos": "./dictionary/cat.csv",
        "net_typ": "0",
        "batch_enc": "False",
        "batch_size": "50",
        "epochs":"10",
        "stop_word":"./stopword/"

    }
    with open("test.ini", 'r+') as configfile:
        config.read(configfile)
        config.write(configfile)

def newProfMauell(name, vars):
    config = cop.ConfigParser()
    config[name] = {
        "text_count" : str(vars[0]),
        "pre_proc" : str(vars[1]),
        "coding" : str(vars[2]),
        "word_max" : str(vars[3]),
        "dict_pos" : vars[4], 
        "save_dir" : vars[5],
        "output_dir": vars[6],
        "cat_pos": vars[7],
        "net_typ": str(vars[8]),
        "batch_enc": str(vars[9]),
        "batch_size": str(vars[10]),
        "epochs": str(vars[11]),
        "stop_word":vars[12]
    }

    with open("test.ini", 'r+') as configfile:
        config.read(configfile)
        config.write(configfile)

def saveProf(name, vars):
    config = cop.ConfigParser()
    with open("test.ini", 'r') as configfile:
        config.read_file(configfile)
    config.set(name,"text_count",str(vars[0]))
    config.set(name,"pre_proc",str(vars[1]))
    config.set(name,"coding",str(vars[2]))
    config.set(name,"word_max",str(vars[3]))
    config.set(name,"dict_pos",vars[4])
    config.set(name,"save_dir",vars[5])
    config.set(name,"output_dir",vars[6])
    config.set(name,"cat_pos",vars[7])
    config.set(name,"net_typ",str(vars[8]))
    config.set(name,"batch_enc",str(vars[9]))
    config.set(name,"batch_size",str(vars[10]))
    config.set(name, "epochs",str(vars[11]))
    config.set(name, "stop_word",vars[12])

    with open("test.ini", 'w') as configfile:
        config.write(configfile)

def updateProf(name, field, value):
    config =cop.ConfigParser()
    config.read("test.ini")
    config.set(name,field, value)
    with open("test.ini", 'w') as configfile:
        config.write(configfile)

def getProf(name):
    config_out = []
    config =cop.ConfigParser()
    config.read("test.ini")
    for item in config[name]:
        config_out.append(config[name].get(item))
    return config_out
