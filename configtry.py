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
        "output_dir": "./output/"

    }
    with open("test.ini", 'r+') as configfile:
        config.read(configfile)
        config.write(configfile)

def newProfMauell(name, tc, pp, co, wm, df, sf,od):
    config = cop.ConfigParser()
    config[name] = {
        "text_count" : str(tc),
        "pre_proc" : str(pp),
        "coding" : str(co),
        "word_max" : str(wm),
        "dict_pos" : df, 
        "save_dir" : sf,
        "output_dir": od
    }

    with open("test.ini", 'r+') as configfile:
        config.read(configfile)
        config.write(configfile)

def saveProf(name, tc, pp, co, wm, df, sf, od):
    config = cop.ConfigParser()
    with open("test.ini", 'r') as configfile:
        config.read_file(configfile)
    config.set(name,"text_count",str(tc))
    config.set(name,"pre_proc",str(pp))
    config.set(name,"coding",str(co))
    config.set(name,"word_max",str(wm))
    config.set(name,"dict_pos",df)
    config.set(name,"save_dir",sf)
    config.set(name,"output_dir",od)

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
