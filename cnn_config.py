import configparser as cop
import os

def existProf(config_location, name):
    config = cop.ConfigParser()
    config.read(config_location)
    return config.has_section(name)


def newProf(name):
    config = cop.ConfigParser()
    config[name] = {
        "config" : "def",
        "input" : "./input/",
        "weight" : "./weight/weight.h5",
        "training" : "True"
    }
    with open("cnn.ini", 'r+') as configfile:
        config.read(configfile)
        config.write(configfile)


def saveProf(name, vars):
    config = cop.ConfigParser()
    with open("cnn.ini", 'r') as configfile:
        config.read_file(configfile)
    config.set(name,"config",str(vars[0]))
    config.set(name,"input",str(vars[1]))
    config.set(name,"weight",str(vars[1]))
    config.set(name,"training",str(vars[1]))


    with open("cnn.ini", 'w') as configfile:
        config.write(configfile)

def updateProf(name, field, value):
    config =cop.ConfigParser()
    config.read("cnn.ini")
    config.set(name,field, value)
    with open("cnn.ini", 'w') as configfile:
        config.write(configfile)

def getProf(name):
    config_out = []
    config =cop.ConfigParser()
    config.read("cnn.ini")
    for item in config[name]:
        config_out.append(config[name].get(item))
    return config_out
