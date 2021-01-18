import csv
import spacy
import statistics as s


nlp = spacy.load('de')

paths = []

paths.append("D:/Datenbank/20k/gesellschaft_2_b.csv")
paths.append("D:/Datenbank/20k/kultur_2_b.csv")
paths.append("D:/Datenbank/20k/leben_2_b.csv")
paths.append("D:/Datenbank/20k/politik_2_b.csv")
paths.append("D:/Datenbank/20k/reisen_2_b.csv")
paths.append("D:/Datenbank/20k/sport_2_b.csv")
paths.append("D:/Datenbank/20k/technik_2_b.csv")
paths.append("D:/Datenbank/20k/wirtschaft_2_b.csv")
paths.append("D:/Datenbank/20k/wissenschaft_2_b.csv")

path = "D:/Datenbank/politik_b.csv"

out_file = "politik"
def countWord(text):
    doc = nlp(text)
    return len(doc)  


def printOut(filename, text_lengths):
    filename =  'C:/Users/Erik/Desktop/wordcount_20k.csv'
    with open(filename, mode='a+', newline='') as dictFile:
        writer = csv.writer(dictFile)
        writer.writerow(text_lengths)

#        for number in text_lengths:
 #           i = [number]
  #          writer.writerow(i)

for text in paths:


    text_sizes = []

    with open(text, mode='r', newline='', encoding="utf8") as dictFile:
        reader= csv.reader(dictFile, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
    #        print("hey")
            text_sizes.append(countWord(row[0]))


    y = max(text_sizes)
    z = min(text_sizes)
    x = s.mean(text_sizes)
    #z = s.median(text_sizes)

    text_sizes = [x,y,z] + text_sizes

    printOut(out_file, text_sizes)

