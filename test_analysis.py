import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import csv
import pre_proc as p
import numpy as np
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
rounding_n = 100
def rounding(in_arr, rounding_number):
    out_arr =[]

    for number in in_arr:
        out_arr.append(int(np.round(number/rounding_number)*rounding_number))
    
    return out_arr




# read textsizes

path = 'C:/Users/Erik/Desktop/wordcount_20k.csv'
information = []
text_length = []
with open(path, mode="r", newline='') as dictFile:
    reader = csv.reader(dictFile, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)
    #next(dictFile)
    for row in reader:
        i = 3
        #print("Anzahl der Felder: "+ str(len(row)))
        information.append([float(row[0]),int(row[1]),int(row[2])])
        tl = []
        while i < len(row) and i<=17500:
            tl.append(int(row[i]))
            i += 1
        tl = rounding(tl,rounding_n)
        text_length.append(tl)


# create diagramm
bow = []

for l in text_length:
    bag = list(p.bagOfWords(l,10801))
    k = []
    it = 0
    for i in bag:
        if it%rounding_n == 0:
            k.append(i)
        it += 1
   # k = [i for i in list(p.bagOfWords(l,10001)) if i != 0]

    bow.append(k)
    
    #bow.append(list(p.bagOfWords(l,10001)))
#bow = p.bagOfWords(np.asarray(text_length[1]),10000)
#print(p.bagOfWords([0,3,4,1,4,4,4,4,4,4,4],5))
to_high =[]
gesamt  = []
for i in bow:
    l = 0
    for j in i:

        l += j
    gesamt.append(l)

for i in bow:
    l = 0
    for j in i[13:]:

        l += j
        to_high.append(l)


#l = float(k)/90000
#print(l)
t = np.arange(0.,10001.,1)
t = np.arange(0.,10801.,rounding_n)


i = np.arange(0,10001,rounding_n)
print(len(t))
'''
# normales plotting
plt.plot(bow[0],"s",label ="gesellschaft")
plt.plot(bow[1],label ="kultur")
plt.plot(bow[2],label ="leben")
plt.plot(bow[3],label ="politik")
plt.plot(bow[4],label ="reisen")
plt.plot(bow[5],label ="sport")
plt.plot(bow[6],label ="technik")
plt.plot(bow[7],label ="wirtschaft")
plt.plot(bow[8],".",label ="wissenschaft")

# bar, überlappend
plt.bar(t,bow[0],label ="gesellschaft")
plt.bar(t,bow[1],label ="kultur")
plt.bar(t,bow[2],label ="leben")
plt.bar(t,bow[3],label ="politik")
plt.bar(t,bow[4],label ="reisen")
plt.bar(t,bow[5],label ="sport")
plt.bar(t,bow[6],label ="technik")
plt.bar(t,bow[7],label ="wirtschaft")
plt.bar(t,bow[8],label ="wissenschaft")
'''
#bar,stacked
fig,ax = plt.subplots()
w = 70
'''
ax.bar(t,bow[0],w,label ="Gesellschaft")
ax.bar(t,bow[1],w,bottom=bow[0],label ="Kultur")
ax.bar(t,bow[2],w,bottom=[sum(data) for data in zip(bow[0],bow[1])],label ="Leben")
ax.bar(t,bow[3],w,bottom=[sum(data) for data in zip(bow[0],bow[1],bow[2])],label ="Politik")
ax.bar(t,bow[4],w,bottom=[sum(data) for data in zip(bow[0],bow[1],bow[2],bow[3])],label ="Reisen")
ax.bar(t,bow[5],w,bottom=[sum(data) for data in zip(bow[0],bow[1],bow[2],bow[3],bow[4])],label ="Sport")
ax.bar(t,bow[6],w,bottom=[sum(data) for data in zip(bow[0],bow[1],bow[2],bow[3],bow[4],bow[5])],label ="Technik")
ax.bar(t,bow[7],w,bottom=[sum(data) for data in zip(bow[0],bow[1],bow[2],bow[3],bow[4],bow[5],bow[6])],label ="Wirtschaft")
ax.bar(t,bow[8],w,bottom=[sum(data) for data in zip(bow[0],bow[1],bow[2],bow[3],bow[4],bow[5],bow[6],bow[7])],label ="Wissenschaft")
'''

'''
plt.bar(t,bow[0],w,label ="Gesellschaft")
plt.bar(t,bow[1],w,bottom=bow[0],label ="Kultur")
plt.bar(t,bow[2],w,bottom=[sum(data) for data in zip(bow[0],bow[1])],label ="Leben")
plt.bar(t,bow[3],w,bottom=[sum(data) for data in zip(bow[0],bow[1],bow[2])],label ="Politik")
plt.bar(t,bow[4],w,bottom=[sum(data) for data in zip(bow[0],bow[1],bow[2],bow[3])],label ="Reisen")
plt.bar(t,bow[5],w,bottom=[sum(data) for data in zip(bow[0],bow[1],bow[2],bow[3],bow[4])],label ="Sport")
plt.bar(t,bow[6],w,bottom=[sum(data) for data in zip(bow[0],bow[1],bow[2],bow[3],bow[4],bow[5])],label ="Technik")
plt.bar(t,bow[7],w,bottom=[sum(data) for data in zip(bow[0],bow[1],bow[2],bow[3],bow[4],bow[5],bow[6])],label ="Wirtschaft")
plt.bar(t,bow[8],w,bottom=[sum(data) for data in zip(bow[0],bow[1],bow[2],bow[3],bow[4],bow[5],bow[6],bow[7])],label ="Wissenschaft")

#mpl.axis.XAxis.minorTicks(50)
#plt.xticks(i)
# design

plt.xlim([0,11000])
#plt.ylim([0,140])

ax.grid(True)
ax.minorticks_on()
ax.xaxis.set_major_locator(MultipleLocator(1000))
ax.xaxis.set_minor_locator(MultipleLocator(100))

#
plt.xlabel("Anzahl der Worte")
plt.ylabel("Anzahl der Texte")
plt.title("Verteilung der Artikellängen",fontsize = "large")
plt.legend()
plt.show()

'''
cat = ["Gesellschaft", "Kultur","Leben","Politik","Reisen","Sport","Technik","Wirtschaft","Wissenschaft"]
c = "c"
d= "mediumblue"

plt.bar(cat[0],gesamt[0],color = c)
plt.bar(cat[0],gesamt[0]-to_high[0],color = d)
plt.bar(cat[1],gesamt[1],color = c)
plt.bar(cat[1],gesamt[1]-to_high[1],color = d)
plt.bar(cat[2],gesamt[2],color = c)
plt.bar(cat[2],gesamt[2]-to_high[2],color = d)
plt.bar(cat[3],gesamt[3],color = c)
plt.bar(cat[3],gesamt[3]-to_high[3],color = d)
plt.bar(cat[4],gesamt[4],color = c)
plt.bar(cat[4],gesamt[4]-to_high[4],color = d)
plt.bar(cat[5],gesamt[5],color = c)
plt.bar(cat[5],gesamt[5]-to_high[5],color = d)
plt.bar(cat[6],gesamt[6],color = c)
plt.bar(cat[6],gesamt[6]-to_high[6],color = d)
plt.bar(cat[7],gesamt[7],color = c)
plt.bar(cat[7],gesamt[7]-to_high[7],color = d)
plt.bar(cat[8],gesamt[8],color = c)
plt.bar(cat[8],gesamt[8]-to_high[8],color = d)

line=[10000,10000]#,10000,10000,10000,10000,10000,10000,10000]
args = [-1,9]
plt.plot(args,line,color ="r")
plt.xlim([-0.5,8.5])
l = ax.get_xticklabels()
#plt.setp(l,rotation =45)
plt.show()