FROM python:3.7.3

# define workdir
WORKDIR /temp/
#--no-cache-dir --no-binary :all:
RUN pip install numpy==1.20.0 
RUN pip install pandas
RUN pip install tensorflow==2.3.1
RUN pip install keras==2.4.3
RUN pip install spacy==2.3.2
RUN pip install matplotlib
RUN pip install texttable

RUN python -m spacy download de

#RUN python -m spacy download en_core_web_sm

COPY . /temp/

CMD ["python","/temp/pipeline.py","-f", "def", "/temp/input/test_Training/"]

# to get logfile
# docker cp 36d9f3a18f6c:temp/log_pipeline.log ./save/