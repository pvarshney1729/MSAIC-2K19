import math
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
#Initialize Global variables 
docIDFDict = {}
avgDocLength = 0
porter = PorterStemmer()
stop_words = set(stopwords.words('english'))
def GetCorpus(inputfile,corpusfile):
    f = open(inputfile,"r",encoding="utf-8")
    fw = open(corpusfile,"w",encoding="utf-8")
    for line in f:
        passage = line.strip().lower().split("\t")[2]
        fw.write(passage+"\n")
    f.close()
    fw.close()



# The following IDF_Generator method reads all the passages(docs) and creates Inverse Document Frequency(IDF) scores for each unique word using below formula 
# IDF(q_i) = log((N-n(q_i)+0.5)/(n(q_i)+0.5)) where N is the total number of documents in the collection and n(q_i) is the number of documents containing q_i
# After finding IDF scores for all the words, The IDF dictionary will be saved in "docIDFDict.pickle" file in the current directory

def IDF_Generator(corpusfile, delimiter=' ', base=math.e) :

    global docIDFDict,avgDocLength

    docFrequencyDict = {}       
    numOfDocuments = 0   
    totalDocLength = 0

    for line in open(corpusfile,"r",encoding="utf-8") :
        doc = word_tokenize(line)
        doc_stemmed = [porter.stem(word) for word in doc]
        doc_stemmed = [w for w in doc_stemmed if not w in stop_words]
        totalDocLength += len(doc_stemmed)

        doc_stemmed = list(set(doc_stemmed)) # Take all unique words

        for word in doc_stemmed : #Updates n(q_i) values for all the words(q_i)
            if word not in docFrequencyDict :
                docFrequencyDict[word] = 0
            docFrequencyDict[word] += 1

        numOfDocuments = numOfDocuments + 1
        if (numOfDocuments%5000==0):
            print(numOfDocuments)                

    for word in docFrequencyDict:  #Calculate IDF scores for each word(q_i)
        docIDFDict[word] = math.log((numOfDocuments - docFrequencyDict[word] +0.5) / (docFrequencyDict[word] + 0.5), base) #"numOfDocuments - docFrequencyDict[word]" vs using "numOfDocuments"

    avgDocLength = totalDocLength / numOfDocuments

     
    pickle_out = open("docIDFDict.pickle","wb") # Saves IDF scores in pickle file, which is optional
    pickle.dump(docIDFDict, pickle_out)
    pickle_out.close()


    print("NumOfDocuments : ", numOfDocuments)
    print("AvgDocLength : ", avgDocLength)



#The following GetBM25Score method will take Query and passage as input and outputs their similarity score based on the term frequency(TF) and IDF values.
def GetBM25Score(Query, Passage, k1=1.5, b=0.75, delimiter=' ') :
    
    global docIDFDict,avgDocLength

    query_words= word_tokenize(Query.lower())
    query_words_stemmed = [porter.stem(word) for word in query_words]
    query_words_stemmed = [w for w in query_words_stemmed if not w in stop_words]
    passage_words = word_tokenize(Passage.lower())
    passage_words_stemmed = [porter.stem(word) for word in passage_words]
    passage_words_stemmed = [w for w in passage_words_stemmed if not w in stop_words]
    passageLen = len(passage_words_stemmed)
    docTF = {}
    for word in set(query_words_stemmed):   #Find Term Frequency of all query unique words
        docTF[word] = passage_words_stemmed.count(word)
    commonWords = set(query_words_stemmed) & set(passage_words_stemmed)
    tmp_score = []
    for word in commonWords :   
        numer = (docTF[word] * (k1+1))   #Numerator part of BM25 Formula
        denom = ((docTF[word]) + k1*(1 - b + b*passageLen/avgDocLength)) #Denominator part of BM25 Formula 
        if(word in docIDFDict) :
            tmp_score.append(docIDFDict[word] * numer / denom)

    score = sum(tmp_score)
    return score

#The following line reads each line from testfile and extracts query, passage and calculates BM25 similarity scores and writes the output in outputfile
def RunBM25OnEvaluationSet(testfile,outputfile):

    lno=0
    tempscores=[]  #This will store scores of 10 query,passage pairs as they belong to same query
    f = open(testfile,"r",encoding="utf-8")
    fw = open(outputfile,"w",encoding="utf-8")
    for line in f:
        tokens = line.strip().lower().split("\t")
        Query = tokens[1]
        Passage = tokens[2]
        score = GetBM25Score(Query,Passage) 
        tempscores.append(score)
        lno+=1
        if(lno%10==0):
            tempscores = [str(s) for s in tempscores]
            scoreString = "\t".join(tempscores)
            qid = tokens[0]
            fw.write(qid+"\t"+scoreString+"\n")
            tempscores=[]
        if(lno%5000==0):
            print(lno)
    print(lno)
    f.close()
    fw.close()


if __name__ == '__main__' :

    inputFileName = "Data.tsv"   # This file should be in the following format : queryid \t query \t passage \t label \t passageid
    testFileName = "eval1_unlabelled.tsv"  # This file should be in the following format : queryid \t query \t passage \t passageid # order of the query
    corpusFileName = "corpus.tsv" 
    outputFileName = "answer.tsv"

    GetCorpus(inputFileName,corpusFileName)    # Gets all the passages(docs) and stores in corpusFile. you can comment this line if corpus file is already generated
    print("Corpus File is created.")
    IDF_Generator(corpusFileName)   # Calculates IDF scores. 
    #RunBM25OnTestData(testFileName,outputFileName)
    print("IDF Dictionary Generated.")
    RunBM25OnEvaluationSet(testFileName,outputFileName)
    print("Submission file created. ")
