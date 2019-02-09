from __future__ import print_function
import numpy as np
import sys
import os
import cntk as C
from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs, INFINITELY_REPEAT, FULL_DATA_SWEEP
from sklearn.metrics import precision_recall_fscore_support
from cntk.device import try_set_default_device, gpu
from sklearn.exceptions import UndefinedMetricWarning
import warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning) 

try_set_default_device(gpu(0))


#Initialize Global variables
validation_query_vectors = []
validation_passage_vectors = []
validation_labels = []   
q_max_words=12
p_max_words=50
emb_dim=50
'''
## The following LoadValidationSet method reads ctf format validation file and creates query, passage feature vectors and also copies labels for each pair.
## the created vectors will be useful to find metrics on validation set after training each epoch which will be useful to decide the best model 
def LoadValidationSet(validationfile):
    f = open(validationfile,'r',encoding="utf-8")
    for line in f:
        tokens = line.strip().split("|")  
        #tokens[0] will be empty token since the line is starting with |
        x1 = tokens[1].replace("qfeatures","").strip() #Query Features
        x2 = tokens[2].replace("pfeatures","").strip() # Passage Features
        y = tokens[3].replace("labels","").strip() # labels
        x1 = [float(v) for v in x1.split()]
        x2 = [float(v) for v in x2.split()]
        y = [int(w) for w in y.split()]        
        y = y[1] # label will be at index 1, i.e. if y = "1 0" then label=0 else if y="0 1" then label=1

        validation_query_vectors.append(x1)
        validation_passage_vectors.append(x2)
        validation_labels.append(y)

        #print("1")
    
    print("Validation Vectors are created")'''

#The following method defines a CNN network which has series of convolution and max pooling steps on query features and passage features and then a merge step and it follows a fully connected layer
def cnn_network(queryfeatures, passagefeatures, num_classes):
    with C.layers.default_options(activation=C.ops.relu, pad=False):
        #Query expression
        conv1_A1 = C.layers.Convolution2D((3,3),4,pad=True,strides=1,activation=C.ops.relu,name='conv1_A1')(queryfeatures) #input : 12*50 #output : 4*12*50
        conv1_A2 = C.layers.Convolution2D((3,3),4,pad=True,strides=1,activation=C.ops.relu,name='conv1_A2')(conv1_A1) #input : 4*12*50 #output : 4*12*50
        conv1_A3 = C.layers.Convolution2D((3,3),4,pad=True,strides=1,activation=C.ops.relu,name='conv1_A3')(conv1_A2) #input : 4*12*50 #output : 4*12*50
        pool1_A = C.layers.MaxPooling((2,2),(2,2),name='pool1_A')(conv1_A3)  #output : 4*6*25
        
        conv1_B1 = C.layers.Convolution2D((3,3),8,pad=True,strides=1,activation=C.ops.relu,name='conv1_B1')(pool1_A) #output : 8*6*25
        conv1_B2 = C.layers.Convolution2D((3,3),8,pad=True,strides=1,activation=C.ops.relu,name='conv1_B2')(conv1_B1) #output : 8*6*25
        conv1_B3 = C.layers.Convolution2D((3,3),8,pad=True,strides=1,activation=C.ops.relu,name='conv1_B3')(conv1_B2) #output : 8*6*25
        pool1_B = C.layers.MaxPooling((2,5),(2,5),name='pool1_B')(conv1_B3)  #output : 8*3*5
        
        conv1_C1 = C.layers.Convolution2D((1,1),2,pad=False,strides=1,activation=C.ops.relu,name='conv1_C1')(pool1_B) #input : 8*3*5 #output : 2*3*5
        conv1_C2 = C.layers.Convolution2D((1,1),2,pad=False,strides=1,activation=C.ops.relu,name='conv1_C2')(pool1_B) #input : 8*3*5 #output : 2*3*5
        pool1_C = C.layers.MaxPooling((3,3),pad=True,strides=(1,1),name='pool1_C')(pool1_B) #output 8*3*5
        conv1_C3 = C.layers.Convolution2D((1,1),4,pad=False,strides=1,activation=C.ops.relu,name='conv1_C3')(pool1_B) #input : 8*3*5 #output : 4*3*5
        conv1_C4 = C.layers.Convolution2D((3,3),6,pad=True,strides=1,activation=C.ops.relu,name='conv1_C4')(conv1_C1) #input : 2*3*5 #output : 6*3*5
        conv1_C5 = C.layers.Convolution2D((3,3),6,pad=True,strides=1,activation=C.ops.relu,name='conv1_C5')(conv1_C2) #input : 2*3*5 #output : 6*3*5
        conv1_C6 = C.layers.Convolution2D((1,1),2,pad=False,strides=1,activation=C.ops.relu,name='conv1_C6')(pool1_C) #input : 8*3*5 #output : 2*3*5
        concat1_C = C.splice(conv1_C3, conv1_C4, conv1_C5, conv1_C6, axis = 0) #output 18*3*5

        conv1_D1 = C.layers.Convolution2D((1,1),2,pad=False,strides=1,activation=C.ops.relu,name='conv1_D1')(concat1_C) #input : 18*3*5 #output : 2*3*5
        conv1_D2 = C.layers.Convolution2D((1,1),2,pad=False,strides=1,activation=C.ops.relu,name='conv1_D2')(concat1_C) #input : 18*3*5 #output : 2*3*5
        pool1_D = C.layers.MaxPooling((3,3),pad=True,strides=(1,1),name='pool1_D')(concat1_C) #output 18*3*5
        conv1_D3 = C.layers.Convolution2D((1,1),4,pad=False,strides=1,activation=C.ops.relu,name='conv1_D3')(concat1_C) #input : 18*3*5 #output : 4*3*5
        conv1_D4 = C.layers.Convolution2D((3,3),6,pad=True,strides=1,activation=C.ops.relu,name='conv1_D4')(conv1_D1) #input : 2*3*5 #output : 6*3*5
        conv1_D5 = C.layers.Convolution2D((3,3),6,pad=True,strides=1,activation=C.ops.relu,name='conv1_D5')(conv1_D2) #input : 2*3*5 #output : 6*3*5
        conv1_D6 = C.layers.Convolution2D((1,1),2,pad=False,strides=1,activation=C.ops.relu,name='conv1_D6')(pool1_D) #input : 18*3*5 #output : 2*3*5
        concat1_D = C.splice(conv1_D3, conv1_D4, conv1_D5, conv1_D6, axis = 0) #output 18*3*5

        pool1_E = C.layers.AveragePooling((3,5),(3,5),name='pool1_E')(concat1_D) #output 18*1*1
        dropout1_E = C.layers.Dropout(dropout_rate=.2, name='dropout1_E')(pool1_E) #output 18*1*1
        dense1_E = C.layers.Dense(6,activation=C.tanh,name='dense1_E')(dropout1_E)  # output : 6

        #Passage expression
        conv2_A1 = C.layers.Convolution2D((3,3),4,pad=True,strides=1,activation=C.ops.relu,name='conv2_A1')(passagefeatures) #input : 50*50 #output : 4*50*50
        conv2_A2 = C.layers.Convolution2D((3,3),4,pad=True,strides=1,activation=C.ops.relu,name='conv2_A2')(conv2_A1) #input : 4*50*50 #output : 4*50*50
        conv2_A3 = C.layers.Convolution2D((3,3),4,pad=True,strides=1,activation=C.ops.relu,name='conv2_A3')(conv2_A2) #input : 4*12*50 #output : 4*50*50
        pool2_A = C.layers.MaxPooling((2,2),(2,2),name='pool2__A')(conv2_A3)  #output : 4*25*25
        
        conv2_B1 = C.layers.Convolution2D((3,3),8,pad=True,strides=1,activation=C.ops.relu,name='conv2_B1')(pool2_A) #output : 8*25*25
        conv2_B2 = C.layers.Convolution2D((3,3),8,pad=True,strides=1,activation=C.ops.relu,name='conv2_B2')(conv2_B1) #output : 8*25*25
        conv2_B3 = C.layers.Convolution2D((3,3),8,pad=True,strides=1,activation=C.ops.relu,name='conv2_B3')(conv2_B2) #output : 8*25*25
        pool2_B = C.layers.MaxPooling((5,5),(5,5),name='pool2_B')(conv2_B3)  #output : 8*5*5
        
        conv2_C1 = C.layers.Convolution2D((1,1),2,pad=False,strides=1,activation=C.ops.relu,name='conv2_C1')(pool2_B) #input : 8*5*5 #output : 2*5*5
        conv2_C2 = C.layers.Convolution2D((1,1),2,pad=False,strides=1,activation=C.ops.relu,name='conv2_C2')(pool2_B) #input : 8*5*5 #output : 2*5*5
        pool2_C = C.layers.MaxPooling((3,3),pad=True,strides=(1,1),name='pool2_C')(pool2_B) #output 8*5*5
        conv2_C3 = C.layers.Convolution2D((1,1),4,pad=False,strides=1,activation=C.ops.relu,name='conv2_C3')(pool2_B) #input : 8*5*5 #output : 4*5*5
        conv2_C4 = C.layers.Convolution2D((3,3),6,pad=True,strides=1,activation=C.ops.relu,name='conv2_C4')(conv2_C1) #input : 2*5*5 #output : 6*5*5
        conv2_C5 = C.layers.Convolution2D((3,3),6,pad=True,strides=1,activation=C.ops.relu,name='conv2_C5')(conv2_C2) #input : 2*5*5 #output : 6*5*5
        conv2_C6 = C.layers.Convolution2D((1,1),2,pad=False,strides=1,activation=C.ops.relu,name='conv2_C6')(pool2_C) #input : 8*5*5 #output : 2*5*5
        concat2_C = C.splice(conv2_C3, conv2_C4, conv2_C5, conv2_C6, axis = 0) #output 18*5*5

        conv2_D1 = C.layers.Convolution2D((1,1),2,pad=False,strides=1,activation=C.ops.relu,name='conv2_D1')(concat2_C) #input : 18*5*5 #output : 2*5*5
        conv2_D2 = C.layers.Convolution2D((1,1),2,pad=False,strides=1,activation=C.ops.relu,name='conv2_D2')(concat2_C) #input : 18*5*5 #output : 2*5*5
        pool2_D = C.layers.MaxPooling((3,3),pad=True,strides=(1,1),name='pool2_D')(concat2_C) #output 18*5*5
        conv2_D3 = C.layers.Convolution2D((1,1),4,pad=False,strides=1,activation=C.ops.relu,name='conv2_D3')(concat2_C) #input : 18*5*5 #output : 4*5*5
        conv2_D4 = C.layers.Convolution2D((3,3),6,pad=True,strides=1,activation=C.ops.relu,name='conv2_D4')(conv2_D1) #input : 2*5*5 #output : 6*5*5
        conv2_D5 = C.layers.Convolution2D((3,3),6,pad=True,strides=1,activation=C.ops.relu,name='conv2_D5')(conv2_D2) #input : 2*5*5 #output : 6*5*5
        conv2_D6 = C.layers.Convolution2D((1,1),2,pad=False,strides=1,activation=C.ops.relu,name='conv2_D6')(pool2_D) #input : 18*5*5 #output : 2*5*5
        concat2_D = C.splice(conv2_D3, conv2_D4, conv2_D5, conv2_D6, axis = 0) #output 18*5*5

        pool2_E = C.layers.AveragePooling((5,5),(5,5),name='pool2_E')(concat2_D) #output 18*1*1
        dropout2_E = C.layers.Dropout(dropout_rate=.2, name='dropout2_E')(pool2_E) #output 18*1*1
        dense2_E = C.layers.Dense(6,activation=C.tanh,name='dense2_E')(dropout2_E)  # output : 6

        mergeQP     = C.element_times(dense1_E,dense2_E) # output : 6

        model   = C.layers.Dense(num_classes, activation=C.softmax,name="overall")(mergeQP) #outupt : 2
        

    return model

def create_reader(path, is_training, query_total_dim, passage_total_dim, label_total_dim):
    return MinibatchSource(CTFDeserializer(path, StreamDefs( queryfeatures = StreamDef(field='qfeatures', shape=query_total_dim,is_sparse=False), 
                                                            passagefeatures = StreamDef(field='pfeatures', shape=passage_total_dim,is_sparse=False), 
                                                            labels   = StreamDef(field='labels', shape=label_total_dim,is_sparse=False)
                                                            )), 
                           randomize=is_training, max_sweeps = INFINITELY_REPEAT if is_training else FULL_DATA_SWEEP)

def TrainAndValidate(trainfile):

    #*****Hyper-Parameters******
    q_max_words= 12
    p_max_words = 50
    emb_dim = 50
    num_classes = 2
    minibatch_size = 13100
    epoch_size = 5241880 #No.of samples in training set
    total_epochs = 145 #Total number of epochs to run
    query_total_dim = q_max_words*emb_dim
    label_total_dim = num_classes
    passage_total_dim = p_max_words*emb_dim


    #****** Create placeholders for reading Training Data  ***********
    query_input_var =  C.ops.input_variable((1,q_max_words,emb_dim),np.float32,is_sparse=False)
    passage_input_var =  C.ops.input_variable((1,p_max_words,emb_dim),np.float32,is_sparse=False)
    output_var = C.input_variable(num_classes,np.float32,is_sparse = False)
    train_reader = create_reader(trainfile, True, query_total_dim, passage_total_dim, label_total_dim)
    input_map = { query_input_var : train_reader.streams.queryfeatures, passage_input_var:train_reader.streams.passagefeatures, output_var : train_reader.streams.labels}

    # ********* Model configuration *******
    model_output = cnn_network(query_input_var, passage_input_var, num_classes)
    model_output.restore('google_54.dnn')
    loss = C.binary_cross_entropy(model_output, output_var)
    pe = C.classification_error(model_output, output_var)
    lr_per_minibatch = C.learning_rate_schedule(0.03, C.UnitType.minibatch)   
    learner = C.adam(model_output.parameters, lr=lr_per_minibatch, momentum=C.learners.momentum_schedule(0.9,minibatch_size))
    progress_printer = C.logging.ProgressPrinter(tag='Training', num_epochs=total_epochs)

    #************Create Trainer with model_output object, learner and loss parameters*************  
    trainer = C.Trainer(model_output, (loss, pe), learner, progress_printer)
    C.logging.log_number_of_parameters(model_output) ; print()

    # **** Train the model in batchwise mode *****
    for epoch in range(total_epochs):       # loop over epochs
        print("Epoch : ",epoch)
        sample_count = 0
        while sample_count < epoch_size:  # loop over minibatches in the epoch
            data = train_reader.next_minibatch(min(minibatch_size, epoch_size - sample_count), input_map=input_map) # fetch minibatch.
            trainer.train_minibatch(data)        # training step
            sample_count += data[output_var].num_samples   # count samples processed so far

        trainer.summarize_training_progress()
                
        model_output.save("google_{}.dnn".format(epoch+55)) # Save the model for every epoch
    return model_output
        #*** Find metrics on validation set after every epoch ******#  (Note : you can skip doing this for every epoch instead to optimize the time, do it after every k epochs)
'''        predicted_labels=[]
        for i in range(len(validation_query_vectors)):
            queryVec   = np.array(validation_query_vectors[i],dtype="float32").reshape(1,q_max_words,emb_dim)
            passageVec = np.array(validation_passage_vectors[i],dtype="float32").reshape(1,p_max_words,emb_dim)
            scores = model_output(queryVec,passageVec)[0]   # do forward-prop on model to get score  
            predictLabel = 1 if scores[1]>=scores[0] else 0
            predicted_labels.append(predictLabel) 
        metrics = precision_recall_fscore_support(np.array(validation_labels), np.array(predicted_labels), average='binary')
        print("precision : "+str(metrics[0])+" recall : "+str(metrics[1])+" f1 : "+str(metrics[2])+"\n")'''


## The following GetPredictionOnEvalSet method reads all query passage pair vectors from CTF file and does forward prop with trained model to get similarity score
## after getting scores for all the pairs, the output will be written into submission file. 
def GetPredictionOnEvalSet(model,testfile,submissionfile):
    global q_max_words,p_max_words,emb_dim

    f = open(testfile,'r',encoding="utf-8")
    all_scores={} # Dictionary with key = query_id and value = array of scores for respective passages
    for line in f:
        tokens = line.strip().split("|")  
        #tokens[0] will be empty token since the line is starting with |
        x1 = tokens[1].replace("qfeatures","").strip() #Query Features
        x2 = tokens[2].replace("pfeatures","").strip() # Passage Features
        query_id = tokens[3].replace("qid","").strip() # Query_id
        x1 = [float(v) for v in x1.split()]
        x2 = [float(v) for v in x2.split()]    
        queryVec   = np.array(x1,dtype="float32").reshape(1,q_max_words,emb_dim)
        passageVec = np.array(x2,dtype="float32").reshape(1,p_max_words,emb_dim)
        score = model(queryVec,passageVec)[0][1] # do forward-prop on model to get score
        if(query_id in all_scores):
            all_scores[query_id].append(score)
        else:
            all_scores[query_id] = [score]
        print(i)
    fw = open(submissionfile,"w",encoding="utf-8")
    for query_id in all_scores:
        scores = all_scores[query_id]
        scores_str = [str(sc) for sc in scores] # convert all scores to string values
        scores_str = "\t".join(scores_str) # join all scores in list to make it one string with  tab delimiter.  
        fw.write(query_id+"\t"+scores_str+"\n")
        print(j)
    fw.close()

    
if __name__ == "__main__":
    trainSetFileName = "TrainData.ctf"
    validationSetFileName = "ValidationData.ctf"
    testSetFileName = "EvaluationData.ctf"
    submissionFileName = "answer.tsv"
    #LoadValidationSet(validationSetFileName)    #Load Validation Query, Passage Vectors from Validation CTF File
    model = TrainAndValidate(trainSetFileName) # Training and validation methods    
    GetPredictionOnEvalSet(model,testSetFileName,submissionFileName) # Get Predictions on Evaluation Set


    
