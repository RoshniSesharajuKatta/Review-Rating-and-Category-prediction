#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

Group Submission: 
Roshni Sesharaju Katta: z5262723 
Shreya Anilkumar: z5269287

Changes made in the student.py
--------------------------------------------------------------------------------

The following paragraph explains the working of each block of the code

Preprocessing():
Preprocessing aimed at removing all the non alpha terms

Stopwords:
The exhaustive list of all possible stop words are provided.

wordVectors' dimension: wordVectors = GloVe(name='6B', dim=200)
Tried with all the provided values (50, 100, 200 or 300). With the value 300, 
the performence of the model was better compared to other values.

convertNetOutput():
For the predictions to be of type LongTensor, taking the values 0 or 1 for the
rating, and 0, 1, 2, 3, or 4 for the business category, Sigmoid and Softmax were used respectively.

class network(tnn.Module):
    def __init__(self)
    here, we have defined, LSTM with linear model and a regulizer tnn.dropout.

    forward()
    The input is transposed after converting it into float, which is passed through LSTM and linear layer.
    This function returns both outputs for rating and category

class loss(tnn.Module):
    def __init__(self):
    
    forward()
    calculates the entropy. 
    Rating loss uses BCEWithLogitsLoss() where this loss combines a Sigmoid layer and the BCELoss in one single class.
    Category loss used Cross Entropy Loss Fn since we have multiple labels.

Optimizer is Adam with lr=0.005. Adam gave us better values than SGD


--------------------------------------------------------------------------------

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables stopWords,
wordVectors, trainValSplit, batchSize, epochs, and optimiser, as well as a basic
tokenise function.  You are encouraged to modify these to improve the
performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.
You may change this variable in the config.py file.

You may only use GloVe 6B word vectors as found in the torchtext package.
"""

import torch
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe
import numpy as np

from config import device

################################################################################
##### The following determines the processing of input data (review text) ######
################################################################################

def tokenise(sample):
    """
    Called before any processing of the text has occurred.
    """

    processed = sample.split()

    return processed

def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """
    for i in range(len(sample)):
        sample[i] = ''.join([j for j in sample[i] if j.isalpha()])

    return sample

def postprocessing(batch, vocab):
    """
    Called after numericalising but before vectorising.
    """

    return batch

stopWords = {"a", "about", "above", "above", "across", "after", "afterwards", "again", "against",
            "all", "almost", "alone", "along", "already", "also","although","always","am","among", 
            "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything",
            "anyway", "anywhere", "are", "around", "as",  "at", "back",
            "be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", 
            "behind", "being", "below", "beside", "besides", "between", "beyond", 
            "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could",
            "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", 
            "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever",
            "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill",
            "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four",
            "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he",
            "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him",
            "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest",
            "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", 
            "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover",
            "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never",
            "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now",
            "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", 
            "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please",
            "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several",
            "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", 
            "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than",
            "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", 
            "therefore", "therein", "thereupon", "these", "they", "thick", "thin", "third", "this", "those",
            "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", 
            "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", 
            "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", 
            "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", 
            "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", 
            "yet", "you", "your", "yours", "yourself", "yourselves", "the"}

wordVectors = GloVe(name='6B', dim=300)

################################################################################
####### The following determines the processing of label data (ratings) ########
################################################################################

def convertNetOutput(ratingOutput, categoryOutput):
    """
    Your model will be assessed on the predictions it makes, which must be in
    the same format as the dataset ratings and business categories.  The
    predictions must be of type LongTensor, taking the values 0 or 1 for the
    rating, and 0, 1, 2, 3, or 4 for the business category.  If your network
    outputs a different representation convert the output here.
    """
    #Conversion using Sigmoid funtion
    sigmoid = tnn.Sigmoid()
    output_rating=torch.round(sigmoid(ratingOutput)).long()
    #Conversion using Softmax function
    softmax = tnn.Softmax(dim=1)
    output_category=torch.argmax(softmax(categoryOutput), 1)
    return output_rating,output_category

################################################################################
###################### The following determines the model ######################
################################################################################

class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network will be a
    batch of reviews (in word vector form).  As reviews will have different
    numbers of words in them, padding has been added to the end of the reviews
    so we can form a batch of reviews of equal length.  Your forward method
    should return an output for both the rating and the business category.
    """

    def __init__(self):
        super(network, self).__init__()
        #the long short term memory layers
        self.lstm1 = tnn.LSTM(300,128,num_layers=2,bidirectional=True,dropout=0.5)
        self.lstm2 = tnn.LSTM(300,128,num_layers=2,bidirectional=True,dropout=0.5)
        #Linear layers
        self.lin1 = tnn.Linear(256, 1)
        self.lin2 = tnn.Linear(256, 5)
        #Dropout regularisation
        self.dropout_1 = tnn.Dropout(0.5)
        self.dropout_2 = tnn.Dropout(0.5)

    def forward(self, input, length):
        input = input.float()
        input = torch.transpose(input, 0, 1)
        #For one layer
        dropout_1 = self.dropout_1(input)
        output_1, (hidden1, _) = self.lstm1(dropout_1)
        hidden1 = self.dropout_1(torch.cat((hidden1[-1,:,:], hidden1[-2,:,:]), dim=1))
        
        #For another layer
        dropout_2 = self.dropout_2(input)
        output_2, (hidden2, _) = self.lstm2(dropout_2)
        hidden2 = self.dropout_2(torch.cat((hidden2[-1,:,:], hidden2[-2,:,:]), dim=1))

        return self.lin1(hidden1), self.lin2(hidden2)

class loss(tnn.Module):
    """
    Class for creating the loss function.  The labels and outputs from your
    network will be passed to the forward method during training.
    """

    def __init__(self):
        super(loss, self).__init__()

    def forward(self, ratingOutput, categoryOutput, ratingTarget, categoryTarget):
        #rating prediction loss
        rat_Loss = tnn.BCEWithLogitsLoss()
        rat_squeeze=ratingOutput.squeeze(1)
        #category prediction loss
        cat_Loss = tnn.CrossEntropyLoss()
        return rat_Loss(rat_squeeze, ratingTarget.float()) + cat_Loss(categoryOutput, categoryTarget)


net = network()
lossFunc = loss()

################################################################################
################## The following determines training options ###################
################################################################################

trainValSplit = 0.8
batchSize = 32
epochs = 10
optimiser = toptim.Adam(net.parameters(), lr=0.005)
