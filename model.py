import pandas as pd
import spacy as sc
import nltk
import numpy as np

#TODO 
#---Build the model architecture following the explanation on StackQuests video on decoder-only architecture.

dataset = [["Is wine good? eos"], ["yes! eos"]]
class decoder_only_model:
    """1. BUILDING THE MODEL ARCHITECTURE"""
    def __init__(self, dataset, embeddings=2):
        np.random.seed(42)
        self.dataset = pd.DataFrame(dataset) #Dataset containing Natural langugage sentences.
        self.vocabulary = self.vocab(dataset) #Vocabulary of unique tokens, including punctuations like "?","!" and "."
        self.embedding_weights = np.random.rand(len(self.vocabulary), embeddings) #Each row/list is the weights of a specific word. 
        
    def vocab(self, dataset): 
        vocab = []
        tokenized_dataset = [nltk.word_tokenize(sentence[0].lower()) for sentence in dataset]
        for sentence in tokenized_dataset:
            vocab.extend(sentence)
        vocab = list(set(vocab))
        return vocab
    
    def word_embedding(self, prompt):
        print("placeholder")


    def positional_encoding(self):
        print("placeholder")

    """2. TRAINING THE MODEL """
    def fit(self):
        print("placeholder")


    """3. PREDICTING """
    def predict(self):
        print("placeholder")


    



def run(dataset):
    test = decoder_only_model(dataset)

#run(dataset)
