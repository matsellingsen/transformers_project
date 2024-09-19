import pandas as pd
import spacy as sc
import nltk
import numpy as np
import math

#TODO 
# IN GENERAL: 1. (DONE) Build the model architecture following the explanation on StackQuests video on decoder-only architecture. (https://www.youtube.com/watch?v=bQ5BoolX9Ag&t=1737s) 
#             2. Train model using backpropagation.
#             
#Specific: 
    # (DONE) Build Decoder-part (Fully conncted layer placed on top of Encoder-part) (Either create new set of weights, or reuse (reversed) embedding_weights ) 
    # Create training scheme/pipeline
    # Create (labeled) dataset

#LAST CHANGE: fully-connected-decoder layer is implemented. Now possible to output a "next word" 
#-----------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------
dataset = [["Is wine good? eos"], ["yes! eos"]]#<-- Should make a function to retrieve a big(ger) dataset at some point.
class decoder_only_model:
    """1. BUILDING THE MODEL ARCHITECTURE"""
    def __init__(self, dataset, embeddings=4):
        np.random.seed(42)#<-- For reproducability

        self.embedding_size = embeddings #NB! This must be an even number. Look at calculation of positional-encoding to see why.
        self.dataset = pd.DataFrame(dataset) #Dataset containing Natural langugage sentences.
        self.vocabulary = self.vocab(dataset) #Vocabulary of unique tokens, including punctuations like "?","!" and "."

        #ENCODINGS (will be assigned values in accordance with prompts/trainingData(?))
        self.current_prompt_embedding = None
        self.current_masked_self_attention = None
        self.current_residual_values = None

        #WEIGHTS
        self.embedding_weights = np.random.rand(len(self.vocabulary), embeddings) #Each row/list is the weights of a specific word. 
        self.key_weights = np.random.rand(embeddings, embeddings)
        self.query_weights = np.random.rand(embeddings, embeddings)
        self.value_weights = np.random.rand(embeddings, embeddings)
#-----------------------------------------------------------------------------------------------------------------------------------------
#   CREATING MODELS VOCABULARY, i.e CREATE A SET OF ALL WORDS THE MODEL KNOWS
    def vocab(self, dataset): #Creates vocabulary based on the dataset 
        vocab = []
        tokenized_dataset = [nltk.word_tokenize(sentence[0].lower()) for sentence in dataset]
        for sentence in tokenized_dataset:
            vocab.extend(sentence)
        vocab = list(set(vocab))
        return vocab
#-----------------------------------------------------------------------------------------------------------------------------------------
#   ENCODER BLOCK (composed of: prompt_embedding, masked_self_attention, residual_connection)
    def prompt_embedding(self, prompt):
      #  prompt = prompt + "eos" #<-- Must always be the last token of a prompt. <-- Wrong
        prompt_tokenized = nltk.word_tokenize(prompt.lower())
        prompt_embedding = [] #<-- A collection of all word-embeddings of the given prompt. Can also be called a sentence-embedding(?)
        for token in prompt_tokenized:
            zeros_array = np.zeros(len(self.vocabulary))
            zeros_array[self.vocabulary.index(token)] = 1
            token_embedding = self.word_embedding(zeros_array)
            token_positional_encoding = self.positional_encoding(self.vocabulary.index(token))
            print(np.add(token_embedding, token_positional_encoding))
            prompt_embedding.append((token, np.add(token_embedding, token_positional_encoding)))#<-- (token_embedding + token_positional_encoding) & token is appended here.
        self.current_prompt_embedding = prompt_embedding

    def word_embedding(self, zeros_array): #<-- Can also be called token-embedding, as it also operates on sumbols like "?"
        token_embedding = np.dot(zeros_array, self.embedding_weights) #<-- calculates dot-product between zeros_array and self.embedding_weights.
        return token_embedding

    def positional_encoding(self, token_index):
        positional_encoding = []
        for i in range(1, int(self.embedding_size/2)+1): #<-- self.embedding_size must always be even for this to make sense.
            positional_encoding.append(math.sin(token_index / i))
            positional_encoding.append(math.cos(token_index / i))
        return positional_encoding
    
    def masked_self_attention(self): #<-- Should be able to stack these(?)
        #As with prompt_embedding(), this should generate an object.
        keys_and_values = []
        masked_self_attention_values = []
        for word_embedding in self.current_prompt_embedding: #Creating keys and values for all tokens. Query is created later.
            #The loops in this for-loop could in theory run in parallell to save time. <-- Possible todo when scaling begins.
            keys = self.key(word_embedding[1])
            values = self.value(word_embedding[1])
            keys_and_values.append([word_embedding, keys, values])  
        
        for word_embedding in self.current_prompt_embedding:
            index = self.current_prompt_embedding.index(word_embedding) #<-- So tokens that come after don't have an effect.
            current_query = self.query(word_embedding[1]) #<-- Query for each token is only used once, so they can all share the same variable.
            key_calculations = [np.dot(current_query, key[1]) for key in keys_and_values[:index+1]]
          #  print("KeyDots: ", key_calculations)
            softmax_values = self.softmax(key_calculations)
           # print("softMax: ", softmax_values)
            masked_self_attention_value = np.dot(softmax_values, [value[2] for value in keys_and_values[:index+1]])
           # print("masked: ", masked_self_attention_value)
            print("-----------------------------------")
            masked_self_attention_values.append(masked_self_attention_value[0])

            #TODO: Calculate masked_self_attention-value for each token in prompt. 
                 # All necessary values for this calculation should already have been made. (Query, key, value)
                 #Find a smart way to retrieve/use key&value 
            #masked_self_attention_value = np.dot(self.key_weights, )
        self.current_masked_self_attention = masked_self_attention_values
        self.current_residual_values = self.residual_values()
        #print("residual: ", self.current_residual_values)
        return("placeholder")
    
    #CALCULATING KVQs
    def query(self, word_embedding):
        return np.dot(word_embedding, self.query_weights)
    def key(self, word_embedding):
        return np.dot(word_embedding, self.key_weights) 
    def value(self, word_embedding):
        return np.dot(word_embedding, self.value_weights)
    
    def softmax(self, l):
        print
        return [np.divide(np.exp(l),sum(np.exp(l)))]

    def residual_values(self):
        #print("prompt_embedding: ", self.current_prompt_embedding)
        #print("masked: ", self.current_masked_self_attention)
        return np.add([prompt[1] for prompt in self.current_prompt_embedding], self.current_masked_self_attention)
#-----------------------------------------------------------------------------------------------------------------------------------------
#   DECODER-COMPONENT
    def fully_connected_ffLayer(self): #Reusing same weights here only transposed.
        current_res_val = self.current_residual_values[len(self.current_residual_values)-1]
        current_word = self.current_prompt_embedding[len(self.current_prompt_embedding)-1][0]
        output_layer = self.softmax(np.dot(current_res_val, self.embedding_weights.transpose()))
        next_word = self.vocabulary[output_layer.index(max(output_layer))]
        return(next_word)
    
#----------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------
    """2. TRAINING THE MODEL """
    def fit(self):
        print("placeholder")
#-----------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------
    """3. PREDICTING """
    def predict(self):
        print("placeholder")



def run(dataset):
    test = decoder_only_model(dataset)
    print(test.vocabulary)
    print(len(test.vocabulary))
    test.prompt_embedding("?")
    print("current prompt_embedding: ", test.current_prompt_embedding)
    test.masked_self_attention()
    test.fully_connected_ffLayer()

run(dataset)

