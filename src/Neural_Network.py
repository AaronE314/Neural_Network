import numpy as np
from scipy.special import expit

class NeuralNetwork:
    
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningrate):
        
        self.innodes = inputNodes
        self.hnodes = hiddenNodes
        self.onodes = outputNodes
        
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.innodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        
        self.lr = learningrate
        
        self.act_func = lambda x: expit(x)
        
#     def act_func(self, x):
#         return 1 / (np.exp(-x) + 1)
    
    def train(self, inputs_list, targets_list):
        
        #Convert to numpy matrix
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        
        #Multiply the weights of the hidden layer by the inputs
        hIn = np.dot(self.wih,inputs)
        #apply the activation function
        hOut = self.act_func(hIn)
        
        #Multiply the weights of the output layer by the hidden layer
        finalIn = np.dot(self.who, hOut)
        #apply the activation function
        finalOut = self.act_func(finalIn)
        
        
        errorO = targets - finalOut
        
        errorH = np.dot(self.who.T, errorO)
        
        self.who += self.lr * np.dot(errorO * finalOut * (1.0-finalOut), np.transpose(hOut))
        
        self.wih += self.lr * np.dot(errorH * hOut * (1.0-hOut), np.transpose(inputs))
        
    
    def query(self, inputs_list):
        
        #Convert to numpy matrix
        inputs = np.array(inputs_list, ndmin=2).T
        
        #Multiply the weights of the hidden layer by the inputs
        hIn = np.dot(self.wih,inputs)
        
        #apply the activation function
        hOut = self.act_func(hIn)
        
        #Multiply the weights of the output layer by the hidden layer
        finalIn = np.dot(self.who, hOut)
        
        #apply the activation function
        finalOut = self.act_func(finalIn)
        
        return finalOut
        
        
        
        
        


