# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 01:43:59 2019

@author: cesar
"""

#Perceptron de n capas


#Construccion de la red
#Utilizacion de la red
#Ingreso de los parametros

import math, random, numpy as np, matplotlib.pyplot as plt


class Perceptron:
    
    def __init__(self):
        #self.perceptron = []
        
        #Variables to represent parts of the network
        self.neurons = [] #Array to store every value for every neuron
        self.threshold = [] #Weigt of the threshold for every neuron
        self.weights = [[]] # Variable values for the Weights for every neuron
        self.delta = []
        self.alpha = 0.8 #Learning ratio
        
        self.numOfLayers = 0
        self.inputs = 0 #Number of inputs
        self.outputs = 0
        self.expectedOutputs = [] #Number of outputs
        
        self.epochs = 0
        
    
    def build(self):
        #Request to enter the number of layers
        self.numOfLayers = (input ("Add number of hidden layers: ") + 2)
        
        #Defining the size of the arrays for neurons and weights
        self.neurons = [[]] * self.numOfLayers
        self.threshold = [[]] * self.numOfLayers
        self.weights = [[[]]] * (self.numOfLayers - 1)
        
        #Built every layer and every neuron
        for i in range (0, self.numOfLayers):
            #Request to enter the number of Neurons for every layer
            if(i == 0):numOfNeurons = self.inputs = input ("Add number of Inputs: ")
            elif(i == self.numOfLayers - 1):numOfNeurons = self.outputs = input ("Add number of Outputs: ")
            else:numOfNeurons = (input ("Add number of Neurons for layer %d:"  % i))
            #Declare size of the layer
            self.neurons[i] = [] * numOfNeurons
            self.threshold[i] = [] * numOfNeurons
            #Create every layer with the number of neurons given
            for j in range (0, numOfNeurons):
                self.neurons[i].append(0.0)
                self.threshold[i].append(0.5)#random.uniform(-1, 1))

        #Built the weights for every connection
        for i in range(0, len(self.neurons)):
            
            #Starts in 1 to create connections in the actual layer according to next one
            if i > 0:             
                #Calculate and create the connections between layers
                self.weights[i - 1] = [[]] * len(self.neurons[i-1])
                for j in range (0, len(self.neurons[i - 1])):
                    
                    #Calculate and create the connections between neurons
                    self.weights[i - 1][j] = [] * len(self.neurons[i])
                    for k in range (0, len(self.neurons[i])):  
                        #Add random values for every connection
                        #Change this to random numbers
                        self.weights[i - 1][j].append(random.uniform(-1, 1))
        
        self.epochs = input("Enter epochs: ") #Number of times that it will calculate the output changing the weights
    #End_def build
    
    def enterData(self):
        print ("Inputs")
        #Adds the data for the inputs
        for i in range (0, self.inputs):
            self.neurons[0][i] = (input("Enter value for input %d:" % (i + 1)))
        
        #Adds data to expected outputs
        print ("Outputs")
        for i in range (0, self.outputs):
            self.expectedOutputs.append(input("Enter expected value for output %d:" % (i + 1)))
            #print i 
    
    def calculateError(self):
        cycles = 0
        while (cycles < self.epochs):
            
            #Calculate and set values for all neurons in all layers
            for i in range(1,self.numOfLayers):
                for j in range(0, len(self.neurons[i])):
                    actualLayer = []
                    #Calculate value for neuron with bias and weights
                    for k in range(0, len(self.neurons[i-1])):
                        actualLayer.append(self.neurons[i - 1][k] * self.weights[i-1][k][j])
                    #Sum every parameter
                    self.neurons[i][j] = (self.threshold[i][j] + math.fsum(actualLayer))
                    #Add the sigmoid function to the result
                    self.neurons[i][j] = (1 / (1 + np.exp(-self.neurons[i][j])))
            cycles += 1
            
            #Declare the final result
            actualOutput = []
            for i in self.neurons[self.numOfLayers - 1]:
                actualOutput.append(i)
        
           
            
            #Calculate error for every output
            self.delta = [[]] * self.numOfLayers
            self.delta[self.numOfLayers - 1] = [] * self.outputs
            error = []
            
            #Calculate Delta for the last Layer
            for e in range(0, self.outputs):
                
                #Derivative of error for the actual output
                error.append(-(self.expectedOutputs[e] - actualOutput[e]))
                self.delta[self.numOfLayers - 1].append(self.neurons[len(self.neurons) - 1][e] * (1 + self.neurons[len(self.neurons) - 1][e]) * error[e])
                
            #Calculate deltas for every layer 
            for i in range (self.numOfLayers - 2, 0, -1):
                self.delta[i] = [] * len(self.neurons[i])
                #print self.delta
                for j in range(0,len(self.neurons[i])):
                    #print i
                    #calculate an array with the weight and next delta to add to sum
                    for k in range(0, len(self.neurons[i + 1])):
                        temporalSum = []
                        temporalSum.append(self.weights[i][j][k] * self.delta[i + 1][k])
                    self.delta[i].append(self.neurons[i][j] * (1 - self.neurons[i][j]) * math.fsum(temporalSum))
            
            #Now that we have every delta we can continue calculating derivative for the error respect to every weight
            for i in range (0, len(self.weights)):
                for j in range (0, len(self.weights[i])):
                    for k in range(0, len(self.weights[i][j])):
                        
                        #Temporal variable to store the derivative for weights
                        deriv = self.neurons[i][j] * self.delta[i + 1][k]
                        
                        #Now we calculate how much and the direction that the weights has to adjust
                        self.weights[i][j][k] = self.weights[i][j][k] - (self.alpha * deriv)
            
            #We adjust the weight of the bias for every neuron
            for i in range(1, len(self.neurons)):
                for j in range(0, len(self.neurons[i])):
                    
                    #Temporal variable to store the derivative for bias
                    deriv = self.delta[i][j]
                    
                    #Now we calculate how much and the direction that the weights has to adjust
                    self.threshold[i][j] = self.threshold[i][j] - (self.alpha * deriv)
            
        print "Expected outputs", self.expectedOutputs
        print "Actual output", actualOutput
        
        cycles += 1
        

            
if __name__ == "__main__":
    neuralNetwork = Perceptron()
    
    neuralNetwork.build()
    neuralNetwork.enterData()
    neuralNetwork.calculateError()
    
    
    
    