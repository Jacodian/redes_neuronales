#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 06:05:49 2018

@author: jcarraascootarola
"""

import numpy as np
import random
from random import shuffle
from SigmoidNeuron import SigmoidNeuron 
from NeuralNetwork import NeuralNetwork
import matplotlib.pyplot as plt

import time




#datos de la red neuronal
initLearningRate=0.05
numberOfInputs=6
#las dos siguientes contienen a las hidden layers mas la capa de output
numberOfLayers =5
numberOfNeuronsPerLayer=[8,5,5,4,3]

numberOfEpochs= 150

# adquisicion, normalizacion y preparacion de datos
#esta parte solo funciona para estos datos en especifico, en caso de usar otros datos hay que procesarlos 
#de distinta forma para que sean inputs validos para la red neuronal
datosSeparados=[[],[],[]]
datos=[]

outputs=['NO','SL','DH' ]
with open("datos.txt", "r") as data:
    
    for line in data:
        aux=line[:-1].split(" ")
        newLine=[]

        for i in range(len(aux)-1):
            newLine.append(float(aux[i]))
        newLine.append(outputs.index(aux[-1]))
        datos.append(newLine)
        

minimos=[]
maximos=[]
columns= list(map(list, zip(*datos)))
for i in range(len(columns)-1):
    minimos.append(min(columns[i]))
    maximos.append(max(columns[i]))

def normalize(minimo, maximo, value):
    return float((value - minimo)*(1 - 0)) / float((maximo - minimo) + 0)

for line in datos:
    newLine=[]
    for i in range(len(line)- 1):
        newLine.append(normalize(minimos[i],maximos[i],line[i]))
    datosSeparados[line[-1]].append(newLine)
        
inputTraining=[]
outputTraining=[]
inputTest=[]
outputTest=[]



#se separan los datos en training data (2/3) y test data (1/3)
for i in range(len(outputs)):
    for j in range(int(2*len(datosSeparados[i])/3)):
        preOutput=[0,0,0]
        inputTraining.append(datosSeparados[i][j])
        preOutput[i]=1
        outputTraining.append(preOutput)
    
    for j in range(int(2*len(datosSeparados[i])/3),len(datosSeparados[i])):
        preOutput=[0,0,0]
        inputTest.append(datosSeparados[i][j])
        preOutput[i]=1
        outputTest.append(preOutput)  
 
#suffle del set de entrenamiento        
c = list(zip(inputTraining, outputTraining))

random.shuffle(c)

inputTraining, outputTraining = zip(*c)

#manejo de la red neuronal
      
#se crea la red  
nn=NeuralNetwork(initLearningRate, numberOfInputs, numberOfLayers, numberOfNeuronsPerLayer)
#se entrena
print("Entrenando, esto puede tomar un tiempo ...")
start = time.time()
result = nn.epochTrainingPrediction([inputTraining,outputTraining],[inputTest,outputTest],numberOfEpochs)
end = time.time()
print("time elapsed: "+str(end - start))
plt.plot(result[0], result[1])
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.title("Performance de la red neuronal")
plt.show()
        















