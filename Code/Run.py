'''Run by Lucia Gordon'''
import numpy as np
from random import sample
from DataEngineering import DataEngineering
from CNN import CNN

Thermal = DataEngineering(orthomosaicPath='Data/Firestorm3/Firestorm3Thermal.tif', csvPath='Data/Firestorm3/MiddenLocations.csv', imageryType='thermal')
trainingImages, trainingLabels, trainingData = [Thermal.trainingImgs, Thermal.trainingLabels, Thermal.trainingData]
middenIndices = np.where(trainingLabels==1)[0]
for index in sample(middenIndices.tolist(),10): # look at 10 random images with middens
    print(index)
    Thermal.plotThermalImage(trainingImages[index])

#ThermalPassive = CNN(trainingData, batchSize=1, epochs=1, learningRate=0.0001, learningMode='passive')