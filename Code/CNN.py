''' CNN by Lucia Gordon '''
from numpy import amax, amin, append, array, around, where, take, min, load
from pandas import read_csv
from random import sample
from torch import device, cuda, nn, sum, optim, max, FloatTensor, no_grad, from_numpy
from torchvision.models import vgg16
from torch.utils.data import random_split, DataLoader
from torchsummary import summary
from math import ceil

class CNN:
    def __init__(self, images, labels, batchSize=8, epochs=30, learningRate=0.0001, learningMode='passive'):
        self.model = None
        self.images = load(images)
        self.labels = load(labels)
        self.data = sample(list(zip(self.images,self.labels)),100)
        #self.data = list(zip(self.images,self.labels))
        self.images, self.labels = zip(*self.data)
        self.dataLength = len(self.images)
        self.batchSize = batchSize
        self.epochs = epochs
        self.learningRate = learningRate
        self.learningMode = learningMode # passive or active
        self.trainingDataLength = 0
        self.testDataLength = 0
        self.trainingData = None
        self.testData = None
        self.trainingLoader = None
        self.testLoader = None
        self.classWeights = []
        self.accuracy = 0
        self.class0Accuracy = 0
        self.class1Accuracy = 0

        self.useVGG16()
        self.preprocessData()
        self.computeClassWeights()

        if self.learningMode == 'passive':
            self.passiveTrain(self.trainingLoader)
        else:
            self.activeTrain()
        self.test()

    def useVGG16(self):
        myDevice = device('cuda' if cuda.is_available() else 'cpu')
        self.model = vgg16(weights='VGG16_Weights.DEFAULT').to(myDevice) # imports a pretrained vgg16 CNN
        lastLayerIndex = len(self.model.classifier)-1
        oldFinalLayer = self.model.classifier.__getitem__(lastLayerIndex)
        newFinalLayers = nn.Sequential(nn.Linear(in_features=oldFinalLayer.in_features,out_features=2,bias=True),nn.Softmax(dim=1)) # changes the output of the last hidden layer to be binary and adds softmax
        self.model.classifier.__setitem__(lastLayerIndex,newFinalLayers)

    def preprocessData(self):
        transformedImages = []

        for image in self.images:
            next = append([image], [image], axis=0) # (2,40,40)
            image = append(next, [image], axis=0) # (3,40,40)
            print(amin(image))
            image = image/amax(image) # rescale pixel values to between 0 and 1
            image = from_numpy((image-0.5)/0.5).float() # rescale pixel values to between -1 and 1
            transformedImages.append(image)
        
        self.images = transformedImages
        data = list(zip(self.images,self.labels))
        self.trainingDataLength = around(0.8*self.dataLength).astype(int) # use 80% of the data for training
        self.testDataLength = self.dataLength - self.trainingDataLength # use 20% of the data for testing
        self.trainingData, self.testData = random_split(data,[self.trainingDataLength,self.testDataLength]) # splits the data into training and test sets
        self.trainingLoader = DataLoader(self.trainingData, batch_size=self.batchSize, shuffle=True, num_workers=1) # batches and shuffles the training data
        self.testLoader = DataLoader(self.testData, batch_size=self.batchSize, shuffle=True, num_workers=1) # batches and shuffles the test data

    def computeClassWeights(self):
        numClass1 = 0

        for _,batch in enumerate(self.trainingLoader):
            numClass1 += batch[1].sum().item() # sum of labels in batch
        
        self.classWeights = [numClass1/self.trainingDataLength,1 - numClass1/self.trainingDataLength] # the weights are proportional to the number of points in the other class and sum to 1
    
    def passiveTrain(self, trainingLoader):
        criterion = nn.CrossEntropyLoss(weight=FloatTensor(self.classWeights))
        optimizer = optim.Adam(self.model.parameters(),lr=self.learningRate)
        interval = ceil(len(trainingLoader)/self.batchSize/10)

        for epoch in range(self.epochs):  # loop over the dataset multiple times
            runningLoss = 0.0

            for i, data in enumerate(trainingLoader,0): # data is a list of [inputs,labels]
                inputs, labels = data
                optimizer.zero_grad() # zero the parameter gradients
                outputs = self.model(inputs) # forward pass
                loss = criterion(outputs,labels)
                loss.backward() # backward pass
                optimizer.step() # optimization
                runningLoss += loss.item()

                if i % interval == interval-1:
                    print(f'Epoch={epoch+1}, Images {self.batchSize*(i+1-interval)}-{self.batchSize*(i+1)}, Loss={runningLoss/interval:.3f}') # average loss per batch
                    runningLoss = 0.0
        
        print('Finished Training')
    
    def activeTrain(self):
        unlabeledData = self.trainingData.copy()
        unlabeledImages = unlabeledData.T[0] # all the training images start out unlabeled
        maxPixelVals = array([amax(unlabeledImages[i]) for i in range(len(unlabeledImages))]) # array of maximum pixel value in each image
        labelingBudget = len(self.trainingData) # number of images we are willing to provide labels for
        e = 0.01

        def brightestIndices(threshold):
            return sorted(range(len(maxPixelVals)), key=lambda sub:maxPixelVals[sub])[-threshold*len(maxPixelVals):]
        

        #newTrainingImageIndices = sample(brightestIndices(99).tolist(), self.batchSize) # randomly picking a batch among the images with the highest max pixel values
        newTrainingImageIndices = sorted(range(len(maxPixelVals)), key=lambda sub:maxPixelVals[sub])[-self.batchSize:]
        trainingLoader = take(unlabeledData, newTrainingImageIndices) # select the images corresponding to the above indices

        def removeFromUnlabeledImgs():
            for index in sorted(newTrainingImageIndices, reverse=True):
                del unlabeledData[index]
                del unlabeledImages[index] # remove image that will be used for training from the unlabeled set
                del maxPixelVals[index] # remove index that will be used for training from the list of max pixel values

        removeFromUnlabeledImgs()

        while labelingBudget > 0:
            self.passiveTrain(trainingLoader)
            #testImageIndices = brightestIndices(99-e).tolist()
            testImageIndices = sorted(range(len(maxPixelVals)), key=lambda sub:maxPixelVals[sub])[-5*self.batchSize:]
            with no_grad():
                sigmoidOutput = self.model(take(unlabeledImages,testImageIndices)).data
            newTrainingImageIndices = sorted(range(len(sigmoidOutput)), key=lambda sub:sigmoidOutput[sub])[-self.batchSize:]
            trainingLoader = append(trainingLoader,take(unlabeledData,newTrainingImageIndices))
            removeFromUnlabeledImgs()
            labelingBudget -= self.batchSize
            e = min([2*e,50])

    def test(self):
        correct = 0
        class0Correct = 0
        class1Correct = 0
        total = 0
        class0Total = 0
        class1Total = 0

        with no_grad(): # since we're not training, we don't need to calculate the gradients for our outputs
            for i, data in enumerate(self.testLoader):
                images, labels = data
                outputs = self.model(images)
                _, predicted = max(outputs.data, 1) # predicted is a vector with batchSize elements corresponding to the index of the most likely class of each image in the batch
                total += labels.size(0) # number of images in the batch
                correct += (predicted == labels).sum().item()
                class0Indices = where(labels == 0)[0] # indices of no midden images
                class0Total += len(class0Indices)
                class0Correct += (take(predicted,class0Indices) == take(labels,class0Indices)).sum().item()
                class1Indices = where(labels == 1)[0] # indices of midden images
                class1Total += len(class1Indices)
                class1Correct += (take(predicted,class1Indices) == take(labels,class1Indices)).sum().item()

        self.accuracy = 100*correct/total
        self.class0Accuracy = 100*class0Correct/class0Total
        self.class1Accuracy = 100*class1Correct/class1Total
        print(f'Accuracy of the neural network on the {self.batchSize*(i+1)} test images = {self.accuracy}%')
        print(f'Accuracy on images without middens = {self.class0Accuracy}%')
        print(f'Accuracy on images with middens = {self.class1Accuracy}%')

ThermalCNN = CNN('Data/TrainingImagesThermal.npy', 'Data/TrainingLabelsThermal.npy', batchSize=8, epochs=1, learningRate=0.0001, learningMode='passive')