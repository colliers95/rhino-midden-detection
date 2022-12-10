''' CNN by Lucia Gordon '''
from numpy import amax, amin, append, array, around, where, take, load, delete, mean, std, save, arange
from random import sample
from torch import device, cuda, nn, sum, optim, min, max, FloatTensor, no_grad, from_numpy, empty, Tensor, cat, float64, tensor
from torchvision.models import vgg16
from torchvision.transforms import transforms
from torch.utils.data import random_split, DataLoader
from torchsummary import summary
from math import ceil

class CNN:
    def __init__(self, batchSize=16, epochs=30, learningRate=0.001, learningMode='passive', trial='1'):
        self.model = None
        self.transform = transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        #self.images = load(input('Please enter the path to your images.\n'))
        self.images = load('Phase2/Data/Thermal/Images.npy').astype(float)
        print('Images shape:',self.images.shape)
        #self.labels = load(input('Please enter the path to your labels.\n'))
        self.labels = load('Phase2/Data/Thermal/Labels.npy').astype(float)
        #self.labels = load(input('Please enter the path to the raw images (single thermal band).\n'))
        self.rawImages = load('Phase2/Data/Thermal/RawImages.npy')
        #range(0,len(self.images)),
        self.indices = arange(len(self.images))
        self.data = list(zip(self.indices,self.images,self.labels))
        self.dataLength = len(self.data)
        self.batchSize = batchSize
        self.epochs = epochs
        self.learningRate = learningRate
        self.learningMode = learningMode # passive or active
        self.trial = trial
        self.trainingDataLength = 0
        self.testDataLength = 0
        self.trainingData = None
        self.testData = None
        self.trainingLoader = None
        self.testLoader = None
        self.testLabels = []
        self.trainingIndices = []
        self.classWeights = []
        self.accuracy = 0
        self.class0Accuracy = 0
        self.class1Accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f1 = 0

        self.useVGG16()
        self.preprocessData()
        self.computeClassWeights()

        if self.learningMode == 'passive':
            self.passiveTrain(self.trainingLoader, self.epochs)
        
        elif self.learningMode == 'active':
            self.activeTrain()

        self.test()

    def useVGG16(self):
        myDevice = device('cuda' if cuda.is_available() else 'cpu')
        self.model = vgg16(weights='VGG16_Weights.DEFAULT').to(myDevice) # imports a pretrained vgg16 CNN

        for parameter in self.model.parameters(): # freeze all parameters
            parameter.requires_grad = False
        
        self.model.classifier = nn.Sequential(nn.Linear(in_features=25088, out_features=512, bias=True),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(p=0.5, inplace=False),
                                 nn.Linear(in_features=512, out_features=256, bias=True),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(p=0.5, inplace=False),
                                 nn.Linear(in_features=256, out_features=128, bias=True),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(p=0.5, inplace=False),
                                 nn.Linear(in_features=128, out_features=1, bias=True),
                                 nn.Sigmoid()) # unfreeze classifier parameters
                
    def preprocessData(self):        
        self.trainingDataLength = around(0.8*self.dataLength).astype(int) # use 80% of the data for training
        self.testDataLength = self.dataLength - self.trainingDataLength # use 20% of the data for testing
        self.trainingData, self.testData = random_split(self.data,[self.trainingDataLength,self.testDataLength]) # splits the data into training and test sets       
        self.trainingLoader = DataLoader(self.trainingData, batch_size=self.batchSize, shuffle=True, num_workers=1) # batches and shuffles the training data
        self.testLoader = DataLoader(self.testData, batch_size=self.batchSize, shuffle=True, num_workers=1) # batches and shuffles the test data
        
        for _, point in enumerate(self.trainingData):
            self.trainingIndices.append(point[0])

    def computeClassWeights(self):
        numClass1 = 0

        for _,batch in enumerate(self.trainingLoader):
            numClass1 += batch[2].sum().item() # sum of labels in batch
        
        self.classWeights = [numClass1/self.trainingDataLength, 1 - numClass1/self.trainingDataLength] # the weights are proportional to the number of points in the other class and sum to 1
        #print(self.classWeights)
        numClass1Test = 0

        for _,batch in enumerate(self.testLoader):
            numClass1Test += batch[2].sum().item() # sum of labels in batch
        
        #print([numClass1Test/self.testDataLength, 1 - numClass1Test/self.testDataLength])

    def rescaleBand(self, image):
        image -= min(image) # set minimum to 0

        if max(image) != 0:
            image /= max(image) # set maximum to 1

        return image
        
    def transformImages(self, images):
        transformedImages = empty((images.shape[0],images.shape[3],images.shape[1],images.shape[2])) # (batchSize, 3, 224, 224)
        
        for i in range(len(images)):
            reshapedImage = empty((images[i].shape[2], images[i].shape[0], images[i].shape[1])) # (3, 224, 224)

            for band in range(images[i].shape[2]):  # change shape from (224,224,3) to (3,224,224)
                reshapedImage[band,:,:] = self.rescaleBand(images[i][:,:,band]) # pixel values in each band between [0,1]

            transformedImage = self.transform(reshapedImage)
            transformedImages[i] = transformedImage
                        
        return transformedImages

    def passiveTrain(self, trainingLoader, epochs):
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(),lr=self.learningRate)
        patience = 0
        previousLoss = 0

        for epoch in range(epochs):  # loop over the dataset multiple times
            totalLoss = 0

            for _, data in enumerate(trainingLoader): # data is a list of [index, images, labels]
                _, images, labels = data
                optimizer.zero_grad() # zero the parameter gradients
                outputs = self.model(self.transformImages(images)).flatten().to(float64) # forward pass
                loss = criterion(outputs,labels)
                loss.backward() # backward pass
                optimizer.step() # optimization
                totalLoss += loss.item()

            print('Epoch ' + str(epoch+1) + ' loss = ' + str(round(totalLoss,3)))

            if epoch > 1:
                if totalLoss >= previousLoss:
                    patience += 1
                else:
                    patience = 0
            
            previousLoss = totalLoss
            print('Patience = ' + str(patience))
            
            if patience == 5:
                break
        
        print('Finished Training')
    
    def activeTrain(self):
        rawUnlabeledImages = take(self.rawImages, self.trainingIndices, axis=0)
        unlabeledImages = take(self.images, self.trainingIndices, axis=0)
        imageLabels = take(self.labels, self.trainingIndices)
        maxPixelVals = array([amax(rawUnlabeledImages[i]) for i in range(len(rawUnlabeledImages))]) # array of maximum pixel value in each image
        #labelingBudget = int(around(self.trainingDataLength/10)) # number of images we are willing to provide labels for
        # labeling budget values: 150, 100, 75
        labelingBudget = 100
        e = 0.1

        def brightestIndices(fraction):
            return sorted(range(len(maxPixelVals)), key=lambda sub:maxPixelVals[sub])[-int(around(fraction*len(maxPixelVals))):]
                
        trainingImageIndices = sample(brightestIndices(e), self.batchSize) # randomly picking a batch among the images with the highest max pixel values
        print('Len training img indices = ',len(trainingImageIndices))
        indexList = tensor(list(arange(len(trainingImageIndices))))
        imagesForNetwork = list(take(unlabeledImages, trainingImageIndices, axis=0))
        imagesForNetworkTensor = empty((len(imagesForNetwork), 224, 224, 3))

        for i in range(len(imagesForNetworkTensor)):
            imagesForNetworkTensor[i] = from_numpy(imagesForNetwork[i])
        
        labelsList = tensor(take(imageLabels, trainingImageIndices, axis=0))
        trainingLoader = [[indexList]+[imagesForNetworkTensor]+[labelsList]] # select the images corresponding to the above indices
        def removeFromUnlabeledImgs():
            for index in sorted(trainingImageIndices, reverse=True):
                delete(unlabeledImages,index) # remove image that will be used for training from the unlabeled set
                delete(imageLabels,index)
                delete(maxPixelVals,index) # remove index that will be used for training from the list of max pixel values

        removeFromUnlabeledImgs()

        while labelingBudget > 0:
            print('Labeling budget = ', labelingBudget)
            print('Training loader len = ', len(trainingLoader))
            #print(trainingLoader[0])
            self.passiveTrain(trainingLoader,2)
            testImageIndices = brightestIndices(2*e)
            print('Len test img indices = ',len(testImageIndices))

            with no_grad():
                imagesForNetwork = take(unlabeledImages,testImageIndices,axis=0)
                imagesForNetworkTensor = empty((len(imagesForNetwork), 224, 224, 3))

                for i in range(len(imagesForNetworkTensor)):
                    imagesForNetworkTensor[i] = from_numpy(imagesForNetwork[i])

                sigmoidOutput = self.model(self.transformImages(imagesForNetworkTensor)).flatten().to(float64)
            
            trainingImageIndices = take(testImageIndices, sorted(range(len(sigmoidOutput)), key=lambda sub:sigmoidOutput[sub])[-self.batchSize:])
            indexList = tensor(list(arange(len(trainingImageIndices))))
            nextImages = take(unlabeledImages, trainingImageIndices, axis=0)
            nextImagesTensor = empty((len(nextImages), 224, 224, 3))

            for i in range(len(nextImagesTensor)):
                nextImagesTensor[i] = from_numpy(nextImages[i])

            nextLabels = tensor(take(imageLabels, trainingImageIndices, axis=0))
            trainingLoader += [[indexList]+[nextImagesTensor]+[nextLabels]]
            removeFromUnlabeledImgs()
            labelingBudget -= self.batchSize
            e = amin([2*e,0.5])

    def test(self):
        correct = 0
        class0Correct = 0
        class1Correct = 0
        total = 0
        class0Total = 0
        class1Total = 0
        predictedPositives = 0

        with no_grad(): # since we're not training, we don't need to calculate the gradients for our outputs
            for i, data in enumerate(self.testLoader):
                _, images, labels = data
                outputs = self.model(self.transformImages(images)).flatten().to(float64)
                predicted = around(outputs) # predicted is a vector with batchSize elements corresponding to the index of the most likely class of each image in the batch
                total += labels.size(0) # number of images in the batch
                correct += (predicted == labels).sum().item() # number of images classified correctly
                class0Indices = where(labels == 0)[0] # indices of no midden images
                class0Total += len(class0Indices) # number of images with no middens
                class0Correct += (take(predicted,class0Indices) == take(labels,class0Indices)).sum().item() # number of true negatives
                class1Indices = where(labels == 1)[0] # indices of midden images
                class1Total += len(class1Indices) # number of images with middens
                class1Correct += (take(predicted,class1Indices) == take(labels,class1Indices)).sum().item() # number of true positives
                predictedPositives += predicted.sum().item() # true positives + false positives

        self.accuracy = round(100*correct/total,3)
        print(f'Accuracy of the neural network on the {self.batchSize*(i+1)} test images = {self.accuracy}%')
        self.class0Accuracy = round(100*class0Correct/class0Total,3)
        print(f'Accuracy on images without middens = {self.class0Accuracy}%')
        self.class1Accuracy = round(100*class1Correct/class1Total,3)
        print(f'Accuracy on images with middens = {self.class1Accuracy}%')

        if predictedPositives > 0:
            self.precision = round(class1Correct/predictedPositives,3)
        else:
            self.precision = 0

        print(f'Precision = {self.precision}') # fraction of images classified as having middens that actually have middens

        self.recall = round(class1Correct/class1Total,3)
        print(f'Recall = {self.recall}') # fraction of images with middens classified as having middens
        
        if self.precision == 0 and self.recall == 0:
            self.f1 = 0
        else:
            self.f1 = round(2*self.precision*self.recall/(self.precision+self.recall),3)
        
        print(f'F1 score = {self.f1}') # harmonic mean of precision and recall
        #save('Results/Test Results: Trial '+self.trial, array([self.accuracy, self.class0Accuracy, self.class1Accuracy, self.precision, self.recall, self.f1]))
if __name__ == '__main__':
    #CNN(epochs=3)
    CNN(epochs=2, learningMode='active')

# path to images: Phase2/Data/Thermal/Images.npy
# path to labels: Phase2/Data/Thermal/Labels.npy

# if argv[1] == 'Thermal':
#     CNN('Phase2/Images.npy', 'Phase2/Labels.npy')

# elif argv[1] == 'RGB':
#     CNN('Data/TrainingImagesRGB.npy', 'Data/TrainingLabelsRGB.npy', batchSize=8, epochs=int(argv[3]), learningRate=0.0001, learningMode=argv[2])

#ThermalCNN = CNN('Data/TrainingImagesCIFAR.npy', 'Data/TrainingLabelsCIFAR.npy', batchSize=8, epochs=int(argv[2]), learningRate=0.0001, learningMode=argv[1])

# for i in range(1,6):
#     CNN('Data/TrainingImagesThermal.npy', 'Data/TrainingLabelsThermal.npy', batchSize=8, epochs=int(argv[2]), learningRate=0.0001, learningMode='passive', trial=str(i))