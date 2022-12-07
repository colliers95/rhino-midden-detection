''' CNN by Lucia Gordon '''
from numpy import amax, amin, append, array, around, where, take, min, load, delete, mean, std, zeros, save
from pandas import read_csv
from random import sample
from torch import device, cuda, nn, sum, optim, max, FloatTensor, no_grad, from_numpy
from torchvision.models import vgg16
from torch.utils.data import random_split, DataLoader
from torchsummary import summary
from math import ceil
from sys import argv
from emails.AL_email import SendReceiveEmail
from matplotlib.pyplot import figure, imshow, xlabel, ylabel, colorbar, show, savefig

class CNN:
    def __init__(self, images, labels, batchSize=8, epochs=30, learningRate=0.0001, learningMode='passive', trial='', sender_email = "", sender_password="", receiver_emails=[]):
        self.model = None
        self.images = load(images)
        self.labels = load(labels)
        #goodMiddenIndices = [916, 1482, 1759, 1760, 1859, 1860, 1943, 1944, 2065, 2581, 2582, 3037, 3489, 3681, 3782, 4217, 4318, 4378, 4379, 4805, 5486, 6673, 6674, 6879, 7044, 7209, 7210, 7356, 7456, 7755, 7756, 8273, 8337, 8338, 8501, 8920, 9438]
        #self.data = sample(list(zip(self.images,self.labels)),100)
        #print(len(goodMiddenIndices))
        self.data = list(zip(self.images,self.labels))
        #goodMiddens = take(self.data, goodMiddenIndices, axis=0)
        allMiddens = []
        for point in self.data:
            if point[1] == 1:
                allMiddens.append(point)
        allEmptyImages = []
        for point in self.data:
            if point[1] == 0:
                allEmptyImages.append(point)
        emptyImages = sample(allEmptyImages, len(allMiddens))
        self.data = allMiddens+emptyImages
        print('Number of images = ' + str(len(self.data)))
        self.images, self.labels = zip(*self.data)
        self.dataLength = len(self.images)
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
        self.trainingImages = []
        self.trainingLabels = []
        self.classWeights = []
        self.accuracy = 0
        self.class0Accuracy = 0
        self.class1Accuracy = 0
        self.precision = 0
        self.recall = 0
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.receiver_emails = receiver_emails

        self.useVGG16()
        self.preprocessData()
        self.computeClassWeights()

        if self.learningMode == 'passive':
            self.passiveTrain(self.trainingLoader, self.batchSize)
        
        elif self.learningMode == 'active':
            self.activeTrain()
        self.test()

    @staticmethod
    def plotImage(imageData, imageryType, title='Figure', show=False):
        assert imageryType in ['Thermal', 'RGB']
        figure(title, dpi=150)

        if imageryType == 'Thermal':
            imagePlot = imshow(imageData)
            imagePlot.set_cmap('plasma')
            cb = colorbar()
            cb.set_label('Pixel value')

        elif imageryType == 'RGB':
            imshow(imageData)

        xlabel('X (pixels)')
        ylabel('Y (pixels)')
        savefig('Images/' + imageryType + 'ImagesTemp/' + title + '.png')
        if show:
            show()

    def useVGG16(self):
        myDevice = device('cuda' if cuda.is_available() else 'cpu')
        self.model = vgg16('VGG16_Weights.DEFAULT').to(myDevice) # imports a pretrained vgg16 CNN
        lastLayerIndex = len(self.model.classifier)-1
        oldFinalLayer = self.model.classifier.__getitem__(lastLayerIndex)
        newFinalLayers = nn.Sequential(nn.Linear(in_features=oldFinalLayer.in_features,out_features=2,bias=True),nn.Softmax(dim=1)) # changes the output of the last hidden layer to be binary and adds softmax
        self.model.classifier.__setitem__(lastLayerIndex,newFinalLayers)

    def preprocessData(self):
        transformedImages = []

        for image in self.images:
            if len(image.shape) == 2:
                next = append([image], [image], axis=0) # reshape to (2,40,40)
                image = append(next, [image], axis=0) # reshape to (3,40,40)
            image = from_numpy(image).float()
            transformedImages.append(image)
        
        self.images = transformedImages
        data = list(zip(self.images,self.labels))
        self.trainingDataLength = around(0.8*self.dataLength).astype(int) # use 80% of the data for training
        self.testDataLength = self.dataLength - self.trainingDataLength # use 20% of the data for testing
        self.trainingData, self.testData = random_split(data,[self.trainingDataLength,self.testDataLength]) # splits the data into training and test sets
        
        for _, point in enumerate(self.trainingData):
            self.trainingImages.append(point[0])
            self.trainingLabels.append(point[1])
        
        self.trainingLoader = DataLoader(self.trainingData, batch_size=self.batchSize, shuffle=True, num_workers=1) # batches and shuffles the training data
        self.testLoader = DataLoader(self.testData, batch_size=self.batchSize, shuffle=True, num_workers=1) # batches and shuffles the test data

    def computeClassWeights(self):
        numClass1 = 0

        for _,batch in enumerate(self.trainingLoader):
            numClass1 += batch[1].sum().item() # sum of labels in batch
        
        self.classWeights = [numClass1/self.trainingDataLength, 1 - numClass1/self.trainingDataLength] # the weights are proportional to the number of points in the other class and sum to 1
        print(self.classWeights)
        numClass1Test = 0

        for _,batch in enumerate(self.testLoader):
            numClass1Test += batch[1].sum().item() # sum of labels in batch
        
        print([numClass1Test/self.testDataLength, 1 - numClass1Test/self.testDataLength])

    def passiveTrain(self, trainingLoader, batchSize):
        criterion = nn.CrossEntropyLoss(weight=FloatTensor(self.classWeights))
        optimizer = optim.Adam(self.model.parameters(),lr=self.learningRate)
        interval = ceil(len(trainingLoader)/self.batchSize/10)
        patience = 0
        previousLoss = 0

        for epoch in range(self.epochs):  # loop over the dataset multiple times
            runningLoss = 0.0
            totalLoss = 0

            for i, data in enumerate(trainingLoader,0): # data is a list of [inputs,labels]
                inputs, labels = data
                optimizer.zero_grad() # zero the parameter gradients
                outputs = self.model(inputs) # forward pass
                loss = criterion(outputs,labels)
                loss.backward() # backward pass
                optimizer.step() # optimization
                runningLoss += loss.item()
                totalLoss += loss.item()

                if i % interval == interval-1:
                    #print(f'Epoch={epoch+1}, Images {self.batchSize*(i+1-interval)}-{self.batchSize*(i+1)}, Loss={runningLoss/interval:.3f}') # average loss per batch
                    runningLoss = 0.0

            print('Epoch ' + str(epoch+1) + ' loss = ' + str(totalLoss))

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

    def hand_label_images(self, ImageIndices, images_to_send):
        """Get hand-labelled results from modelled-defined images

        Parameters
        ----------
        ImageIndices : np.ndarray
                    The indices of the model-selected images within the set of training data
        images_to_send : np.ndarray
                    The selected images in array form
        
        Returns
        ----------
        hand_labels : np.ndarray
                    An ordered array of labels indicating whether the associated images contain middens
        """
        image_paths = []
        img_indices = []
        for idx, img in zip(ImageIndices, images_to_send):
            # Make and save plot
            title = "tile_{}_thermal".format(str(idx))
            CNN.plotImage(img, imageryType="Thermal", title=title)
            image_paths = image_paths.append('Images/ThermalImagesTemp/' + title + '.png')
            img_indices = img_indices.append(str(idx))
        
        mail = SendReceiveEmail(
            self.sender_email, self.sender_password, self.receiver_emails, img_indices, image_paths
        )
        mail.create_email_parts()
        mail.send_email()
        mail.read_email()
        email_answers = mail.get_email_answers() # returns an OrderedDict
        hand_labels = SendReceiveEmail.convert_to_labels(email_answers) # returns a numpy array of ints

        return hand_labels
    
    def activeTrain(self):
        unlabeledImages = self.trainingImages # all the training images start out unlabeled
        imageLabels = self.trainingLabels
        maxPixelVals = array([max(unlabeledImages[i]) for i in range(len(unlabeledImages))]) # array of maximum pixel value in each image
        labelingBudget = int(around(self.trainingDataLength/10)) # number of images we are willing to provide labels for
        e = 0.01

        def brightestIndices(fraction):
            return sorted(range(len(maxPixelVals)), key=lambda sub:maxPixelVals[sub])[-int(around((1-fraction)*len(maxPixelVals))):]
        
        trainingImageIndices = sample(brightestIndices(0.99), self.batchSize) # randomly picking a batch among the images with the highest max pixel values
        
        # With indices decided, create thermal plots and send to user (RGB to come later)
        imgs_to_send = take(unlabeledImages, trainingImageIndices, axis=0)

        hand_labels = self.hand_label_images(trainingImageIndices, )

        trainingLoader = DataLoader(list(zip(imgs_to_send,hand_labels)), batch_size=1, shuffle=True, num_workers=1) # select the images corresponding to the above indices

        def removeFromUnlabeledImgs():  
            for index in sorted(trainingImageIndices, reverse=True):
                delete(unlabeledImages,index) # remove image that will be used for training from the unlabeled set
                delete(imageLabels,index)
                delete(maxPixelVals,index) # remove index that will be used for training from the list of max pixel values

        removeFromUnlabeledImgs()

        while labelingBudget > 0:
            print(labelingBudget)
            self.passiveTrain(trainingLoader, 1)
            testImageIndices = brightestIndices(0.99-e).tolist()
            with no_grad():
                sigmoidOutput = self.model(take(unlabeledImages,testImageIndices)).data
            trainingImageIndices = sorted(range(len(sigmoidOutput)), key=lambda sub:sigmoidOutput[sub])[-self.batchSize:]
            nextImages = take(unlabeledImages, trainingImageIndices, axis=0)
            new_hand_labels = self.hand_label_images(trainingImageIndices, nextImages)
            trainingLoader = append(trainingLoader,list(zip(nextImages,new_hand_labels)))
            removeFromUnlabeledImgs()
            labelingBudget -= self.batchSize
            e = min([2*e,0.5])

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
                images, labels = data
                outputs = self.model(images)
                _, predicted = max(outputs.data, 1) # predicted is a vector with batchSize elements corresponding to the index of the most likely class of each image in the batch
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
            print(f'Precision = {self.precision}') # fraction of images classified as having middens that actually have middens
        else:
            print(f'Precision = 0.0')

        self.recall = round(class1Correct/class1Total,3)
        print(f'Recall = {self.recall}') # fraction of images with middens classified as having middens
        save('Results/Test Results'+self.trial, array([self.accuracy, self.class0Accuracy, self.class1Accuracy, self.precision, self.recall]))

# Get credentials
with open("./emails/email.config") as config:
    creds = {
        line.strip().split("|")[0]: line.strip().split("|")[1]
        for line in config.readlines()
    }

# Update to change sender (unlikely) and target (likely)
sender_email = creds.get("sender_email")
sender_password = creds.get("sender_password")
receiver_emails = [
    creds.get("coauthor_email"),  # coauthor_email
]

ThermalCNN = CNN('Data/TrainingImagesThermal.npy', 'Data/TrainingLabelsThermal.npy', batchSize=8, epochs=int(argv[2]), learningRate=0.0001, learningMode=argv[1], sender_email=sender_email, sender_password=sender_password, receiver_emails=receiver_emails)
#ThermalCNN = CNN('Data/TrainingImagesCIFAR.npy', 'Data/TrainingLabelsCIFAR.npy', batchSize=8, epochs=int(argv[2]), learningRate=0.0001, learningMode=argv[1])

# for i in range(1,6):
#     CNN('Data/TrainingImagesThermal.npy', 'Data/TrainingLabelsThermal.npy', batchSize=8, epochs=1, learningRate=0.0001, learningMode='passive', trial=str(i))