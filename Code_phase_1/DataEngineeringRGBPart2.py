''' Data Engineering by Lucia Gordon & Samuel Collier '''
from numpy import array, zeros, ma, around, flip, arange, all, sum, amax, amin, uint8, save, where, linspace, any, vstack, hstack, ceil, load
from matplotlib.pyplot import figure, imshow, xlabel, ylabel, colorbar, show, savefig
from pandas import read_csv
from osgeo import gdal
from random import sample
from sys import argv

class DataEngineeringRGBPart2:
    def __init__(self, orthomosaicPath, middenLocationsPath, imageryType='RGB'):
        self.imageryType = imageryType # Thermal or RGB
        #self.dataset = gdal.Open(orthomosaicPath)
        self.orthomosaic = load(orthomosaicPath)
        self.startCol = 0
        self.endCol = 0
        self.startRow = 0
        self.endRow = 0
        if self.imageryType == 'Thermal':
            self.interval = 40 # 20m / 0.5m/pixel = 40 pixels
            self.stride = 10 # 5m / 0.5m/pixel = 10 pixels
        elif self.imageryType == 'RGB':
            self.interval = 400
            self.stride = 100
        self.middenCoords = []
        self.middenLocsInOrthomosaic = load(middenLocationsPath)
        self.trainingImages = [] # list of 40x40 pixel images cropped from orthomosaic
        self.trainingLabelMatrices = [] # list of (interval,interval) arrays cropped from the full array of midden locations in the orthomosaic
        self.trainingLabels = []
        self.trainingData = []

        self.plotImage(self.orthomosaic, self.imageryType + ' Orthomosaic')
        #self.getMiddenLocs(middenLocationsPath)
        self.generateTrainingData()
        self.showMiddensOnImage()
        
    def plotImage(self, imageData, title='Figure'):
        figure(title, dpi=150)

        if self.imageryType == 'Thermal':
            imagePlot = imshow(imageData)
            imagePlot.set_cmap('plasma')
            cb = colorbar()
            cb.set_label('Pixel value')

        elif self.imageryType == 'RGB':
            imshow(imageData)

        xlabel('X (pixels)')
        ylabel('Y (pixels)')
        savefig('Images/' + self.imageryType + 'Images/' + title + '.png')
        show()
    
    def normalizeImage(self, image):
        image -= amin(image) # set minimum to 0

        if amax(image) != 0:
            image /= amax(image) # set maximum to 1

        image = (image-0.5)/0.5 # set pixel values to be between -1 and 1
        return image

    def generateTrainingData(self):
        top = 0
        bottom = self.stride + self.interval/2 + self.stride
        
        while bottom < int(self.orthomosaic.shape[0]):
            left = 0
            right = self.stride + self.interval/2 + self.stride

            while right < int(self.orthomosaic.shape[1]):
                image = self.orthomosaic[int(top):int(bottom),int(left):int(right)].copy()
                normalizedImage = zeros((image.shape))

                for band in range(image.shape[0]):
                    normalizedImage[band,:,:] = self.normalizeImage(image[band,:,:])
                
                image = normalizedImage

                self.trainingImages.append(image)
                self.trainingLabelMatrices.append(self.middenLocsInOrthomosaic[int(top):int(bottom),int(left):int(right)])
                left += self.stride + self.interval/2
                right += self.interval/2 + self.stride

            top += self.stride + self.interval/2
            bottom += self.interval/2 + self.stride

        for i in flip(arange(len(self.trainingImages))):
            if all(self.trainingImages[i]==-1):
                del self.trainingImages[i] # remove images whose entries are all 0
                del self.trainingLabelMatrices[i] # remove label matrices whose corresponding images have all zeros
        self.trainingLabels = sum(sum(self.trainingLabelMatrices,axis=1),axis=1) # collapses each label matrix to 1 if there's a 1 in the matrix or 0 otherwise
        for index in range(len(self.trainingLabels)):
            if self.trainingLabels[index] > 1:
                self.trainingLabels[index] = 1
        print(self.imageryType + ' sum of training labels ' + str(sum(self.trainingLabels)))
        self.trainingData = array(list(zip(self.trainingImages,self.trainingLabels)),dtype=object) # pairs each training image with its label
        print(self.imageryType + ' data shape: ' + str(self.trainingData.shape))
        save('Data/TrainingImages'+self.imageryType, self.trainingImages)
        save('Data/TrainingLabels'+self.imageryType, self.trainingLabels)
    
    def showMiddensOnImage(self):
        middenIndices = where(self.trainingLabels == 1)[0]

        for index in sample(middenIndices.tolist(),10): # look at 10 random images with middens
            print(index)
            figure(self.imageryType + ' Image ' + str(index), dpi=150)
            image = imshow(self.trainingImages[index])
            
            if self.imageryType == 'Thermal':
                image.set_cmap('plasma')
                cb = colorbar()
                cb.set_label('Pixel value')
            midden = imshow(ma.masked_less(self.trainingLabelMatrices[index],1))
            midden.set_cmap('inferno')
            xlabel('X (pixels)')
            ylabel('Y (pixels)')
            savefig('Images/' + self.imageryType + 'Images/Image' + str(index) + '.png')
            show()

Data = DataEngineeringRGBPart2(orthomosaicPath='Data/RGBOrthomosaic.npy', middenLocationsPath='Data/RGBMiddenLocsInOrthomosaic.npy')