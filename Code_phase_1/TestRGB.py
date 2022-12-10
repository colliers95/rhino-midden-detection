''' Data Engineering by Lucia Gordon & Samuel Collier '''
from numpy import array, zeros, ma, around, flip, arange, all, sum, amax, amin, uint8, save, where, linspace
from matplotlib.pyplot import figure, imshow, xlabel, ylabel, colorbar, show, savefig
from pandas import read_csv
from osgeo import gdal
from random import sample

class DataEngineering:
    def __init__(self, orthomosaicPath, csvPath, imageryType):
        self.imageryType = imageryType # Thermal or RGB
        self.dataset = gdal.Open(orthomosaicPath)
        self.orthomosaic = []
        self.middenCoords = []
        self.middenLocsInOrthomosaic = []
        self.trainingImages = [] # list of 40x40 pixel images cropped from orthomosaic
        self.trainingLabelMatrices = []
        self.trainingLabels = []
        self.trainingData = []

        self.getPixelValuesFromTiff()
        self.plotImage(self.orthomosaic, self.imageryType + ' Orthomosaic')
        self.getMiddenLocs(csvPath)
        self.generateTrainingData()

    def getPixelValuesFromTiff(self):
        numCols = self.dataset.RasterXSize
        numRows = self.dataset.RasterYSize
        numBands = self.dataset.RasterCount
        imageData = zeros((numRows, numCols, numBands))
        for band in range(numBands):
            data = self.dataset.GetRasterBand(band+1)
            imageData[:,:,band] = data.ReadAsArray(0,0,numCols,numRows) # (x offset, y offset, x size, y size)
        if self.imageryType == 'Thermal':
            self.orthomosaic = imageData[:,:,3] # extracting fourth band, which corresponds to temperature
            orthomosaicMin = amin(ma.masked_less(self.orthomosaic,2000)) # 7638 = min pixel value in orthomosaic
            self.orthomosaic = ma.masked_less(self.orthomosaic-orthomosaicMin,0).filled(0) # shift the pixel values such that the min of the orthomosaic is 0 and set the background pixels to 0
            self.orthomosaic = (255/amax(self.orthomosaic)*self.orthomosaic).astype('uint8') # convert to grayscale
        elif self.imageryType == 'RGB':
            self.orthomosaic = imageData.astype(uint8)

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
        savefig('Data/' + title + '.png')
        show()
    
    def getMiddenLocs(self, csvPath):
        dataframe = read_csv(csvPath, usecols=['x','y'])
        self.middenCoords = dataframe.to_numpy() # in meters
        xOrigin, pixelWidth, _, yOrigin, _, pixelHeight = self.dataset.GetGeoTransform()
        self.middenCoords.T[0] = (self.middenCoords.T[0]-xOrigin)/pixelWidth # in pixels
        self.middenCoords.T[1] = (self.middenCoords.T[1]-yOrigin)/pixelHeight # in pixels
        self.middenCoords = around(self.middenCoords).astype(int)
        self.middenLocsInOrthomosaic = zeros((self.orthomosaic.shape[0],self.orthomosaic.shape[1])).astype(int)
        for loc in self.middenCoords:
            self.middenLocsInOrthomosaic[loc[1],loc[0]] = 1
    
    def generateTrainingData(self):
        if self.imageryType == 'Thermal':
            interval = 40 # 20m / 0.5m/pixel = 40 pixels
            stride = 10 # 5m / 0.5m/pixel = 10 pixels
        elif self.imageryType == 'RGB':
            interval = 400
            stride = 100
        #trainingLabelMatrices = [] # list of (interval,interval) arrays cropped from the full array of midden locations in the orthomosaic
        bottom = 0
        top = stride + interval/2 + stride
        
        while top < int(self.orthomosaic.shape[1]):
            left = 0
            right = stride + interval/2 + stride
            while right < int(self.orthomosaic.shape[0]):
                self.trainingImages.append(self.orthomosaic[int(bottom):int(top),int(left):int(right)])
                self.trainingLabelMatrices.append(self.middenLocsInOrthomosaic[int(bottom):int(top),int(left):int(right)])
                left += stride + interval/2
                right += interval/2 + stride
            bottom += stride + interval/2
            top += interval/2 + stride

        for i in flip(arange(len(self.trainingImages))):
            if all(self.trainingImages[i])==0:
                del self.trainingImages[i] # remove images whose entries are all 0
                del self.trainingLabelMatrices[i] # remove label matrices whose corresponding images have all zeros
        self.trainingLabels = sum(sum(self.trainingLabelMatrices,axis=1),axis=1) # collapses each label matrix to 1 if there's a 1 in the matrix or 0 otherwise
        self.trainingData = array(list(zip(self.trainingImages,self.trainingLabels)),dtype=object) # pairs each training image with its label
        save('Data/TrainingImages'+self.imageryType, self.trainingImages)
        save('Data/TrainingLabels'+self.imageryType, self.trainingLabels)
    
    def showMiddensOnImage(self):
        middenIndices = where(self.trainingLabels == 1)[0]

        for index in sample(middenIndices.tolist(),10): # look at 10 random images with middens
            figure(self.imageryType + ' Image ' + str(index), dpi=150)
            if self.imageryType == 'Thermal':
                thermalImage = imshow(self.trainingImages[index])
                thermalImage.set_cmap('plasma')
                cb = colorbar()
                cb.set_label('Pixel value')
            elif self.imageryType == 'RGB':
                imshow(self.trainingImages[index])
            midden = imshow(ma.masked_less(self.trainingLabelMatrices[index],1))
            midden.set_cmap('inferno')
            xlabel('X (pixels)')
            ylabel('Y (pixels)')
            savefig('Data/Image' + str(index) + '.png')
            show()


ThermalData = DataEngineering(orthomosaicPath='Tiffs/Firestorm3Thermal.tif', csvPath='Data/MiddenLocations.csv', imageryType='Thermal')
#RGBData = DataEngineering(orthomosaicPath='Tiffs/Firestorm3RGB.tif', csvPath='Data/MiddenLocations.csv', imageryType='RGB')

# RGBData.plotImage(RGBData.trainingImages[10])
# RGBData.plotImage(RGBData.trainingImages[20])
# RGBData.plotImage(RGBData.trainingImages[30])
# RGBData.plotImage(RGBData.trainingImages[40])
# RGBData.plotImage(RGBData.trainingImages[50])

ThermalData.showMiddensOnImage()

# middenIndices = where(ThermalData.trainingLabels==0)[0]
# for index in sample(middenIndices.tolist(),10): # look at 10 random images with middens
#     print(index)
#     ThermalData.plotImage(ThermalData.trainingImages[index])
#     ThermalData.plotImage(ThermalData.trainingLabelMatrices[index])
# print('swap to random')
# for index in sample(around(linspace(0,10000)).astype(int).tolist(),10): # look at 10 random images with middens
#     print(index)
#     ThermalData.plotImage(ThermalData.trainingImages[index])