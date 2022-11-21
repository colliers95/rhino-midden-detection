''' Data Engineering by Lucia Gordon & Samuel Collier '''
from numpy import array, zeros, ma, around, flip, arange, all, sum, amax, amin, uint8, save, where, linspace, any, vstack, hstack
from matplotlib.pyplot import figure, imshow, xlabel, ylabel, colorbar, show, savefig
from pandas import read_csv
from osgeo import gdal
from random import sample
from sys import argv

class DataEngineering:
    def __init__(self, orthomosaicPath, csvPath, imageryType):
        self.imageryType = imageryType # Thermal or RGB
        self.dataset = gdal.Open(orthomosaicPath)
        self.orthomosaic = []
        self.startCol = 0
        self.endCol = 0
        self.startRow = 0
        self.endRow = 0
        self.middenCoords = []
        self.middenLocsInOrthomosaic = []
        self.trainingImages = [] # list of 40x40 pixel images cropped from orthomosaic
        self.trainingLabelMatrices = [] # list of (interval,interval) arrays cropped from the full array of midden locations in the orthomosaic
        self.trainingLabels = []
        self.trainingData = []

        self.getPixelValuesFromTiff()
        self.plotImage(self.orthomosaic, self.imageryType + ' Orthomosaic')
        self.getMiddenLocs(csvPath)
        self.generateTrainingData()
        self.showMiddensOnImage()

    def cropArray(self, matrix):
        startCol = 0
        endCol = matrix.shape[1]
        startRow = 0
        endRow = matrix.shape[0]

        for colIndex in range(len(matrix.T)):
            if any(matrix.T[colIndex]!=0):
                startCol = colIndex
                break

        matrix = matrix.T[startCol:].T

        for colIndex in range(len(matrix.T)):
            if all(matrix.T[colIndex]==0):
                endCol = colIndex
                break

        matrix = matrix.T[:endCol].T

        for rowIndex in range(len(matrix)):
            if any(matrix[rowIndex]!=0):
                startRow = rowIndex
                break

        matrix = matrix[startRow:]


        for rowIndex in range(len(matrix)):
            if all(matrix[rowIndex]==0):
                endRow = rowIndex
                break

        matrix = matrix[:endRow]

        return matrix, startCol, endCol, startRow, endRow

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
            self.orthomosaic, self.startCol, self.endCol, self.startRow, self.endRow = self.cropArray(self.orthomosaic)
            array1 = zeros((4000-self.orthomosaic.shape[0],self.orthomosaic.shape[1]))
            self.orthomosaic = vstack((self.orthomosaic,array1))
            array2 = zeros((self.orthomosaic.shape[0],3400-self.orthomosaic.shape[1]))
            self.orthomosaic = hstack((self.orthomosaic,array2))
        elif self.imageryType == 'RGB':
            self.orthomosaic = imageData.astype(uint8)
            print(self.imageryType + ' orthomosaic shape: ' + str(self.orthomosaic.shape))
            indices = []
            for band in range(self.orthomosaic.shape[-1]):
                print(self.orthomosaic[:,:,band].shape)
                _, startCol, endCol, startRow, endRow = self.cropArray(self.orthomosaic[:,:,band])
                indices.append([startCol, endCol, startRow, endRow])
            self.startCol = amin(array(indices).T[0], axis=0)
            self.endCol = amax(array(indices).T[1], axis=0)
            self.startRow = amin(array(indices).T[2], axis=0)
            self.endRow = amax(array(indices).T[3], axis=0)

            for band in range(self.orthomosaic.shape[-1]):
                self.orthomosaic[:,:,band] = self.orthomosaic[:,:,band][self.startRow:self.endRow, self.startCol:self.endCol]

            array1 = zeros((4000*135/11-self.orthomosaic.shape[0],self.orthomosaic.shape[1]))
            self.orthomosaic = vstack((self.orthomosaic,array1))
            array2 = zeros((self.orthomosaic.shape[0],3400*135/11-self.orthomosaic.shape[1]))
            self.orthomosaic = hstack((self.orthomosaic,array2))
        print(self.imageryType + ' orthomosaic shape: ' + str(self.orthomosaic.shape))

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
        savefig('Data/' + self.imageryType + 'Images/' + title + '.png')
        show()
    
    def getMiddenLocs(self, csvPath):
        dataframe = read_csv(csvPath, usecols=['x','y'])
        self.middenCoords = dataframe.to_numpy() # in meters
        xOrigin, pixelWidth, _, yOrigin, _, pixelHeight = self.dataset.GetGeoTransform()
        print(self.imageryType + ': ' + str(xOrigin) + ' ' + str(yOrigin) + ' ' + str(pixelWidth) + ' ' + str(pixelHeight))
        self.middenCoords.T[0] = (self.middenCoords.T[0]-xOrigin)/pixelWidth - self.startCol # in pixels
        self.middenCoords.T[1] = (self.middenCoords.T[1]-yOrigin)/pixelHeight - self.startRow # in pixels
        self.middenCoords = around(self.middenCoords).astype(int)
        self.middenLocsInOrthomosaic = zeros((self.orthomosaic.shape[0],self.orthomosaic.shape[1])).astype(int)
        for loc in self.middenCoords:
            self.middenLocsInOrthomosaic[loc[1],loc[0]] = 1
        #print(self.imageryType + ' num of middens ' + str(sum(self.middenLocsInOrthomosaic)))
    
    def generateTrainingData(self):
        if self.imageryType == 'Thermal':
            interval = 40 # 20m / 0.5m/pixel = 40 pixels
            stride = 10 # 5m / 0.5m/pixel = 10 pixels
        elif self.imageryType == 'RGB':
            interval = 400
            stride = 100
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
            if all(self.trainingImages[i]==0):
                del self.trainingImages[i] # remove images whose entries are all 0
                del self.trainingLabelMatrices[i] # remove label matrices whose corresponding images have all zeros
        for arr in self.trainingLabelMatrices:
            if arr.shape != (40,40):
                print('wrong shape')
        self.trainingLabels = sum(sum(self.trainingLabelMatrices,axis=1),axis=1) # collapses each label matrix to 1 if there's a 1 in the matrix or 0 otherwise
        print(self.imageryType + ' sum of training labels ' + str(sum(self.trainingLabels)))
        self.trainingData = array(list(zip(self.trainingImages,self.trainingLabels)),dtype=object) # pairs each training image with its label
        print(self.imageryType + ' data shape: ' + str(self.trainingData.shape))
        save('Data/TrainingImages'+self.imageryType, self.trainingImages)
        save('Data/TrainingLabels'+self.imageryType, self.trainingLabels)
    
    def showMiddensOnImage(self):
        middenIndices = where(self.trainingLabels == 1)[0]

        for index in sample(middenIndices.tolist(),5): # look at 10 random images with middens
            figure(self.imageryType + ' Image ' + str(index), dpi=150)
            #image = imshow(self.trainingImages[index])
            #if self.imageryType == 'Thermal':
                #image.set_cmap('plasma')
                #cb = colorbar()
                #cb.set_label('Pixel value')
            midden = imshow(ma.masked_less(self.trainingLabelMatrices[index],1))
            #print(sum(sum(self.trainingLabelMatrices[index],axis=1),axis=1))
            midden.set_cmap('inferno')
            xlabel('X (pixels)')
            ylabel('Y (pixels)')
            savefig('Data/' + self.imageryType + 'Images/Image' + str(index) + '.png')
            show()

Data = DataEngineering(orthomosaicPath='Tiffs/Firestorm3' + argv[1] + '.tif', csvPath='Data/MiddenLocations.csv', imageryType=argv[1])