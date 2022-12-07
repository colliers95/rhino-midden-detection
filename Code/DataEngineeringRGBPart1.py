''' Data Engineering by Lucia Gordon & Samuel Collier '''
from numpy import array, zeros, ma, around, flip, arange, all, sum, amax, amin, uint8, save, where, linspace, any, vstack, hstack, ceil, load
from matplotlib.pyplot import figure, imshow, xlabel, ylabel, colorbar, show, savefig
from pandas import read_csv
from osgeo import gdal
from random import sample
from sys import argv

class DataEngineeringRGBPart1:
    def __init__(self, orthomosaicPath, middenLocationsPath, imageryType='RGB'):
        self.imageryType = imageryType # Thermal or RGB
        self.dataset = gdal.Open(orthomosaicPath)
        self.orthomosaic = []
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
        self.middenLocsInOrthomosaic = []
        self.trainingImages = [] # list of 40x40 pixel images cropped from orthomosaic
        self.trainingLabelMatrices = [] # list of (interval,interval) arrays cropped from the full array of midden locations in the orthomosaic
        self.trainingLabels = []
        self.trainingData = []

        self.getPixelValuesFromTiff()
        self.plotImage(self.orthomosaic, self.imageryType + ' Orthomosaic')
        self.getMiddenLocs(middenLocationsPath)

    def cropArray(self, matrix):
        startRow = 0
        endRow = matrix.shape[0]
        startCol = 0
        endCol = matrix.shape[1]

        for rowIndex in range(len(matrix)):
            if any(matrix[rowIndex]!=0):
                startRow = rowIndex
                break

        matrix = matrix[startRow:]

        for rowIndex in range(len(matrix)): 
            if all(matrix[rowIndex]==0):
                endRow = rowIndex
                break
            else:
                endRow = matrix.shape[0]

        matrix = matrix[:endRow]
        
        for colIndex in range(len(matrix.T)):
            if any(matrix.T[colIndex]!=0):
                startCol = colIndex
                break

        matrix = matrix.T[startCol:].T

        for colIndex in range(len(matrix.T)):
            if all(matrix.T[colIndex]==0):
                endCol = colIndex
                break
            else:
                endCol = matrix.shape[1]

        matrix = matrix.T[:endCol].T

        return matrix, startRow, startRow+endRow, startCol, startCol+endCol

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
            #self.orthomosaic, self.startRow, _, self.startCol, _ = self.cropArray(self.orthomosaic) # crop out rows and columns that are 0
            newRows = zeros((int(ceil((self.orthomosaic.shape[0]-self.interval)/(self.interval/2+self.stride)))*int(self.interval/2+self.stride)+self.interval-self.orthomosaic.shape[0],self.orthomosaic.shape[1]))
            self.orthomosaic = vstack((self.orthomosaic, newRows))
            newColumns = zeros((self.orthomosaic.shape[0],int(ceil((self.orthomosaic.shape[1]-self.interval)/(self.interval/2+self.stride)))*int(self.interval/2+self.stride)+self.interval-self.orthomosaic.shape[1]))
            self.orthomosaic = hstack((self.orthomosaic, newColumns))
        
        elif self.imageryType == 'RGB':
            self.orthomosaic = imageData.astype('uint8')
            indices = []

            for band in range(self.orthomosaic.shape[2]):
                _, startRow, endRow, startCol, endCol = self.cropArray(self.orthomosaic[:,:,band])
                indices.append([startRow, endRow, startCol, endCol])

            self.startRow = amin(array(indices).T[0], axis=0)
            self.endRow = amax(array(indices).T[1], axis=0)
            self.startCol = amin(array(indices).T[2], axis=0)
            self.endCol = amax(array(indices).T[3], axis=0)
            croppedOrthomosaic = zeros((self.endRow-self.startRow, self.endCol-self.startCol, self.orthomosaic.shape[2]))

            for band in range(self.orthomosaic.shape[2]):
                croppedOrthomosaic[:,:,band] = self.orthomosaic[:,:,band][self.startRow:self.endRow, self.startCol:self.endCol]

            self.orthomosaic = croppedOrthomosaic # crop out rows and columns that are 0
            newRows = zeros((int(ceil((self.orthomosaic.shape[0]-self.interval)/(self.interval/2+self.stride)))*int(self.interval/2+self.stride)+self.interval-self.orthomosaic.shape[0],self.orthomosaic.shape[1],self.orthomosaic.shape[2]))
            self.orthomosaic = vstack((self.orthomosaic, newRows))
            newColumns = zeros((self.orthomosaic.shape[0],int(ceil((self.orthomosaic.shape[1]-self.interval)/(self.interval/2+self.stride)))*int(self.interval/2+self.stride)+self.interval-self.orthomosaic.shape[1],self.orthomosaic.shape[2]))
            self.orthomosaic = hstack((self.orthomosaic, newColumns)).astype('uint8')
        save('Data/'+self.imageryType+'Orthomosaic', self.orthomosaic)
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
        savefig('Images/' + self.imageryType + 'Images/' + title + '.png')
        show()
    
    def getMiddenLocs(self, middenLocationsPath):
        xOrigin, pixelWidth, _, yOrigin, _, pixelHeight = self.dataset.GetGeoTransform()
        print(self.imageryType + ': ' + str(xOrigin) + ' ' + str(yOrigin) + ' ' + str(pixelWidth) + ' ' + str(pixelHeight))
        self.middenCoords = load(middenLocationsPath).T # in meters
        self.middenCoords[0] = (self.middenCoords[0]-xOrigin)/pixelWidth - self.startCol # in pixels
        self.middenCoords[1] = (self.middenCoords[1]-yOrigin)/pixelHeight - self.startRow # in pixels
        self.middenCoords = around(self.middenCoords).astype(int)
        self.middenLocsInOrthomosaic = zeros((self.orthomosaic.shape[0],self.orthomosaic.shape[1])).astype(int)
        for index in range(len(self.middenCoords.T)):
            if self.middenCoords.T[index][1] >= self.orthomosaic.shape[0]:
                print(index, self.middenCoords.T[index])
            if self.middenCoords.T[index][0] >= self.orthomosaic.shape[1]:
                print(index, self.middenCoords.T[index])

        for loc in self.middenCoords.T:
            self.middenLocsInOrthomosaic[loc[1],loc[0]] = 1
        save('Data/'+self.imageryType+'MiddenLocsInOrthomosaic', self.middenLocsInOrthomosaic)
        print(self.imageryType + ' num of middens ' + str(sum(self.middenLocsInOrthomosaic)))

Data = DataEngineeringRGBPart1(orthomosaicPath='Tiffs/Firestorm3' + argv[1] + '.tif', middenLocationsPath='Data/AllMiddenLocations.npy', imageryType=argv[1])