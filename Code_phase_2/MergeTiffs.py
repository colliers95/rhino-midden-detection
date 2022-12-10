'''Merge Tiffs by Samuel Collier & Lucia Gordon'''
from osgeo import gdal

class MergeTiffs:
    def __init__(self):
        self.tiffPaths = []
        self.mergedTiffPath = ''

        self.getPaths()
        gdal.Warp(self.mergedTiffPath,self.tiffPaths,format='GTiff')
    
    def getPaths(self):
        numTiffs = int(input('How many tiffs would you like to merge?\n'))
        for n in range(numTiffs):
            self.tiffPaths.append(input('Please enter the path of tiff ' + str(n+1) + '.\n'))
        self.mergedTiffPath = input('Please enter a new path for the merged tiff.\n')

MergeTiffs()

# tiff n path: Tiffs/Firestorm3/imageryType/n.tif
# path for merged tiff: Tiffs/Firestorm3/imageryType/Merged.tif