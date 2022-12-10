'''Generate Data by Lucia Gordon'''
from numpy import load, amin, flip, arange, save, array, all, sum
from shutil import rmtree
from os import mkdir, path, remove
from random import sample
from matplotlib.pyplot import figure, imshow, axis, savefig, close
from cv2 import imread

class GenerateData:
    def __init__(self):
        # Ask user for input
        # self.imageryType = input('Do you want to crop a Thermal or RGB orthomosaic?\n') # Thermal or RGB
        # self.orthomosaicMatrix = load(input('Please enter the path to the '+self.imageryType+' orthomosaic matrix.\n'))
        # self.middenMatrix = load(input('Please enter the path to the matrix of midden locations in the orthomosaic.\n'))

        # Automatic
        self.imageryType = 'Thermal' # Thermal or RGB
        self.orthomosaicMatrix = load('Phase2/Data/Thermal/OrthomosaicMatrix.npy')
        self.middenMatrix = load('Phase2/Data/Thermal/MiddenMatrix.npy')

        self.images = [] # images cropped from orthomosaic
        self.labelMatrices = [] # midden locations cropped from orthomosaic
        self.labels = [] # labels for each of the cropped images
        self.middenImages = [] # subset of the cropped images that contain middens
        self.emptyImages = [] # subset of the cropped images that are empty
        self.sampledEmptyImages = [] # subset of the empty images
        self.pngArrays = [] # the midden and empty images as arrays after being converted to PNG form

        if self.imageryType == 'Thermal':
            self.interval = 40 # 20m / 0.5m/pixel = 40 pixels
            self.stride = 10 # 5m / 0.5m/pixel = 10 pixels

        elif self.imageryType == 'RGB':
            self.interval = 400 # 20m / 0.05m/pixel = 400 pixels
            self.stride = 100 # 5m / 0.05 m/pixel = 100 pixels
        
        if path.exists('Phase2/Images/'+self.imageryType+'/Middens'):
            rmtree('Phase2/Images/'+self.imageryType+'/Middens')

        if path.exists('Phase2/Images/'+self.imageryType+'/Empty'):
            rmtree('Phase2/Images/'+self.imageryType+'/Empty')

        mkdir('Phase2/Images/'+self.imageryType+'/Middens')     
        mkdir('Phase2/Images/'+self.imageryType+'/Empty')

        self.cropImage() # crop image into smaller individual images
        self.saveData() # save the data to feed into the model

    def cropImage(self):
        top = 0 # begin cropping from the top of the orthomosaic
        bottom = self.stride + self.interval/2 + self.stride # set the height of the image
        
        while bottom < int(self.orthomosaicMatrix.shape[0]): # keep incrementing the bottom value until you hit the bottom of the orthomosaic
            left = 0 # begin cropping from the left end of the orthomosaic
            right = self.stride + self.interval/2 + self.stride # set the width of the image

            while right < int(self.orthomosaicMatrix.shape[1]): # keep incrementing the right value until you hit the right end of the orthomosaic
                croppedImage = self.orthomosaicMatrix[int(top):int(bottom),int(left):int(right)].copy() # create an image cropped from the orthomosaic

                if self.imageryType == 'Thermal': # want to be able to compare the absolute thermal pixel values
                    croppedImage -= amin(croppedImage) # set the minimum pixel value to 0
                
                self.images.append(croppedImage) # save cropped image to list
                self.labelMatrices.append(self.middenMatrix[int(top):int(bottom),int(left):int(right)]) # save the same cropping from the matrix of midden locations
                left += self.stride + self.interval/2 # increment the leftmost boundary for the next image
                right += self.interval/2 + self.stride # increment the rightmost boundary for the next image

            top += self.stride + self.interval/2 # increment the top boundary for the next set of images
            bottom += self.interval/2 + self.stride # increment the bottom boundary for the next set of images

        for i in flip(arange(len(self.images))): # loops through a list of the indices in descending order
            if all(self.images[i]==0): # if an image is blank
                del self.images[i] # remove cropped images whose entries are all 0
                del self.labelMatrices[i] # remove label matrices whose corresponding images have all zeros
        
        self.labels = sum(sum(self.labelMatrices,axis=1),axis=1) # collapses each label matrix to the number of middens in the corresponding cropped image
        
        for index in range(len(self.labels)):
            if self.labels[index] > 1: # if there happens to be more than 1 midden in an image
                self.labels[index] = 1 # set the label to 1 since we only care about whether there is a midden or not
        
        for index in range(len(self.images)):
            if self.labels[index] == 1: # if the image at the index contains a midden
                self.middenImages.append(self.images[index]) # add the image to the list of midden images
            elif self.labels[index] == 0: # if the image at the index does not contain a midden
                self.emptyImages.append(self.images[index]) # add the image to the list of empty images
        
        self.sampledEmptyImages = sample(self.emptyImages,len(self.middenImages)) # generate a list of empty images whose length is the same as the number of images with middens
        
        if path.exists('Phase2/Data/'+self.imageryType+'/RawImages'):
            remove('Phase2/Data/'+self.imageryType+'/RawImages')

        save('Phase2/Data/'+self.imageryType+'/RawImages', self.middenImages + self.sampledEmptyImages)

    def matrixToPNG(self, matrix, path):
        figure(dpi=60.7) # to get resultant arrays of (224,224,3)
        imshow(matrix) # plot the array of pixel values as an image
        axis('off') # remove axes        
        savefig('Phase2/Images/'+self.imageryType+path+'.png', bbox_inches='tight', pad_inches=0) # save the image containing a midden
        close() # close the image to save memory
    
    def saveData(self):
        for index in range(len(self.middenImages)):
            self.matrixToPNG(self.middenImages[index], '/Middens/Midden'+str(index)) # convert the matrix corresponding to a midden image to a PNG
        
        for index in range(len(self.sampledEmptyImages)):
            self.matrixToPNG(self.sampledEmptyImages[index], '/Empty/Empty'+str(index)) # convert the matrix corresponding to an empty image to a PNG

        for index in range(len(self.middenImages)):
            self.pngArrays.append(imread('Phase2/Images/'+self.imageryType+'/Middens/Midden'+ str(index)+'.png')) # convert each midden image PNG to an RGB array

        for index in range(len(self.sampledEmptyImages)):
            self.pngArrays.append(imread('Phase2/Images/'+self.imageryType+'/Empty/Empty'+str(index)+'.png')) # convert each empty image PNG to an RGB array

        if path.exists('Phase2/Data/'+self.imageryType+'/Images.npy'):
            remove('Phase2/Data/'+self.imageryType+'/Images.npy')
        
        if path.exists('Phase2/Data/'+self.imageryType+'/Labels.npy'):
            remove('Phase2/Data/'+self.imageryType+'/Labels.npy')

        save('Phase2/Data/'+self.imageryType+'/Images', array(self.pngArrays)) # save the PNG arrays
        save('Phase2/Data/'+self.imageryType+'/Labels', array([1] * len(self.middenImages) + [0] * len(self.sampledEmptyImages))) # save the labels for the PNG arrays

if __name__ == '__main__':
    GenerateData()

# path to Thermal orthomosaic: Phase2/Data/Thermal/OrthomosaicMatrix.npy
# path to midden matrix for Thermal: Phase2/Data/Thermal/MiddenMatrix.npy
# path to RGB orthomosaic: Phase 2/Data/RGB/OrthomosaicMatrix.npy
# path to midden matrix for RGB: Phase2/Data/RGB/MiddenMatrix.npy