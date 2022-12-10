Code.py

1. MergeTiffs.py: Combines several tiffs into one
2. MapMiddensOntoOrthomosaic.py: Maps middens onto an orthomosaic and saves the orthomosaic and midden locations as numpy arrays
3. GenerateData.py: Crops the orthomosaic with middens to yield an equal number of images with and without middens along with their labels; the set of empty images is generated randomly and will be different for each run
4. CNN.py: Trains and tests a neural network on a set of 3-band images
5. RunTrials.py: Runs trials and plots the model performance