from GenerateData import GenerateData
from CNN import CNN
from numpy import array, mean, std, save
from shutil import rmtree
from os import mkdir, path
from matplotlib.pyplot import figure, scatter, plot, xlabel, ylabel, xticks, legend, title, savefig, show

results = []
numTrials = 5

for i in range(1,numTrials+1):
    print('Trial ' + str(i))
    GenerateData()
    trial = CNN(epochs=15)
    results.append([trial.accuracy, trial.class0Accuracy, trial.class1Accuracy, trial.precision, trial.recall, trial.f1])

results = array(results).T

print('Mean accuracy over ' + str(numTrials) + ' trials = ', round(mean(results[0]),3))
print('Standard deviation of the accuracy over ' + str(numTrials) + ' trials = ', round(std(results[0]),3))
print('Mean recall over ' + str(numTrials) + ' trials = ', round(mean(results[4]),3))
print('Standard deviation of the recall over ' + str(numTrials) + ' trials = ', round(std(results[4]),3))
print('Mean F1-score over ' + str(numTrials) + ' trials = ', mean(results[5]))
print('Standard deviation of the F1-secore over ' + str(numTrials) + ' trials = ', round(std(results[5]),3))

if path.exists('Phase2/Results/Thermal'):
    rmtree('Phase2/Results/Thermal')

mkdir('Phase2/Results/Thermal')

save('Phase2/Results/Thermal/Results', results)

def plotQuantity(quantity, index):
    figure(quantity+' Results', dpi=200)
    avg = mean(results[index])
    scatter(range(1, numTrials+1), results[index], c='b')
    plot(range(1, numTrials+1), results[index], c='b')
    plot(range(1, numTrials+1), [avg]*numTrials, c='g', label='mean')
    xlabel('Trial')

    if quantity == 'Accuracy':
        ylabel(quantity + ' (%)')
    else:
        ylabel(quantity)

    xticks(range(1, numTrials+1))
    legend()
    title(quantity + ' Over ' + str(numTrials) + ' Trials')
    savefig('Phase2/Results/Thermal/'+quantity+'.png')
    show()

plotQuantity('Accuracy', 0)
plotQuantity('Recall', 4)
plotQuantity('F1-Score', 5)