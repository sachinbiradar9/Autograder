from datasets import *
import datasets,gd,linear,runClassifier
f = linear.LinearClassifier({'lossFunction': linear.HingeLoss(), 'lambda': 1, 'numIter': 500000, 'stepSize': 0.5})
runClassifier.trainTestSet(f, datasets.TwoDDiagonal)
print(f)


