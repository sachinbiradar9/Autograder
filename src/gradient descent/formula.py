from datasets import *
import datasets,gd,linear,runClassifier
f = linear.LinearClassifier({'lossFunction': linear.HingeLoss(), 'lambda': 1, 'numIter': 100000, 'stepSize': 1})
runClassifier.trainTestSet(f, datasets.TwoDDiagonal)
print(f)


