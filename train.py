import matplotlib.pyplot as plt
import numpy as np

import seaborn as sb
import argparse
from ClassifierModelDirectory import ClassifierModelDirectory
from ClassifierModel import ClassifierModel

parser = argparse.ArgumentParser(description = 'training model for flower classification.')
parser.add_argument('data_dir', help="Set directory for the training data")
parser.add_argument('--save_dir', help='Set directory to save checkpoints')
parser.add_argument('--arch', help='Choose architecture')
parser.add_argument('--learning_rate', help='Set learning rate.', type=float)
parser.add_argument('--hidden_units', help='Set hidden units.', type=int)
parser.add_argument('--epochs', help='Set epochs.', type=int)
parser.add_argument('--gpu', help='Use GPU for training.', action=argparse.BooleanOptionalAction)

args = parser.parse_args()
data_dir = args.data_dir
checkPointSaveDir = args.save_dir
arch = args.arch or 'efficientnet'
learningRate = args.learning_rate or 0.001
hiddenUnits = args.hidden_units
epochs = args.epochs or 1
device = 'cuda' if args.gpu != None and args.gpu else 'cpu'

classifierModelDirectory = ClassifierModelDirectory(data_dir + '/train', data_dir + '/valid', data_dir + '/test')
model = ClassifierModel(arch, 102, classifierModelDirectory, hiddenUnits)
print('model created')
#model.Train(epochs, learningRate)
#model.SaveCheckPoint(dir or '')