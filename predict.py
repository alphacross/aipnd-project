from ClassifierModel import ClassifierModel
from ClassifierModelDirectory import ClassifierModelDirectory
import argparse
import json
import os.path

parser = argparse.ArgumentParser(description = 'Flower classification prediction.')
parser.add_argument('imageDir', help="Image directory.")
parser.add_argument('checkpoint', help="Checkpoint directory.")
parser.add_argument('--top_k', help='Return top K most likely class', type=int)
parser.add_argument('--category_names', help='Use a mapping of categories to real names')
parser.add_argument('--gpu', help='Use GPU for inference.', action=argparse.BooleanOptionalAction)

args = parser.parse_args()
imageDirectory = args.imageDir
checkpointDirectory = args.checkpoint
topKCount = args.top_k or 1
catNames = args.category_names or 'cat_to_name.json'

if not os.path.isfile(imageDirectory):
    print(f'{imageDirectory} not found. Cancelling prediction')

if not os.path.isfile(checkpointDirectory):
    print(f'{checkpointDirectory} not found. Cancelling prediction')

if not os.path.isfile(catNames):
    print(f'{catNames} not found. Cancelling prediction')

if not os.path.isfile(imageDirectory):
    print(f'{imageDirectory} not found. Cancelling prediction')

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

data_dir = 'flowers'
classifierModelDirectory = ClassifierModelDirectory(data_dir + '/train', data_dir + '/valid', data_dir + '/test')

model = ClassifierModel(checkpointDirectory=checkpointDirectory, 
                        dataDirectory= classifierModelDirectory,
                        useGpu= args.gpu != None and args.gpu)

prob, classes = model.Predict(imageDirectory, topKCount)
print(prob)
print(classes)

className = [cat_to_name[i] for i in classes]

print()
print('Results:')
print()
for i, c in enumerate(className):
    print(f'{c}: {prob[i] * 100}%')