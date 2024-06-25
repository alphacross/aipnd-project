from torchvision import datasets, models
from torchvision.transforms import v2
from torch import nn, optim, utils, no_grad, exp, mean, FloatTensor, save, load, from_numpy
from ClassifierModelDirectory import ClassifierModelDirectory
import time
import os.path
from PIL import Image
import numpy as np


class ClassifierModel:
    def __init__(self, checkpointDirectory, useGpu, architecture = 'EfficientNet', outputNo = 102, dataDirectory: ClassifierModelDirectory = None, hiddenUnits = None, isTraining = False):
        self.UseGPU = useGpu
        self.checkpointDirectory = checkpointDirectory

        if architecture == 'vgg13':
            self.SelectedWeights = models.VGG13_Weights.DEFAULT
            self.model = models.vgg13(weights=self.SelectedWeights)
            self.__FreezeWeights()
            self.model.classifier = nn.Sequential([nn.Linear(25088, 4096),
                                                   nn.ReLU(),
                                                   nn.Dropout(.2),
                                                   nn.Linear(4096, hiddenUnits or 2048),
                                                   nn.ReLU(),
                                                   nn.Dropout(.2),
                                                   nn.Linear(hiddenUnits or 2048, outputNo),
                                                   nn.LogSoftmax(dim = 1)])
        else:
            # Use Efficientnet if no model is specified
            self.SelectedWeights = models.EfficientNet_V2_S_Weights.DEFAULT
            self.model = models.efficientnet_v2_s(weights = self.SelectedWeights)
            self.__FreezeWeights()
            self.model.classifier = nn.Sequential(nn.Linear(1280, 640),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(640, hiddenUnits or 256),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(hiddenUnits or 256, outputNo),
                                    nn.LogSoftmax(dim = 1))
        
        modelCheckPointName = f'{type(self.model).__name__}_modelCheckPoint.pth'
        if(self.checkpointDirectory is not None and self.checkpointDirectory != ''):
            modelCheckPointName = self.checkpointDirectory

        checkpointLoaded = self.TryLoadModelCheckpoint(modelCheckPointName)
        if isTraining:
            self.model.class_to_idx = self.__SetDataLoader(dataDirectory)
        elif not isTraining and not checkpointLoaded:
            print("Model is set to evaluation mode but no checkpoint is loaded")

    def __FreezeWeights(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def __SetDataLoader(self, dataDirectory: ClassifierModelDirectory):
        transforms = self.SelectedWeights.transforms()
        traningDataTransforms = v2.Compose([v2.RandomRotation(30),
                                v2.RandomResizedCrop(224),
                                v2.RandomHorizontalFlip(),
                                transforms])
        
        trainingDataSets = datasets.ImageFolder(dataDirectory.trainingDirectory, transform=traningDataTransforms)
        validationDataSets = datasets.ImageFolder(dataDirectory.validationDirectory, transform=transforms)
        testingDataSets = datasets.ImageFolder(dataDirectory.testDirectory, transform=transforms)

        batchSize = 64

        self.trainingDataLoaders = utils.data.DataLoader(trainingDataSets, batch_size = batchSize, shuffle = True)
        self.validationDataLoaders = utils.data.DataLoader(validationDataSets, batch_size = batchSize)
        self.testingDataLoaders = utils.data.DataLoader(testingDataSets, batch_size = batchSize)

        return trainingDataSets.class_to_idx

    def Train(self, epochs, learningRate):
        device = 'cuda' if self.UseGPU else 'cpu'
        self.model.to(device)

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(params = self.model.classifier.parameters(), lr = learningRate)

        print(f'training model "{type(self.model).__name__}", learn rate: {learningRate}, on "{next(self.model.parameters()).device.type}"')
        for e in range(epochs):
            startTime = time.time()
            print(f"Begin training {e + 1}/{epochs}.")
            self.model.train()
            
            trainingLoss = 0

            for i, (images, labels) in enumerate(self.trainingDataLoaders):
                print(f"\rprocessing: {i + 1}/{len(self.trainingDataLoaders)}", end="", flush=True)
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()

                output = self.model(images)
                loss = criterion(output, labels)
                trainingLoss += loss.item()

                loss.backward()
                optimizer.step()
            
            else:
                print()
                validationLoss, accuracy = self.Validate(self.validationDataLoaders)
                endTime = time.time()
                elapsedTime = endTime - startTime
                print(f'Elapsed {int(elapsedTime // 60)}:{int(elapsedTime % 60)}')
            
                print("Epoch: {}/{}.. ".format(e + 1, epochs),
                    "Training Loss: {:.3f}.. ".format(trainingLoss / len(self.trainingDataLoaders)),
                    "Validation Loss: {:.3f}.. ".format(validationLoss / len(self.validationDataLoaders)),
                    "Validation Accuracy: {:.3f}%".format(accuracy / len(self.validationDataLoaders) * 100))
                print()

    def Validate(self, dataLoaders):
        runningLoss = 0
        accuracy = 0
        device = 'cuda' if self.UseGPU else 'cpu'
        criterion = nn.NLLLoss()

        with no_grad():
            self.model.eval()
            self.model.to(device)
            for i, (images, labels) in enumerate(dataLoaders):
                print(f"\rprocessing: {i + 1}/{len(dataLoaders)}", end="", flush=True)
                images, labels = images.to(device), labels.to(device)

                output = self.model(images)
                loss = criterion(output, labels)
                runningLoss += loss.item()

                ps = exp(output)
                top_ps, top_class = ps.topk(1, dim = 1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += mean(equals.type(FloatTensor)).item()
        print()
        return runningLoss, accuracy
            
    def SaveCheckPoint(self):
        modelCheckPointName = f'{type(self.model).__name__}_modelCheckPoint.pth'
        
        checkpoint = {
            'state_dict': self.model.state_dict(),
            'class_to_idx': self.model.class_to_idx,
            'classifier': self.model.classifier,
        }
        if(self.checkpointDirectory is not None and self.checkpointDirectory != ''):
            modelCheckPointName = dir + '/' + modelCheckPointName
        save(checkpoint, modelCheckPointName)

    def TryLoadModelCheckpoint(self, modelCheckPointName):
        if not os.path.isfile(modelCheckPointName):
            print('No Checkpoint found')
            return False

        modelCheckpoint = load(modelCheckPointName)
        print('Checkpoint loaded')

        self.__FreezeWeights()

        self.model.classifier = modelCheckpoint['classifier']
        self.model.load_state_dict(modelCheckpoint['state_dict'])
        self.model.class_to_idx = modelCheckpoint['class_to_idx']

        return True

    def __ProcessImage(self, image):
        print('Process image.')

        pImage = Image.open(image)

        width, height = pImage.size

        aspectRatio = width / height
        
        if width <= height:
            newWidth = 256
            newHeight = int(height / aspectRatio)
        else:
            newHeight = 256
            newWidth = int(256 * aspectRatio)

        resizedImage = pImage.resize((newWidth, newHeight))
        
        # center crop
        cropSize = 224
        width, height = resizedImage.size
        
        left = (width - cropSize) / 2
        top = (height - cropSize) / 2
        right = (width + cropSize) / 2
        bottom = (height + cropSize) / 2

        croppedImage = resizedImage.crop((left, top, right, bottom))

        npImage = np.array(croppedImage).astype(np.float32) / 255

        npImage -= np.array([0.485, 0.456, 0.406])
        npImage /= np.array([0.229, 0.224, 0.225])

        npImage = npImage.transpose((2, 0, 1))

        return npImage

    def Predict(self, image_path, topk=5):
        img = self.__ProcessImage(image_path)
        img = from_numpy(img)
        img = img.unsqueeze(0)
        
        device = 'cuda' if self.UseGPU else 'cpu'

        print('Begin predict')
        with no_grad():
            self.model.eval()
            self.model.to(device)
            img = img.to(device)

            output = self.model(img)

            ps = exp(output)
            prob, indeces = ps.topk(topk, dim=1)

            prob, indeces = prob.cpu(), indeces.cpu()

            prob = prob.numpy()[0].tolist()
            indeces = indeces.numpy()[0].tolist()
            
            invertDict = {v: k for k, v in self.model.class_to_idx.items()}
            classes = [invertDict[i] for i in indeces]

            return prob, classes