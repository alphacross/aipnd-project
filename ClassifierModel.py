from torchvision import datasets, models
from torchvision.transforms import v2
from torch import nn, optim, utils, no_grad, exp, mean, FloatTensor, save, load
from ClassifierModelDirectory import ClassifierModelDirectory
import time

class ClassifierModel:
    def __init__(self, architecture, outputNo, dataDirectory: ClassifierModelDirectory, useGpu, hiddenUnits = None):
        super().__init__()
        self.UseGPU = useGpu
        
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
            
        self.model.class_to_idx = self.__SetDataLoader(dataDirectory)

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
        self.trainingDataLoaders = utils.data.DataLoader(testingDataSets, batch_size = batchSize)

        return trainingDataSets.class_to_idx

    def Train(self, epochs, learningRate):
        device = 'cuda' if self.UseGPU else 'cpu'
        self.model.to(device)

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(params = self.model.classifier.parameters(), lr = learningRate)

        for e in epochs:
            startTime = time()
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

                endTime = time()
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
                images, labels = images.to(device), labels.to(device)

                output = self.model(images)
                loss = criterion(output, labels)
                runningLoss += loss.item()

                ps = exp(output)
                top_ps, top_class = ps.topk(1, dim = 1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += mean(equals.type(FloatTensor)).item()
        
        return runningLoss, accuracy
            
    def SaveCheckPoint(self, dir):
        modelCheckPointName = f'{type(self.model).__name__}_modelCheckPoint'
        
        checkpoint = {
            'state_dict': self.model.state_dict(),
            'class_to_idx': self.model.class_to_idx,
            'classifier': self.model.classifier,
        }
        if(len(dir) > 0):
            modelCheckPointName = dir + '/' + modelCheckPointName
        save(modelCheckPointName, modelCheckPointName)

    def LoadModelCheckpoint(self):
        modelCheckPointName = f'{type(self.model).__name__}_modelCheckPoint'
        modelCheckpoint = load(modelCheckPointName)

        for param in self.model.parameters():
            param.requires_grad = False
            
        self.model.classifier = modelCheckpoint['classifier']
        self.model.load_state_dict(modelCheckpoint['state_dict'])
        self.model.class_to_idx = modelCheckpoint['class_to_idx']