class ClassifierModel():
    def __init__(self, trainingDir: str, validationDir: str, testDir: str):
        self.trainingDirectory = trainingDir
        self.validationDirectory = validationDir
        self.testDirectory = testDir