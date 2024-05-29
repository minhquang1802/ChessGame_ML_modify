# config.py
class Initial:
    def __init__(self, mode=0, difficulty=0):
        self.mode = mode
        self.difficulty = difficulty

    def setMode(self, mode):
        self.mode = mode

    def setDifficulty(self, difficulty):
        self.difficulty = difficulty

    def getMode(self):
        return self.mode

    def getDifficulty(self):
        return self.difficulty


initial = Initial()