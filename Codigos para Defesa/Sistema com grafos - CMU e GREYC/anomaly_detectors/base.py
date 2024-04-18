from abc import ABC, abstractmethod

class UserModel(ABC):
    """ Base class for user_models classification algorithms. In case of adaptive system, features must be adapted."""
    def __init__(self,name,features):
        self.name=name
        self.features=features
        super().__init__()

    @abstractmethod
    def update(self, model, features):
        pass

class ClassificationAlgorithm(ABC):
    """ Base class classification algorithms of keystroke dynamics. """
    def __init__(self, name, parameters=dict()):
        self.name = name
        self.parameters = parameters
        super().__init__()
    
    @abstractmethod
    def train(self, data):
        pass
    
    @abstractmethod
    def test(self, data):
        pass
