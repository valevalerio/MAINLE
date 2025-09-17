from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    def execute(self, *args, **kwargs):
        pass

    @abstractmethod
    def history(self):
        pass
