# A define of abstract Methods class for RL algorithms
from abc import ABC, abstractmethod
from gymnasium import Env
import torch

class RLMethod(ABC):
    
    # must implement methods:
    @abstractmethod
    def __init__(self, state_dim: int, action_dim: int, max_action: float, actor=None, critic=None, params=None):
        pass

    @abstractmethod
    def select_action(self, state):
        pass


    @abstractmethod
    def update(self, states ,batch_size: int = 256):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass