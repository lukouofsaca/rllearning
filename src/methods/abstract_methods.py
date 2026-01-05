# A define of abstract Methods class for RL algorithms
from abc import ABC, abstractmethod
import torch


class ABSRLMethod(ABC):
    @abstractmethod
    # variable parameters can be added as needed
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        pass
    
    
class RLMethod(ABSRLMethod):
    
    @abstractmethod
    def __init__(self, state_dim: int, action_dim: int, max_action: float, params=None):
        pass

    @abstractmethod
    def select_action(self, state):
        """
        Select action given state.
        Returns: action, logprob, value (logprob and value can be None or dummy for off-policy)
        """
        pass

    @abstractmethod
    def store_transition(self, transition):
        """
        Store transition in buffer.
        transition: (state, next_state, action, action_logprob, reward, done, value)
        """
        pass

    @abstractmethod
    def update(self):
        """
        Update the agent.
        """
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass

class OnPolicyMethod(RLMethod):
    @abstractmethod
    def __init__(self, state_dim: int, action_dim: int, max_action: float, params=None):
        super(OnPolicyMethod, self).__init__(state_dim, action_dim, max_action, params)