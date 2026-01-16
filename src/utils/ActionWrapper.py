from abc import ABC, abstractmethod

class PolicyNoiseActionWrapper(ABC):
    """
    用于在策略中添加噪声的抽象基类。
    a base class for adding noise to actions in a policy.
    """
    @abstractmethod
    def __call__(self, action):
        """
        向动作添加噪声。
        add noise to the action.
        """
        pass
    
import numpy as np
  
class EpsilonGreedyActionWrapper(PolicyNoiseActionWrapper):
    """
    用于在策略中添加ε-贪婪噪声的类。

    a base class for adding epsilon-greedy noise to actions in a policy.
    """
    def __init__(self, epsilon: float, action_dim: int, action_space):
        """
        初始化ε-贪婪动作包装器。
        initialize the epsilon-greedy action wrapper.
        :param epsilon: ε值，表示选择随机动作的概率。
                        the epsilon value representing the probability of choosing a random action.
        :param action_dim: 动作的维度。
                           the dimension of the action.
        """
        self.epsilon = epsilon
        self.action_dim = action_dim
        self.action_space = action_space
    def __call__(self, action_idx: np.ndarray) -> np.ndarray:
        """
        向动作添加ε-贪婪噪声。
        add epsilon-greedy noise to the action.
        """
        self.epsilon = max(0.01, self.epsilon * 0.995)

        if np.random.rand() < self.epsilon:
            action_idx = np.random.randint(0, self.action_space.n)
            # decay epsilon
        return action_idx

class GaussianNoiseActionWrapper(PolicyNoiseActionWrapper):
    """
    用于在策略中添加高斯噪声的类。
    a class for adding Gaussian noise to actions in a policy.
    """
    def __init__(self, mu: float, sigma: float, action_dim: int):
        """
        初始化高斯噪声动作包装器。
        initialize the Gaussian noise action wrapper.
        :param mu: 高斯噪声的均值。
                    the mean of the Gaussian noise.
        :param sigma: 高斯噪声的标准差。
                       the standard deviation of the Gaussian noise.
        :param action_dim: 动作的维度。
                           the dimension of the action.
        """
        self.mu = mu
        self.sigma = sigma
        self.action_dim = action_dim
    def __call__(self, action: np.ndarray) -> np.ndarray:
        """
        向动作索引添加高斯噪声。
        add Gaussian noise to the action index.
        """
        noise = np.random.normal(self.mu, self.sigma, size=self.action_dim)
        noisy_action = action + noise
        return noisy_action