import abc
import collections
import gym
import numpy as np
import six
import statistics
import tensorflow as tf
import time

import models

class Agent(six.with_metaclass(abc.ABCMeta, object)):
  def __init__(self, env):
    self._env = env  # environment

  # Returns one-hot encoding; list where ath value is 1
  def encode_discrete_action(self, a):
    l = [0] * self._env.action_space.n
    l[a] = 1
    return l

  @abc.abstractmethod
  def step(self, observation):
    pass

class RandomAgent(Agent):
  def step(self, observation):
    return self._env.action_space.sample()

class DefaultAgent(Agent):
  def __init__(self, env):
    self.model_wrapper = models.DefaultModelWrapper(load=True)
    super(DefaultAgent, self).__init__(env)

  def step(self, observation):
    return self.model_wrapper.step(observation)
