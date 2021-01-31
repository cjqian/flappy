import abc
import gym
import random
import numpy as np
import six
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter

import game_lib

class ModelWrapper(six.with_metaclass(abc.ABCMeta, object)):
  def __init__(self, training_data = None, load = True): # 
    if not training_data:
      training_data = np.load(game_lib.TRAINING_DATA_LOC, allow_pickle=True)

    self.model = self._load_model(training_data) if load else self._train_and_save_model(training_data)

  @property
  def save_location(self):
    return 'models/' + self._get_id()

  @abc.abstractmethod
  def _get_id(self):
    pass

  @abc.abstractmethod
  def _initialize_model(self, input_size): # Returns initial model architecture
    pass

  def step(self, observation):
    return np.argmax(self.model.predict(observation.reshape(-1,len(observation),1))[0])

  def _load_model(self, training_data):
    X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)

    model = self._initialize_model(input_size = len(X[0]))
    model.load(self.save_location)
    return model

  def _train_and_save_model(self, training_data):
    X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
    y = [i[1] for i in training_data]

    model = self._initialize_model(input_size = len(X[0]))

    model.fit({'input': X}, {'targets': y}, n_epoch=10, snapshot_step=500, show_metric=True, run_id=self._get_id())
    model.save(self.save_location)

    print('Saved!')
    return model

# Simple neural network 
class DefaultModelWrapper(ModelWrapper):
  def _get_id(self):
    return "default"

  def _initialize_model(self, input_size):
    network = input_data(shape=[None, input_size, 1], name='input')

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=.001, loss='categorical_crossentropy', name='targets')
    dnnm = tflearn.DNN(network, tensorboard_dir='log')

    return dnnm
