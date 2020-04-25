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

class Model(six.with_metaclass(abc.ABCMeta, object)):
  @property
  def save_location(self):
    return 'models/' + self._get_id()
  
  @abc.abstractmethod
  def _get_id(self):
    pass

  @abc.abstractmethod
  def model(self, input_size):
    pass

  def load(self, training_data= None):
    if not training_data:
      training_data = np.load(game_lib.TRAINING_DATA_LOC, allow_pickle=True)

    X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)

    return self.model(input_size = len(X[0])).load(self.save_location)

  def train_and_save(self, training_data = None):
    if not training_data:
      training_data = np.load(game_lib.TRAINING_DATA_LOC, allow_pickle=True)

    X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
    y = [i[1] for i in training_data]

    m = self.model(input_size = len(X[0]))

    m.fit({'input': X}, {'targets': y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id=self._get_id())
    m.save(self.save_location)

    print('Saved')
    return m

# Simple neural network 
class DefaultModel(Model):
  def _get_id(self):
    return "default"

  def model(self, input_size):
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
