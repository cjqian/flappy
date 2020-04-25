import abc
import collections
import gym
import numpy as np
import six
import statistics
import time

WINNING_SCORE = 100 # This is actually 195 but let's be gentle for now
TIMESTEPS = 100

TRAINING_DATA_LOC = 'data/training.npy'

class GameLog(object):
  def __init__(self):
    self.history = []
    self.score = 0

def play_game(env, agent, render=True, game_number=None): # Returns a game log
  log = GameLog()

  observation = env.reset()
  for t in range(TIMESTEPS): # timesteps
      if render:
        env.render()
      action = agent.step(observation)
      # Observation at t - 1
      log.history.append(
        (observation, agent.encode_discrete_action(action)))
      observation, reward, done, info = env.step(action)
      log.score += reward
      if done:
          print("Episode {game_number} finished after {ts} timesteps; score {sc}".format(game_number=game_number, ts=t+1, sc=log.score))
          break

  return log

def generate_training_data(env, agent, episodes=1000, score_threshold = 50, render=False):
  scores = []
  training_data = []
  for i in range(episodes):
    log = play_game(env, agent, render=render, game_number=i)
    if log.score > score_threshold:
      training_data.extend(log.history)
    scores.append(log.score)

  # Print metrics
  print('\nDone. Mean score: {m}'.format(m=statistics.mean(scores)))
  accepted_scores = [x for x in scores if x > score_threshold]

  a = len(accepted_scores)
  s = len(scores)
  print('Passed: {f} ({t}/{s})'.format(
    f = (a / (s + 0.0)), t = a, s = s))
  print(collections.Counter(accepted_scores))

  # Save training data
  #np.save(TRAINING_DATA_LOC, np.array(training_data))
