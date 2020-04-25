import abc
import collections
import gym
import numpy as np
import six
import statistics
import time

WINNING_SCORE = 195
TIMESTEPS = 500

TRAINING_DATA_LOC = 'data/training.npy'

# Would be nice to pass in an ID flag and store multiple models.
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

def play(env, agent, episodes=1000, score_threshold = 50, render=False, save_training_data=False):
  scores = []
  training_data = []
  for i in range(episodes):
    log = play_game(env, agent, render=render, game_number=i)
    if log.score > score_threshold:
      training_data.extend(log.history)
    scores.append(log.score)

  # Print metrics
  print('\nDone. Mean score: {m} +- {s}'.format(m=statistics.mean(scores), s=statistics.stdev(scores)))

  if save_training_data:
    accepted_scores = [x for x in scores if x > score_threshold]
    print(collections.Counter(accepted_scores))

    a = len(accepted_scores)
    s = len(scores)

    print('Passed: {f} ({t}/{s})'.format(
      f = (a / (s + 0.0)), t = a, s = s))
    print(collections.Counter(accepted_scores))

    np.save(TRAINING_DATA_LOC, np.array(training_data))
    print('Saved training data.')
  else:
    print(collections.Counter(scores))
    if statistics.mean(scores) > WINNING_SCORE:
      print('YOU WIN!')
    else:
      print('YOU LOSE')

