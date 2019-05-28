import torch
import tensorflow as tf
import os
import numpy as np
import cv2


class EpisodeScalerSummary(object):
    """docstring for EpisodeScalerSummary."""

    def __init__(self, keys):
        self.keys = keys
        self.episode_reward = {}
        self.final_rewards = {}
        self.reset()

    def at_step(self, key_value):
        if set(key_value.keys()) == set((dict((el, 0) for el in self.keys)).keys()):
            for key in self.keys:
                self.episode_reward[key] += key_value[key]
        else:
            raise Exception('Must match!')

    def at_done(self):
        for key in self.keys:
            try:
                self.final_rewards[key] += [self.episode_reward[key]]
            except Exception as e:
                self.final_rewards[key] = [self.episode_reward[key]]
            self.episode_reward[key] = 0.0

    def to_print_str(self):
        print_str = ''
        for key in self.keys:
            if len(self.final_rewards[key]) > 0:
                print_str += '[{}-{:.2f}:{:.2f}:{:.2f}:{}(Min:Mean:Max:Num)]'.format(
                    key,
                    np.min(self.final_rewards[key]),
                    np.mean(self.final_rewards[key]),
                    np.max(self.final_rewards[key]),
                    len(self.final_rewards[key]),
                )
            else:
                print_str += '[{}-summarizing-{}]'.format(
                    key,
                    self.episode_reward[key],
                )
        return print_str

    def get_length(self):
        return len(self.final_rewards[list(self.final_rewards.keys())[0]])

    def to_mean(self):
        final_rewards_mean = {}
        for key in self.keys:
            final_rewards_mean[key] = np.mean(self.final_rewards[key])
        return final_rewards_mean

    def to_recent(self):
        final_rewards_mean = {}
        for key in self.keys:
            final_rewards_mean[key] = self.final_rewards[key][-1]
        return final_rewards_mean

    def to_mean_and_reset(self):
        final_rewards_mean = self.to_mean()
        self.reset()
        return final_rewards_mean

    def reset(self):
        for key in self.keys:
            self.episode_reward[key] = 0.0
            self.final_rewards[key] = []
