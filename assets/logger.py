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
        '''call this at step'''
        if set(key_value.keys()) == set((dict((el, 0) for el in self.keys)).keys()):
            for key in self.keys:
                self.episode_reward[key] += key_value[key]
        else:
            raise Exception(
                'key_value dic must match the keys you set when create this EpisodeScalerSummary.')

    def at_done(self):
        '''call this at episode done'''
        for key in self.keys:
            try:
                self.final_rewards[key] += [self.episode_reward[key]]
            except Exception as e:
                self.final_rewards[key] = [self.episode_reward[key]]
            self.episode_reward[key] = 0.0

    def to_print_str(self):
        '''get a print string of final_rewards'''
        print_str = ''

        for key in self.keys:
            if len(self.final_rewards[key]) > 0:
                print_str += '[{} - {:.2f}|{:.2f}|{:.2f}|{} (Min|Mean|Max|Num)]'.format(
                    key,
                    np.min(self.final_rewards[key]),
                    np.mean(self.final_rewards[key]),
                    np.max(self.final_rewards[key]),
                    len(self.final_rewards[key]),
                )
            else:
                print_str += '[{}-still summarizing the first episode]'.format(
                    key,
                )

        return print_str

    def get_length(self):
        '''get the length of summaried final_rewards'''
        return len(self.final_rewards[list(self.final_rewards.keys())[0]])

    def summary(self, mode='mean'):
        '''
            get final_rewards summary
            mode: min, mean, max, recent
        '''
        final_rewards_mean = {}
        for key in self.keys:
            if mode in ['min']:
                final_rewards_mean[key] = np.min(self.final_rewards[key])
            elif mode in ['mean']:
                final_rewards_mean[key] = np.mean(self.final_rewards[key])
            elif mode in ['max']:
                final_rewards_mean[key] = np.max(self.final_rewards[key])
            elif mode in ['recent']:
                final_rewards_mean[key] = self.final_rewards[key][-1]
        return final_rewards_mean

    def reset(self):
        '''reset summary'''
        for key in self.keys:
            self.episode_reward[key] = 0.0
            self.final_rewards[key] = []
