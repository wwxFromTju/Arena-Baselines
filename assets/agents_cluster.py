import torch
import time
import numpy as np
import os
import scipy


class MultiAgentCluster(object):
    """docstring for MultiAgentCluster."""

    def __init__(self, agents, learning_agent_id, store_interval, log_dir,
                 reload_agents_interval, reload_playing_agents_principle, tf_summary):
        super(MultiAgentCluster, self).__init__()
        self.all_agents = agents
        self.learning_agent_id = learning_agent_id
        self.learning_agents = self.all_agents[self.learning_agent_id:self.learning_agent_id + 1]
        self.playing_agents = self.all_agents[:self.learning_agent_id] + \
            self.all_agents[self.learning_agent_id + 1:]
        self.num_agents = len(self.all_agents)
        self.log_dir = log_dir
        self.tf_summary = tf_summary

        self.store_interval = store_interval
        self.last_time_store = time.time()

        self.reload_agents_interval = reload_agents_interval
        self.last_time_reload_agents = time.time()
        self.reload_playing_agents_principle = reload_playing_agents_principle

        self.checkpoints_reward_record = None
        self.checkpoints_reward_record_FRAME = 0
        self.checkpoints_reward_record_REWARD = 1

        if self.reload_playing_agents_principle in ['prioritized']:
            if len(self.learning_agents) == 1 and len(self.playing_agents) == 1:
                self.checkpoints_reward_record = 'empty'
            else:
                input('# WARNING: reload_playing_agents_principle of prioritized is only support for one playing_agent and one learning_agent')

    def reset(self, obs):
        for agent in self.all_agents:
            agent.reset(obs[:, agent.id])

    def before_rollout(self):
        for agent in self.learning_agents:
            agent.schedule_trainer()

    def experience_not_enough(self):
        '''currently only support one learning_agent'''
        assert len(self.learning_agents) == 1

        return self.learning_agents[0].experience_not_enough()

    def act(self, obs, deterministic):
        action = []
        for agent in self.all_agents:
            action += [agent.act(
                obs=obs[:, agent.id],
                deterministic=deterministic,
            ).unsqueeze(1)]
        return torch.cat(action, 1)

    def observe(self, obs, reward, done, infos):
        for agent in self.all_agents:
            agent.observe(
                obs[:, agent.id], reward[:, agent.id], done[:, agent.id], infos,
            )

    def store(self):
        '''
            store training status
        '''
        print('# INFO: Storing MultiAgentCluster')

        '''store learning_agents'''
        for agent in self.learning_agents:
            agent.store()

        if self.checkpoints_reward_record is not None:

            new_record = np.array(
                [[self.learning_agents[0].current_checkpoint_by_frame, 0.0]])
            if self.checkpoints_reward_record is 'empty':
                self.checkpoints_reward_record = new_record
            else:
                self.checkpoints_reward_record = np.concatenate(
                    (
                        self.checkpoints_reward_record,
                        new_record
                    ),
                    axis=0,
                )
            print('# INFO: Append new_record {} to checkpoints_reward_record (shape of {})'.format(
                new_record,
                self.checkpoints_reward_record.shape,
            ))

            try:
                '''store checkpoints_reward_record'''
                np.save(
                    os.path.join(
                        self.log_dir, "checkpoints_reward_record.npy"),
                    self.checkpoints_reward_record,
                )
                print('# INFO: Store checkpoints_reward_record ok: {}'.format(
                    self.checkpoints_reward_record.shape))
            except Exception as e:
                print('# WARNING: Store checkpoints_reward_record failed: {}'.format(e))

    def restore(self):
        '''
            restore training status
        '''
        print('# INFO: Restoring MultiAgentCluster')

        '''restore learning_agents'''
        for agent in self.learning_agents:
            agent.randomlize_population_id()
            agent.restore(principle='recent')

        if self.checkpoints_reward_record is not None:
            try:
                '''restore checkpoints_reward_record'''
                self.checkpoints_reward_record = np.load(
                    os.path.join(
                        self.log_dir, "checkpoints_reward_record.npy"),
                )
                print('# INFO: Restore checkpoints_reward_record ok, shape {}.'.format(
                    self.checkpoints_reward_record.shape,
                ))
            except Exception as e:
                print(
                    '# WARNING: No checkpoint for checkpoints_reward_record found.')

    def reload_playing_agents(self):
        '''reload playing agent will restore a new checkpoint'''

        for agent in self.playing_agents:
            agent.randomlize_population_id()

        if self.reload_playing_agents_principle in ['prioritized']:

            if ((len(self.learning_agents) != 1) or (len(self.playing_agents) != 1) or (self.playing_agents[0].population_number != 1)):
                input(
                    '# ACTION REQUIRED: prioritized only supports two player game with population_number=1')

            '''update current'''
            current_checkpoint_by_index = np.where(
                self.checkpoints_reward_record == self.playing_agents[0].current_checkpoint_by_frame)[0][0]
            record_update_target = self.learning_agents[0].episode_scaler_summary.summary(
                mode='mean')['raw']
            print('# INFO: checkpoints_reward_record of {} is being updated from {} to {} (from {} episodes)'.format(
                self.playing_agents[0].current_checkpoint_by_frame,
                self.checkpoints_reward_record[current_checkpoint_by_index,
                                               self.checkpoints_reward_record_REWARD],
                record_update_target,
                len(self.learning_agents[0].episode_scaler_summary.final_rewards['raw']),
            ))
            self.tf_summary.add_scalar(
                '{}/{}'.format('global', 'record_improvement'),
                (record_update_target -
                 self.checkpoints_reward_record[current_checkpoint_by_index, self.checkpoints_reward_record_REWARD]),
                self.learning_agents[0].get_num_trained_frames(),
            )
            self.checkpoints_reward_record[current_checkpoint_by_index,
                                           self.checkpoints_reward_record_REWARD] = record_update_target
            self.learning_agents[0].episode_scaler_summary.reset()

            '''reload new'''
            probability_to_checkpoints = scipy.special.softmax(
                -self.checkpoints_reward_record[:,
                                                self.checkpoints_reward_record_REWARD],
                axis=0,
            )
            checkpoint_picked_by_frame = np.random.choice(
                self.checkpoints_reward_record[:,
                                               self.checkpoints_reward_record_FRAME],
                p=probability_to_checkpoints,
            )
            checkpoint_picked_by_frame = int(checkpoint_picked_by_frame)
            checkpoint_picked_by_index = np.where(
                self.checkpoints_reward_record == checkpoint_picked_by_frame)[0][0]
            print('# INFO: Pick checkpoint {} from {} records, the probability this pick is {} (uniform is {}), the reward recorded is {}'.format(
                checkpoint_picked_by_frame,
                self.checkpoints_reward_record.shape[0],
                probability_to_checkpoints[checkpoint_picked_by_index],
                1.0 / self.checkpoints_reward_record.shape[0],
                self.checkpoints_reward_record[checkpoint_picked_by_index,
                                               self.checkpoints_reward_record_REWARD]
            ))

            self.playing_agents[0].restore(
                principle='{}_frame'.format(
                    checkpoint_picked_by_frame
                )
            )

        else:
            for agent in self.playing_agents:
                agent.restore(principle=self.reload_playing_agents_principle)

    def reload_learning_agents(self):
        '''reload learning agent will store agent first and then restore it, during which population id is re-generated'''
        for agent in self.learning_agents:
            agent.store()
            agent.randomlize_population_id()
            agent.restore(principle='recent')

    def after_rollout(self):
        '''for all agents'''
        for agent in self.all_agents:
            agent.after_rollout()

        '''reload agents'''
        if ((time.time() - self.last_time_reload_agents) > self.reload_agents_interval):
            self.last_time_reload_agents = time.time()
            self.reload_playing_agents()
            self.reload_learning_agents()
            # episode_scaler_summary is reset here because reload agent function under some options use episode_scaler_summary
            for agent in self.all_agents:
                agent.episode_scaler_summary.reset()

        '''for all'''
        if ((time.time() - self.last_time_store) > self.store_interval):
            self.last_time_store = time.time()
            self.store()
