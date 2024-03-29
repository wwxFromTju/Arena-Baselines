import time
import os
import torch
import numpy as np
from .utils import update_linear_schedule
from .logger import EpisodeScalerSummary


class Agent(object):
    """docstring for Agent."""

    def __init__(self, id, mode, envs, recurrent_brain, num_processes, num_steps, use_linear_lr_decay, use_linear_clip_decay, use_gae, gamma, tau, num_env_steps, num_updates, log_dir, tf_summary, cuda, device,
                 trainer_id, value_loss_coef, entropy_coef, lr, eps, alpha, max_grad_norm, clip_param, ppo_epoch, num_mini_batch, population_number):
        super(Agent, self).__init__()

        '''basic settings and references'''
        self.id = id
        self.mode = mode
        self.envs = envs
        self.num_processes = num_processes
        self.num_steps = num_steps
        self.num_env_steps = num_env_steps
        self.num_updates = num_updates
        self.log_dir = log_dir
        self.tf_summary = tf_summary
        self.cuda = cuda
        self.device = device

        '''brain settings'''
        self.recurrent_brain = recurrent_brain

        '''trainer settings'''
        self.trainer_id = trainer_id
        self.lr = lr
        self.clip_param = clip_param
        self.use_linear_lr_decay = use_linear_lr_decay
        self.use_linear_clip_decay = use_linear_clip_decay
        self.use_gae = use_gae
        self.gamma = gamma
        self.tau = tau

        '''multi-agent settings'''
        self.population_number = population_number
        self.population_id = None

        self.brain = self.build_brain()

        if self.mode in ['learning']:
            '''build trainer'''
            if self.trainer_id == 'a2c':
                from .trainers import A2C_ACKTR
                self.trainer = A2C_ACKTR(self.brain, value_loss_coef,
                                         entropy_coef, lr=self.lr,
                                         eps=eps, alpha=alpha,
                                         max_grad_norm=max_grad_norm)
            elif self.trainer_id == 'ppo':
                from .trainers import PPO
                self.trainer = PPO(self.brain, self.clip_param, ppo_epoch, num_mini_batch,
                                   value_loss_coef, entropy_coef, lr=self.lr,
                                   eps=eps,
                                   max_grad_norm=max_grad_norm)
            elif self.trainer_id == 'acktr':
                from .trainers import A2C_ACKTR
                self.trainer = A2C_ACKTR(self.brain, value_loss_coef,
                                         entropy_coef, acktr=True)

        '''build rollouts'''
        from assets.storage import RolloutStorage
        self.rollouts = RolloutStorage(self.num_steps, self.num_processes,
                                       self.envs.observation_space.shape,
                                       self.envs.action_space,
                                       self.brain.recurrent_hidden_state_size,
                                       save_cuda_mem=True
                                       ).to(self.device)

        '''initialize training progress'''
        if self.mode in ['learning']:
            self.update_i = 0
            self.num_trained_frames_start = self.get_num_trained_frames()

        self.step_i = 0
        self.episode_scaler_summary = EpisodeScalerSummary(['raw', 'len'])
        self.time_start = time.time()
        self.current_checkpoint_by_frame = 0

        '''prepare for test_obs'''
        if len(self.envs.observation_space.shape) == 1:
            print(
                '# INFO: ram obs does not supporte test_obs, no video will be logged to tensorboard')
            self.test_obs = False
        else:
            # visual obs support test_obs, this will log the video of first episode since this run to tensorboard
            self.test_obs = True

        self.obs_video = None

    def build_brain(self):
        '''build brain'''
        base_kwargs = {'recurrent': self.recurrent_brain}
        if len(self.envs.observation_space.shape) == 1:
            # for ram obs, the hidden_size should be same as the obs size, but not smaller than 64
            base_kwargs['hidden_size'] = max(
                int(self.envs.observation_space.shape[0]), 64)

        from .brains import Policy
        return Policy(self.envs.observation_space.shape, self.envs.action_space,
                      base_kwargs=base_kwargs).to(self.device)

    def get_log_tag(self):
        return 'Agent {} Population {}'.format(self.id, self.population_id)

    def randomlize_population_id(self):
        '''randomlize the population id'''
        self.previous_population_id = self.population_id
        self.population_id = np.random.randint(self.population_number)

        print('# INFO: [{}][Population_id updated from {} to {}]'.format(
            self.get_log_tag(),
            self.previous_population_id,
            self.population_id,
        ))

    def reset(self, obs):
        self.rollouts.obs[0].copy_(obs)

    def schedule_trainer(self):

        if self.mode in ['learning']:
            '''update schedule'''
            if self.use_linear_lr_decay:
                '''decrease learning rate linearly'''
                if self.trainer_id == "acktr":
                    '''use optimizer's learning rate since it's hard-coded in kfac.py'''
                    update_linear_schedule(
                        self.trainer.optimizer, self.update_i, self.num_updates, self.trainer.optimizer.lr)
                else:
                    update_linear_schedule(
                        self.trainer.optimizer, self.update_i, self.num_updates, self.lr)

            '''clip parameters'''
            if self.trainer_id == 'ppo' and self.use_linear_clip_decay:
                self.trainer.clip_param = self.clip_param * \
                    (1 - self.update_i / float(self.num_updates))

        else:
            input('# ACTION REQUIRED: only learning agent can call schedule_trainer()')

    def act(self, obs, deterministic):

        if self.mode in ['playing']:
            # playing agent are always act deterministic
            deterministic = True

        self.test_obs_at_act(obs)

        with torch.no_grad():
            obs, recurrent_hidden_states, masks = self.rollouts.get_policy_inputs(
                self.step_i)
            self.value, self.action, self.action_log_prob, self.recurrent_hidden_states = self.brain.act(
                inputs=obs,
                rnn_hxs=recurrent_hidden_states,
                masks=masks,
                deterministic=deterministic,
            )
        return self.action

    def observe(self, obs, reward, done, infos):
        masks = torch.FloatTensor(
            [[0.0] if done_ else [1.0] for done_ in done])

        '''store experiences'''
        self.rollouts.insert(obs, self.recurrent_hidden_states, self.action,
                             self.action_log_prob, self.value, reward, masks)

        '''log'''
        self.episode_scaler_summary.at_step({
            'raw': reward[0].squeeze().item(),
            'len': 1.0,
        })

        if done[0]:
            self.episode_scaler_summary.at_done()
            self.test_obs_at_done()

        self.step_i += 1

    def test_obs_at_act(self, obs):
        if self.test_obs:
            # [thread, stack, H, W] --> (T, H, W)
            obs_frame = obs[0, -1, :, :].unsqueeze(0)
            if self.obs_video is None:
                self.obs_video = obs_frame
            else:
                self.obs_video = torch.cat([self.obs_video, obs_frame], 0)

    def test_obs_at_done(self):
        if self.test_obs:
            self.test_obs = False
            # (T, H, W) --> vid_tensor: :math:`(N, T, C, H, W)`.
            self.obs_video = self.obs_video.unsqueeze(0).unsqueeze(2)
            self.tf_summary.add_video(
                'test_obs/agent_{}'.format(self.id),
                self.obs_video,
            )
            print('# INFO: [Agent {} Population {}] test obs done'.format(
                self.id, self.population_id, self.population_id))

    def experience_not_enough(self):
        return (self.step_i < self.num_steps)

    def after_rollout(self):
        '''called after a rollout'''

        if not self.experience_not_enough():
            self.step_i = 0

        if self.mode in ['learning']:
            self.update_brain()

    def update_brain(self):

        if self.mode in ['learning']:
            '''update brain from rollouts'''

            '''prepare for update brain'''
            with torch.no_grad():
                obs, recurrent_hidden_states, masks = self.rollouts.get_policy_inputs(
                    -1)
                next_value = self.brain.get_value(
                    inputs=obs,
                    rnn_hxs=recurrent_hidden_states,
                    masks=masks,
                ).detach()
            self.rollouts.compute_returns(
                next_value, self.use_gae, self.gamma, self.tau)

            '''update brain'''
            value_loss, action_loss, dist_entropy = self.trainer.update(
                self.rollouts)

            self.rollouts.after_update()

            '''log info by print'''
            print(self.to_print_str())

            '''vis curves'''
            self.vis_curves()

            self.update_i += 1
            if self.update_i == self.num_updates:
                input('# ACTION REQUIRED: Train end.')

        else:
            input('# ACTION REQUIRED: only learning agent can call update_brain()')

    def get_num_trained_frames(self):
        if self.mode in ['learning']:
            return self.update_i * self.num_processes * self.num_steps
        else:
            input(
                '# ACTION REQUIRED: only learning agent can call get_num_trained_frames()')

    def to_print_str(self):

        print_str = ''

        '''basic info'''
        print_str += "# INFO: [{}][{}]".format(
            self.get_log_tag(),
            self.mode,
        )

        '''learning info'''
        if self.mode in ['learning']:
            end = time.time()
            FPS = ((self.get_num_trained_frames() + self.num_processes * self.num_steps) -
                   self.num_trained_frames_start) / (time.time() - self.time_start)
            print_str += "[{}/{} Updates][{}/{} Frames][FPS {}][Remain {:.2f} hrs]".format(
                self.update_i, self.num_updates,
                self.get_num_trained_frames(), self.num_env_steps,
                int(FPS),
                ((self.num_env_steps - self.get_num_trained_frames()) / FPS / 60.0 / 60.0),
            )

        '''episode_scaler_summary info'''
        print_str += self.episode_scaler_summary.to_print_str()

        return print_str

    def vis_curves(self):

        if self.mode in ['learning']:
            '''update curves'''
            if self.episode_scaler_summary.get_length() > 0:
                for summary_mode in ['min', 'mean', 'max', 'recent']:
                    tmp = self.episode_scaler_summary.summary(
                        mode=summary_mode)
                    for key in tmp.keys():
                        self.tf_summary.add_scalar(
                            '{}/{}_{}'.format(self.mode, key, summary_mode),
                            tmp[key],
                            self.get_num_trained_frames(),
                        )

        else:
            input('# ACTION REQUIRED: only learning agent can call vis_curves()')

    def store(self):
        try:
            checkpoint = self.get_num_trained_frames()
            self.store_to_checkpoint(checkpoint)
            self.current_checkpoint_by_frame = checkpoint
            print('# INFO: [{}][Store to checkpoint {} ok]'.format(
                self.get_log_tag(), checkpoint))

        except Exception as e:
            print('# WARNING: [{}][Store failed: {}]'.format(
                self.get_log_tag(), e))

    def restore(self, principle='recent'):
        '''
            principle:
                recent: the most recent checkpoint,
                uniform: uniformly sampled from all historical checkpoints
                XX_th: the XX-th checkpoint
        '''
        try:
            possible_checkpoints = self.get_possible_checkpoints()
            checkpoint = self.get_checkpoint(possible_checkpoints, principle)
            self.restore_from_checkpoint(checkpoint)
            self.current_checkpoint_by_frame = checkpoint
            print('# INFO: [{}][Restore checkpoint {} ok from {} possible checkpoints, following principle of {}]'.format(
                self.get_log_tag(), checkpoint, len(possible_checkpoints), principle))

        except Exception as e:
            print(
                '# WARNING: [{}][Restore failed: {}][reinitialize agent and store]'.format(self.get_log_tag(), e))

            '''reinitialize agent'''
            torch.manual_seed(self.population_id)
            torch.cuda.manual_seed_all(self.population_id)
            self.brain = self.build_brain()

            self.store()

    def store_to_checkpoint(self, checkpoint):
        import copy
        from .utils import get_vec_normalize

        '''store brain to CPU'''
        save_model = self.brain
        if self.cuda:
            save_model = copy.deepcopy(self.brain).cpu()
        save_model = [save_model,
                      getattr(get_vec_normalize(self.envs), 'ob_rms', None)]
        torch.save(
            save_model,
            os.path.join(
                self.log_dir,
                self.add_population_id_to_checkpoint_name(
                    'agent_{}.pt'.format(
                        checkpoint,
                    )
                )
            )
        )

        if self.mode in ['learning']:
            '''store training progress'''
            np.save(
                os.path.join(self.log_dir, "update_i.npy"),
                np.array([self.update_i]),
            )

    def add_population_id_to_checkpoint_name(self, agent_checkpoint_name):
        if self.population_number > 1:
            agent_checkpoint_name = '{}_{}'.format(
                self.population_id,
                agent_checkpoint_name,
            )
        return agent_checkpoint_name

    def restore_from_checkpoint(self, checkpoint):
        '''restore brain'''
        self.brain, ob_rms = torch.load(
            os.path.join(
                self.log_dir,
                self.add_population_id_to_checkpoint_name(
                    'agent_{}.pt'.format(
                        checkpoint,
                    )
                )
            )
        )
        if self.cuda:
            self.brain = self.brain.cuda()
        self.envs.ob_rms = ob_rms

        if self.mode in ['learning']:
            '''restore training progress'''
            self.update_i = np.load(
                os.path.join(self.log_dir, "update_i.npy"),
            )[0]
            self.num_trained_frames_start = self.get_num_trained_frames()

    def get_possible_checkpoints(self):
        '''get possible_checkpoints'''
        import glob
        import os

        possible_checkpoints = []
        for file in glob.glob(os.path.join(self.log_dir, self.add_population_id_to_checkpoint_name('agent_*.pt'))):
            possible_checkpoints += [int(file.split(self.log_dir)
                                         [1].split('.pt')[0].split('agent_')[1])]
        possible_checkpoints = np.asarray(possible_checkpoints)
        possible_checkpoints.sort()
        return possible_checkpoints

    def get_checkpoint(self, possible_checkpoints, principle):
        '''
            recent
            uniform
            {}_th, according to time, ranking
            {}_frame, according to frame
        '''

        if principle in ['recent']:
            return possible_checkpoints[-1]

        elif principle in ['uniform']:
            num_possible_checkpoints = possible_checkpoints.shape[0]
            p = [1.0 / num_possible_checkpoints] * num_possible_checkpoints
            return np.random.choice(possible_checkpoints, p=p)

        elif 'th' in principle:
            return possible_checkpoints[int(principle.split('_th')[0])]

        elif 'frame' in principle:
            return int(principle.split('_frame')[0])
