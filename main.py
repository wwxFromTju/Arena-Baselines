import copy
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from assets.arguments import get_args
import assets.utils as utils
from assets.evaluate import evaluate

args = utils.check_and_prepare_args(get_args())
utils.check_and_prepare_torch(args.seed,args.cuda,args.cuda_deterministic)
utils.check_log_dir(args.log_dir)

def main():

    '''prepare log'''
    if args.vis:
        from tensorboardX import SummaryWriter
        tf_summary = SummaryWriter(log_dir=args.log_dir)

    if args.mode in ['scaler2fig']:
        print('# WARNING: Working in progress')
        import tensorflow as tf
        import glob
        event_paths = sorted(glob.glob(os.path.join(args.log_dir, "event*")))
        print(tf.train.summary_iterator(event_paths[-2]))
        for e in tf.train.summary_iterator(event_paths[-2]):
            print(e)
            for v in e.summary.value:
                print(v.tag)
                print(v.simple_value)
        input('# ACTION REQUIRED: Export Done.')

    '''build env'''
    from assets.arena_python_interface import make_arena

    if args.mode in ['train']:
        envs = make_arena(
            env_name=args.env_name,
            max_episode_steps = args.max_episode_steps,
            num_env=args.num_processes,
            use_visual=True,
            start_index=args.arena_start_index,
            device = args.device,
        )

    if (args.mode in ['eval_population','eval_human','eval_round','test_obs']):
        eval_envs = make_arena(
            env_name=args.env_name,
            max_episode_steps = args.max_episode_steps,
            num_env=1,
            use_visual=True,
            start_index=args.arena_start_index+args.num_processes,
            device = args.device,
        )

    '''build agent'''
    try:
        num_agents = envs.unwrapped.number_agents()
        envs_agent_refer = envs
    except Exception as e:
        num_agents = eval_envs.unwrapped.number_agents()
        envs_agent_refer = eval_envs
    agents = []
    learning_agent_id = 0
    from assets.agents import Agent
    for i in range(0,num_agents):
        agents += [Agent(
            id = i,
            envs = envs_agent_refer,
            recurrent_brain = args.recurrent_brain,
            num_processes = args.num_processes,
            num_steps = args.num_steps,
            use_linear_lr_decay = args.use_linear_lr_decay,
            use_linear_clip_decay = args.use_linear_clip_decay,
            use_gae = args.use_gae,
            gamma = args.gamma,
            tau = args.tau,
            num_env_steps = args.num_env_steps,
            num_updates = args.num_updates,
            log_dir = args.log_dir,
            tf_summary = tf_summary,
            device = args.device,
            cuda = args.cuda,

            trainer_id = args.trainer_id,
            value_loss_coef = args.value_loss_coef,
            entropy_coef = args.entropy_coef,
            lr = args.lr,
            eps = args.eps,
            alpha = args.alpha,
            max_grad_norm = args.max_grad_norm,
            clip_param = args.clip_param,
            ppo_epoch = args.ppo_epoch,
            num_mini_batch = args.num_mini_batch,

            log_interval = args.log_interval,
            vis = args.vis,
            vis_interval = args.vis_interval,
        )]

    from assets.agents_cluster import MultiAgentCluster
    agents = MultiAgentCluster(
        agents = agents,
        learning_agent_id = learning_agent_id,
        store_interval = args.store_interval,
        log_dir = args.log_dir,
        reload_playing_agents_interval = args.reload_playing_agents_interval,
        reload_playing_agents_principle = args.reload_playing_agents_principle,
        tf_summary = tf_summary,
    )
    agents.restore()
    agents.store()

    if args.mode in ['train']:

        print('# INFO: Train Starting')

        obs = envs.reset()
        agents.reset(obs)

        while True:

            agents.schedule()

            '''interact'''
            while agents.experience_not_enough():

                action = agents.act(
                    obs=obs,
                    learning_agents_mode='learning',
                )

                '''step'''
                obs, reward, done, infos = envs.step(action)
                agents.observe(obs, reward, done, infos,
                    learning_agents_mode='learning')

            agents.at_update(learning_agents_mode='learning')

    elif args.mode in ['test_obs']:
        # eval_envs.unwrapped.set_train_mode(False)
        evaluate(
            eval_envs=eval_envs,
            agents=agents,
            num_eval_episodes=args.num_eval_episodes,
            summary_video=False,
            vis_curves=False,
            compute_win_loss_rate=True,
            tf_summary=tf_summary,
            is_save_obs=True,
            log_dir=args.log_dir,
        )

    elif args.mode in ['eval_population','eval_human','eval_round']:

        from assets.evaluate import load_win_loss_matrix
        checkpoints_start_from, num_possible_checkpoints, skip_interval, \
            num_evaled_rounds_total, win_loss_matrix, status = load_win_loss_matrix(agents,args)

        if status in ['initilized']:

            if args.mode in ['eval_human','eval_round']:
                input('# WARNING: Mode requires a win_loss_matrix, but no win_loss_matrix is found on the disk. \
                    Thus, I will run eval_polulation first, this could take a while. Hit enter to continue.')

            from assets.evaluate import generate_win_loss_matrix
            win_loss_matrix = generate_win_loss_matrix(
                checkpoints_start_from=checkpoints_start_from,
                num_possible_checkpoints=num_possible_checkpoints,
                skip_interval=skip_interval,
                num_evaled_rounds_total = num_evaled_rounds_total,
                win_loss_matrix=win_loss_matrix,
                agents=agents,
                eval_envs=eval_envs,
                tf_summary=tf_summary,
                args=args,
            )

        if args.mode in ['eval_population']:

            print('# INFO: [eval_population][visualizing]')
            from assets.population_evaluate import vis_win_loss_matrix
            vis_win_loss_matrix(win_loss_matrix,log_dir=args.log_dir)

        if args.mode in ['eval_human']:

            from assets.evaluate import eval_human
            eval_human(
                checkpoints_start_from=checkpoints_start_from,
                num_possible_checkpoints=num_possible_checkpoints,
                skip_interval=skip_interval,
                win_loss_matrix=win_loss_matrix,
                agents=agents,
                eval_envs=eval_envs,
                tf_summary=tf_summary,
                args=args,
            )

        if args.mode in ['eval_round']:

            from assets.evaluate import eval_round
            eval_round(
                checkpoints_start_from=checkpoints_start_from,
                num_possible_checkpoints=num_possible_checkpoints,
                skip_interval=skip_interval,
                win_loss_matrix=win_loss_matrix,
                agents=agents,
                eval_envs=eval_envs,
                tf_summary=tf_summary,
                args=args,
            )


if __name__ == "__main__":
    main()
