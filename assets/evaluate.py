import torch
import tensorflow as tf
import os
import numpy as np
import cv2
import time
import pygame
from pygame.locals import *
from .utils import flatten_agent_axis_to_img_axis
import matplotlib.pyplot as plt

def display_obs(name,x):
    cv2.imshow(name,cv2.resize(x, (720, 720)))
    cv2.waitKey(100)

def save_obs(log_dir,name,x):
    cv2.imwrite(os.path.join(log_dir,name+'.jpg'),x)
    cv2.waitKey(100)

def evaluate(eval_envs,agents,num_eval_episodes,summary_video=False,vis_curves=False,compute_win_loss_rate=False,\
    agent_1_is_human=False,tf_summary=None,display_obs=False,save_obs=False,log_dir=None):

    '''reset'''
    obs = eval_envs.reset()
    for agent in agents.all_agents:
        agent.episode_scaler_summary.reset()

    if summary_video:
        obs_video = None

    if agent_1_is_human:
        pygame.init()
        screen = pygame.display.set_mode( (200,150) )
        pygame.display.set_caption('Control Window')
        background = pygame.image.load('./picforcontrol.jpg').convert()
        background = pygame.transform.scale(background,(200,150))
        screen.blit(background,(0,0))
        pygame.display.update()
        playground_left = pygame.image.load('./Left.jpg').convert()
        playground_left = pygame.transform.scale(playground_left,(200,150))
        playground_right = pygame.image.load('./Right.jpg').convert()
        playground_right = pygame.transform.scale(playground_right,(200,150))

        print("# ACTION REQUIRED: Press p to start")
        while True:
            pygame.event.pump()
            pressed = pygame.key.get_pressed()
            if pressed[pygame.K_p]:
                print('# INFO: Go!')
                break

    step_i = -1

    while agents.learning_agent.episode_scaler_summary.get_length() < num_eval_episodes:

        step_i += 1

        '''act'''
        action = agents.act(
            obs=obs,
            learning_agent_mode='playing',
        )

        if agent_1_is_human:
            pygame.event.pump()
            action[0,1,0] = 0
            pressed = pygame.key.get_pressed()
            if pressed[pygame.K_f]:
                action[0,1,0] = 5
            elif pressed[pygame.K_w]:
                action[0,1,0] = 3
            elif pressed[pygame.K_a]:
                action[0,1,0] = 1
            elif pressed[pygame.K_s]:
                action[0,1,0] = 4
            elif pressed[pygame.K_d]:
                action[0,1,0] = 2
            elif pressed[K_ESCAPE]:
                break

        '''step'''
        obs, reward, done, infos = eval_envs.step(action)
        agents.observe(obs, reward, done, infos,
            learning_agent_mode='playing')

        if agent_1_is_human:
            if infos[0]['shift']==0:
                screen.blit(playground_right,(0,0))
                pygame.display.update()
            elif infos[0]['shift']==1:
                screen.blit(playground_left,(0,0))
                pygame.display.update()

        if display_obs or save_obs:
            for agent_i in range(agents.num_agents):
                x = obs[0,agent_i][-1].cpu().numpy()
                name = 'obs_a-{}'.format(agent_i)
                if display_obs:
                    display_obs(
                        name = name,
                        x = x,
                    )
                elif save_obs:
                    name += '_f-{}'.format(step_i)
                    save_obs(
                        log_dir = log_dir,
                        name = name,
                        x = x,
                    )

        '''summary_video'''
        if summary_video:
            obs_frame = flatten_agent_axis_to_img_axis(obs[0,:,-1,:,:]).unsqueeze(0)
            if obs_video is None:
                obs_video = obs_frame
            else:
                obs_video = torch.cat([obs_video,obs_frame],0)

        '''log at done'''
        if done.any():
            agents.learning_agent.log(mode='playing')

    '''terminate and summary'''

    if vis_curves:
        agents.learning_agent.vis_curves(mode='playing')

    if summary_video:
        # T,H,W --> vid_tensor: :math:`(N, T, C, H, W)`.
        obs_video=obs_video.unsqueeze(0).unsqueeze(2)
        tf_summary.add_video(
            'eval/obs',
            obs_video,
            agents.learning_agent.get_num_trained_frames(),
        )

    if compute_win_loss_rate:
        assert agents.num_agents==2
        '''return win-loss rate'''
        win_loss_record = []
        for espisode_i in range(len(agents.learning_agent.episode_scaler_summary.final_rewards['raw'])):
            if agents.learning_agent.episode_scaler_summary.final_rewards['raw'][espisode_i]>agents.playing_agents[0].episode_scaler_summary.final_rewards['raw'][espisode_i]:
                win_loss_record += [1.0]
            elif agents.learning_agent.episode_scaler_summary.final_rewards['raw'][espisode_i]<agents.playing_agents[0].episode_scaler_summary.final_rewards['raw'][espisode_i]:
                win_loss_record += [0.0]
            elif agents.learning_agent.episode_scaler_summary.final_rewards['raw'][espisode_i]==agents.playing_agents[0].episode_scaler_summary.final_rewards['raw'][espisode_i]:
                win_loss_record += [0.5]
            else:
                raise NotImplemented

        print('# INFO: [compute_win_loss_rate][win_loss_record-{}]'.format(
            win_loss_record
        ))
        '''compute win_loss_rate'''
        win_loss_rate = np.mean(win_loss_record)

        return win_loss_rate

def load_win_loss_matrix(agents,args):
    '''prepare'''
    num_possible_checkpoints = agents.learning_agent.get_possible_checkpoints().shape[0]
    if args.population_eval_start is None:
        checkpoints_start_from = int(input('# ACTION REQUIRED: {} possible checkpoints, population eval start from: '.format(num_possible_checkpoints)))
    else:
        checkpoints_start_from = args.population_eval_start
        print('# INFO: {} possible checkpoints, population eval start from {} (default setting loaded)'.format(
            num_possible_checkpoints,checkpoints_start_from))
    if args.skip_interval is None:
        skip_interval = int(input('# ACTION REQUIRED: skip_interval:'))
    else:
        skip_interval = args.skip_interval
        print('# INFO: skip_interval: {} (default setting loaded)'.format(skip_interval))
    num_evaled_agents = (num_possible_checkpoints - checkpoints_start_from)//skip_interval+1
    num_evaled_rounds_total = int(num_evaled_agents**2)

    win_loss_matrix = np.zeros((num_evaled_agents,num_evaled_agents))

    try:
        '''load win_loss_matrix'''
        win_loss_matrix = np.load(
            os.path.join(args.log_dir, "win_loss_matrix_for_{}_{}_{}_agents.npy".format(
                checkpoints_start_from,
                num_possible_checkpoints,
                skip_interval,
            )),
        )
        print('# INFO: [eval_polulation][{}-{}-{}][win_loss_matrix loaded]'.format(
            checkpoints_start_from,
            num_possible_checkpoints,
            skip_interval,
        ))
        return checkpoints_start_from, num_possible_checkpoints, skip_interval, num_evaled_rounds_total, win_loss_matrix, 'loaded'
    except Exception as e:
        print('# INFO: [eval_polulation][{}-{}-{}][win_loss_matrix load failed with error {}]'.format(
            checkpoints_start_from,
            num_possible_checkpoints,
            skip_interval,
            e,
        ))
        return checkpoints_start_from, num_possible_checkpoints, skip_interval, num_evaled_rounds_total, win_loss_matrix, 'initilized'

def generate_win_loss_matrix(checkpoints_start_from, num_possible_checkpoints, skip_interval, num_evaled_rounds_total, win_loss_matrix, agents, eval_envs, tf_summary, args):

    print('# INFO: [generate win_loss_matrix][{}-{}-{}][start]'.format(
        checkpoints_start_from,
        num_possible_checkpoints,
        skip_interval,
    ))

    eval_start_time = time.time()
    num_rounds_evaled = 0
    for x in range(checkpoints_start_from,num_possible_checkpoints,skip_interval):
        for y in range(checkpoints_start_from,num_possible_checkpoints,skip_interval):
            print('# INFO: [generate win_loss_matrix][x-{},y-{}][start]'.format(
                x,y,
            ))
            agents.all_agents[0].restore(principle='{}_th'.format(x))
            agents.all_agents[1].restore(principle='{}_th'.format(y))
            win_loss_rate = evaluate(
                eval_envs=eval_envs,
                agents=agents,
                num_eval_episodes=args.num_eval_episodes,
                summary_video=False,
                vis_curves=False,
                compute_win_loss_rate=True,
                tf_summary=tf_summary,
            )
            num_rounds_evaled += 1
            print('# INFO: [generate win_loss_matrix][x-{},y-{}][done][win_loss_rate-{}][Remain {:.2f} hrs]'.format(
                x,y,
                win_loss_rate,
                (time.time()-eval_start_time)/num_rounds_evaled*(num_evaled_rounds_total-num_rounds_evaled)/60.0/60.0,
            ))
            win_loss_matrix[(x-checkpoints_start_from)//skip_interval,(y-checkpoints_start_from)//skip_interval] = win_loss_rate

    np.save(
        os.path.join(args.log_dir, "win_loss_matrix_for_{}_{}_{}_agents.npy".format(
            checkpoints_start_from,
            num_possible_checkpoints,
            skip_interval,
        )),
        win_loss_matrix,
    )

    print('# INFO: [generate win_loss_matrix][{}-{}-{}][done and saved]'.format(
        checkpoints_start_from,
        num_possible_checkpoints,
        skip_interval,
    ))

    return win_loss_matrix

def get_win_percentage(win_loss_matrix,sort):
    win_percentage = np.concatenate(
        (
            np.mean(win_loss_matrix,axis=1,keepdims=True),
            np.expand_dims(np.array(range(win_loss_matrix.shape[0])),axis=1)
        ),
        axis = 1,
    )
    if sort:
        win_percentage = win_percentage[np.argsort(win_percentage[:, 0])[::-1]]
    return win_percentage

def eval_round(checkpoints_start_from, num_possible_checkpoints, skip_interval, win_loss_matrix, agents, eval_envs, tf_summary, args):

    while True:

        print('# INFO: [eval_round][starting]')

        matching_agent_checkpoint_ids = []

        for agent_i in range(agents.num_agents):

            checkpoint_id = int(input('# ACTION REQUIRED: choose agent {} (checkpoint_id): (input from {} to {})'.format(
                agent_i,
                0,
                win_loss_matrix.shape[agent_i],
            )))

            matching_agent_checkpoint_ids += [checkpoint_id]

            checkpoint_id = int(checkpoint_id*skip_interval+checkpoints_start_from)

            agents.all_agents[agent_i].restore(principle='{}_th'.format(
                checkpoint_id
            ))

            print('# INFO: [eval_round][agent {} loaded, checkpoint_id {}]'.format(
                agent_i,
                checkpoint_id,
            ))

        eval_envs.unwrapped.set_train_mode(False)
        win_loss_rate = evaluate(
            eval_envs=eval_envs,
            agents=agents,
            num_eval_episodes=10000,
            summary_video=False,
            vis_curves=False,
            compute_win_loss_rate=True,
            agent_1_is_human=False,
            tf_summary=tf_summary,
        )

        print('# INFO: [eval_round][done][match between {} (matching_agent_checkpoint_ids), agent at checkpoint {} win with {} (<0.5 means lost)], it should be {}'.format(
            matching_agent_checkpoint_ids,
            matching_agent_checkpoint_ids[0],
            win_loss_rate,
            win_loss_matrix[matching_agent_checkpoint_ids[0],matching_agent_checkpoint_ids[1]],
        ))

def eval_human(checkpoints_start_from, num_possible_checkpoints, skip_interval, win_loss_matrix, agents, eval_envs, tf_summary, args):

    print('# INFO: [eval_human][starting]')

    win_percentage = get_win_percentage(win_loss_matrix=win_loss_matrix,sort=True)

    ranking_i = 0
    while True:

        checkpoint_id = int(win_percentage[ranking_i,1]*skip_interval)
        win_percentage_i = win_percentage[ranking_i,0]
        print('# INFO: [eval_human][testing agent ranking at {}, against checkpoint_id {}, win_percentage_i {}]'.format(
            ranking_i,
            checkpoint_id,
            win_percentage_i,
        ))

        agents.all_agents[0].restore(principle='{}_th'.format(checkpoint_id))
        eval_envs.unwrapped.set_train_mode(False)
        win_loss_rate = evaluate(
            eval_envs=eval_envs,
            agents=agents,
            num_eval_episodes=args.num_eval_episodes,
            summary_video=False,
            vis_curves=False,
            compute_win_loss_rate=True,
            agent_1_is_human=True,
            tf_summary=tf_summary,
        )

        if win_loss_rate<0.5:
            print('You win this agent with win_loss_rate of {}! You rank at {}.'.format(win_loss_rate,sting_i))
            np.save(
                os.path.join(args.log_dir, "human_ranking_for_{}_{}_{}_agents.npy".format(
                    checkpoints_start_from,
                    num_possible_checkpoints,
                    skip_interval,
                )),
                np.array([ranking_i]),
            )
            input('# ACTION REQUIRED: Done')
        else:
            print('You lost this agent with win_loss_rate of {}'.format(win_loss_rate))

        ranking_i += 1
