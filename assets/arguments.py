import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')

    parser.add_argument('--mode', type=str, default='train',
                        help='\
                            [test_obs: test if obs make sense (normally neeeded when you setup the virtual display)]\
                            [train: standard training]\
                            [eval_population: evaluate population performance]\
                            [eval_human: evaluate against human player]\
                            [eval_round: evaluate agent against agent]\
                            [scaler2fig: convert scalers logged in tensorboardX to fig]')

    '''general and import settings'''
    parser.add_argument('--env-name', default='PongNoFrameskip-v4',
                        help='[general][environment to train on (default: PongNoFrameskip-v4)]')
    parser.add_argument('--num-env-steps', type=int, default=10e6,
                        help='[general][number of environment steps to train (default: 10e6)]')
    parser.add_argument('--log-dir', default='/tmp/gym/',
                        help='[general][directory to save agent logs (default: /tmp/gym)]')
    parser.add_argument('--seed', type=int, default=1,
                        help='[general][random seed (default: 1)]')

    '''settings for brain'''
    parser.add_argument('--add-timestep', action='store_true', default=False,
                        help='[brain][add timestep to observations]')
    parser.add_argument('--recurrent-brain', action='store_true', default=False,
                        help='[brian][use a recurrent policy]')

    '''settings for trainer'''
    parser.add_argument('--trainer-id', default='a2c',
                        help='[trainer][trainer to use: a2c | ppo | acktr]')
    parser.add_argument('--lr', type=float, default=7e-4,
                        help='[trainer][learning rate (default: 7e-4)]')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='[trainer][RMSprop optimizer epsilon (default: 1e-5)]')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='[trainer][RMSprop optimizer apha (default: 0.99)]')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='[trainer][discount factor for rewards (default: 0.99)]')
    parser.add_argument('--use-gae', action='store_true', default=False,
                        help='[trainer][use generalized advantage estimation]')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='[trainer][gae parameter (default: 0.95)]')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='[trainer][entropy term coefficient (default: 0.01)]')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='[trainer][value loss coefficient (default: 0.5)]')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='[trainer][max norm of gradients (default: 0.5)]')
    parser.add_argument('--num-processes', type=int, default=16,
                        help='[trainer][how many training CPU processes to use (default: 16)]')
    parser.add_argument('--num-steps', type=int, default=5,
                        help='[trainer][number of forward steps in A2C (default: 5)]')
    parser.add_argument('--ppo-epoch', type=int, default=4,
                        help='[trainer][number of ppo epochs (default: 4)]')
    parser.add_argument('--num-mini-batch', type=int, default=32,
                        help='[trainer][number of batches for ppo (default: 32)]')
    parser.add_argument('--clip-param', type=float, default=0.2,
                        help='[trainer][ppo clip parameter (default: 0.2)]')
    parser.add_argument('--use-linear-lr-decay', action='store_true', default=False,
                        help='[trainer][use a linear schedule on the learning rate]')
    parser.add_argument('--use-linear-clip-decay', action='store_true', default=False,
                        help='[trainer][use a linear schedule on the ppo clipping parameter]')

    '''settings for self-play'''
    parser.add_argument('--reload-playing-agents-interval', type=int, default=(60*5),
                        help='[self-play][interval to switch component in seconds]')
    parser.add_argument('--reload-playing-agents-principle', type=str, default=50,
                        help = '[self-play][principle of choosing a component]\
                            [\
                                recent(the most recent checkpoint),\
                                uniform(uniformly sample from historical checkpoint),\
                                prioritized(prioritized based on win rate),\
                            ]')

    '''general but not important settings'''
    parser.add_argument('--cuda-deterministic', action='store_true', default=False,
                        help="[general][sets flags for determinism when using CUDA (potentially slow!)]")
    parser.add_argument('--log-interval', type=int, default=10,
                        help='[general][log interval, one log per n updates (default: 10)]')
    parser.add_argument('--store-interval', type=int, default=int(60*10),
                        help='[general][save interval in seconds')
    parser.add_argument('--vis-interval', type=int, default=100,
                        help='[general][vis interval, one log per n updates (default: 100)]')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='[general][disables CUDA training]')
    parser.add_argument('--vis', action='store_true', default=False,
                        help='[general][enable visdom visualization]')
    parser.add_argument('--num-eval-episodes', type=int, default=2,
                        help='[general][how many episodes to run for one evaluation]')
    parser.add_argument('--arena-start-index', type=int, default=2394,
                        help='[general][each arena runs on a port, specify the ports to run the arena]')
    parser.add_argument('--aux', type=str, default='',
                        help='[general][some aux information you may want to record along with this run]')

    '''debug'''
    parser.add_argument('--test-env', action='store_true', default=False,
                        help='[debug][enable visdom visualization]')
    parser.add_argument('--eval-against', type=str, default=None,
                        help='[debug][eval against an agent]')

    args = parser.parse_args()
    import os
    args.log_dir = '../results'

    '''env'''
    args.log_dir = os.path.join(args.log_dir, 'en-{}'.format(args.env_name))

    '''trainer'''
    args.log_dir = os.path.join(args.log_dir, 'ti-{}'.format(args.trainer_id))

    '''self-play'''
    args.log_dir = os.path.join(args.log_dir, 'sscp-{}'.format(args.reload_playing_agents_principle))

    '''general'''
    args.log_dir = os.path.join(args.log_dir, 'a-{}'.format(args.aux))

    '''default args'''
    if (args.test_env) or (args.eval_against is not None):
        '''eval settings'''
        args.num_processes = 1
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    try:
        args.max_episode_steps = {
            'MoveBall-BW': 21,
            'Tennis': 25000,
        }[args.env_name]
    except Exception as e:
        args.max_episode_steps = -1
        print('# WARNING: args.max_episode_steps is default to be -1 (no limit)')

    try:
        args.population_eval_start = 0
    except Exception as e:
        args.population_eval_start = None

    try:
        args.skip_interval = {
        'Boomer-v2': 8,
        'Shooter-v4-Random': 4,
        'Snake-v3-Random': 12,
        'Billiards-v1': 12,
        'AirHockey-v1': 8,
        'Fallflat-v2': 8,
        'Tank_TP-v1': 3,
        }[args.env_name]
    except Exception as e:
        args.skip_interval = None

    return args
