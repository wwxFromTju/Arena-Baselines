import torch
import torch.nn as nn
from .envs import VecNormalize
import tensorflow as tf
import os
import numpy as np
import cv2
import pyscreenshot as ImageGrab
import imageio
import scipy.misc


class ScreenRecorder(object):
    """docstring for ScreenRecorder."""

    def __init__(self, log_dir):
        super(ScreenRecorder, self).__init__()
        self.log_dir = log_dir
        self.vis_train_episode = 0
        self.image_list = []

    def at_step(self):
        # TODO: The screen recorder cannot get the window or window demension or set window position,
        # so you have to manually drag the window to the left up corner
        self.image_list.append(ImageGrab.grab())

    def at_done(self):
        '''write video'''
        height, width, channel = np.array(self.image_list[-1]).shape
        out = cv2.VideoWriter(
            os.path.join(
                self.log_dir,
                "vis_train_{}.avi".format(
                    self.vis_train_episode,
                ),
            ),
            cv2.VideoWriter_fourcc(*'DIVX'),
            5,
            (width, height),
        )
        for images in self.image_list:
            out.write(
                cv2.cvtColor(
                    np.array(images),
                    cv2.COLOR_BGR2RGB,
                ),
            )
        out.release()

        '''write gif'''
        with imageio.get_writer(
            os.path.join(
                self.log_dir,
                "vis_train_{}.gif".format(
                    self.vis_train_episode,
                ),
            ),
            mode='I',
        ) as writer:
            for images in self.image_list:
                writer.append_data(
                    np.array(images),
                )

        '''write png'''
        png = np.zeros(np.array(images).shape).astype(float)
        for images in self.image_list:
            png += np.array(images).astype(float)
        png /= float(len(self.image_list))
        scipy.misc.imsave(
            os.path.join(
                self.log_dir,
                "vis_train_{}.png".format(
                    self.vis_train_episode,
                ),
            ),
            png,
        )

        self.vis_train_episode += 1
        self.image_list = []

        if self.vis_train_episode == 1:
            input('# ACTION REQUIRED: The window is normally not at the center at the first episode. So now drag the window to the center and press enter, we will take a screen record for your agent.')

        else:
            input('# ACTION REQUIRED: vis_train logged {} episodes (recent tag {}), log another? (Press enter to log another)'.format(
                self.vis_train_episode,
                self.vis_train_episode - 1,
            ))


def flatten_agent_axis_to_img_axis(x):
    # dataformats='HW',
    return torch.cat(list(x), 0)


def add_text(img_tensor, text, font=cv2.FONT_HERSHEY_SIMPLEX, position_bl=(0, 10), font_scale=0.2, font_color=(0, 0, 0), line_type=2):
    img_tensor = torch.cat(
        [torch.ones(int(img_tensor.size()[0] / 4), img_tensor.size()[1]), img_tensor.cpu()], 0)
    img = (img_tensor * 255.0).numpy().astype(np.uint8)
    cv2.putText(
        img,
        text,
        position_bl,
        font,
        font_scale,
        font_color,
        line_type,
    )
    img_tensor = torch.from_numpy(img).float() / 255.0
    return img_tensor


def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None


def get_vec_norm_for_eval(eval_envs, envs):
    vec_norm = get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = get_vec_normalize(envs).ob_rms
    return vec_norm


def check_and_prepare_args(args):
    '''parpare args'''
    assert args.trainer_id in ['a2c', 'ppo', 'acktr']
    if args.recurrent_brain:
        assert args.trainer_id in ['a2c', 'ppo'], \
            'Recurrent policy is not implemented for ACKTR'
    args.num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    '''prepare torch'''
    torch.set_num_threads(1)
    args.device = torch.device("cuda:0" if args.cuda else "cpu")

    return args


def check_and_prepare_torch(seed, cuda, cuda_deterministic):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if cuda and torch.cuda.is_available() and cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def check_log_dir(log_dir):
    import glob
    try:
        os.makedirs(log_dir)
        print('# WARNING: Dir empty, make new log dir :{}'.format(log_dir))
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)
        print('# INFO: Dir exists {}'.format(log_dir))

    eval_log_dir = log_dir + "/eval"
    try:
        os.makedirs(eval_log_dir)
    except OSError:
        files = glob.glob(os.path.join(eval_log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)

# Get a render function


def get_render_func(venv):
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None

# Necessary for my KFAC implementation.


class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def clear_port(port, ask):
    '''
    clear process running on a port
    '''

    '''get pid'''
    import subprocess
    command = 'ps axf | grep \'port {}\' | grep -v grep | awk \'{{print $1}}\''.format(
        port)
    pid = subprocess.check_output(
        [command], shell=True, stderr=subprocess.STDOUT).decode("utf-8").split('\n')[0]

    '''kill'''
    if len(pid) > 0:
        info = 'Find process of port {} with pid {}'.format(port, pid)
        if ask:
            input('# ACTION REQUIRED: {}, kill?'.format(info))
        else:
            print('# INFO: {}, killing'.format(info))
        subprocess.call(['kill', '-9', pid])


def clear_ports(start_index, num_env, ask):
    '''
        clear ports
    '''

    for port in range(start_index, (start_index + num_env), 1):
        clear_port(port, ask)
