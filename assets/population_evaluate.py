from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
sns.set()
mpl.use("Agg")

color_map = LinearSegmentedColormap.from_list(
    'gr', ["g", "w", "r"], N=256)  # Red and Green


def symetrify(a):
    for i in range(a.shape[0]):
        for j in range(i):
            a[i, j] = -a[j, i]
        a[i, i] = 0
    return a


def init(N=36):
    a = np.random.random((N, N))
    b = a * 2
    c = 1 - b
    return c


def game_random(a):
    return symetrify(a)


def game_purely_transitive(a):
    for i in range(N):
        a[i, i] = 0
        for j in range(i + 1, N):
            a[i, j] = a[i, j - 1] + 1 / N
    b = symetrify(a)
    return b


def game_transitive(a):
    a = np.sort(a, axis=1)
    am = np.mean(a, axis=1)
    b = a[(-am).argsort()]
    c = symetrify(b)
    return c


def game_cyclic(a):
    return a


def vis_win_loss_matrix(win_loss_matrix, log_dir='.'):
    win_loss_matrix = win_loss_matrix.transpose()

    fig = plt.figure()
    ax = sns.heatmap(win_loss_matrix, cmap="coolwarm")
    ax.invert_yaxis()
    ax.set_title('Win Rate of Agent 0')
    ax.set(xlabel='Agent 0', ylabel='Agent 1')
    plt.savefig('{}/winrate_map.pdf'.format(log_dir))

    fig = plt.figure()
    win_loss_percantage = np.mean(win_loss_matrix, axis=0, keepdims=False)
    plt.plot(win_loss_percantage)
    ax = fig.gca()
    ax.set(xlabel='Agent 0', ylabel='Winning Rate')
    ax.set_title('Population Performance of Agent 0')
    plt.savefig('{}/population_performance.pdf'.format(log_dir))

    fig = plt.figure()
    import scipy
    egs = scipy.linalg.schur(win_loss_matrix, sort='ouc')[0]
    ax = sns.heatmap(egs, cmap="coolwarm")
    ax.invert_yaxis()
    ax.set_title('EGS')
    ax.set(xlabel='Agent 0', ylabel='Agent 1')
    plt.savefig('{}/egs.pdf'.format(log_dir))


def generate_egs(win_loss_matrix, k, log_dir='.'):  # empirical gamescape (EGS)
    fig = plt.figure(figsize=(4, 4))
    ax = fig.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.scatter(range(win_loss_matrix.shape[0]), win_loss_matrix[k, :])
    plt.savefig('{}/egs.pdf'.format(log_dir))


if __name__ == "__main__":
    win_loss_matrix = init(N=36)
    # win_loss_matrix = game_random(win_loss_matrix)
    win_loss_matrix = game_transitive(win_loss_matrix)  # roughly
    # win_loss_matrix = game_purely_transitive(win_loss_matrix)

    generate_winrate_map(win_loss_matrix)
    generate_egs(win_loss_matrix, 0)
    # generate_egs(win_loss_matrix, int(N / 2))
