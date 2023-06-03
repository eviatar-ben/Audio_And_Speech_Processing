import librosa

import digits_classifier as dc
import numpy as np
import matplotlib.pyplot as plt


def test_dwt():
    time = np.linspace(0, 20, 1000)
    amplitude_a = 5 * np.sin(time)
    amplitude_b = 5 * np.sin(time)

    m = dc.ClassifierHandler().get_pretrained_model()
    # distance, wp = librosa.sequence.dtw(amplitude_a, amplitude_b)
    # distance = distance[-1 ,- 1]
    amplitude_a = np.array([np.array([x]) for x in amplitude_a])
    amplitude_a1 = np.array([np.array([x]) for x in amplitude_b])
    equals_signals_distance, _ = m.dtw(amplitude_a, amplitude_a1)

    time = np.linspace(0, 20, 1000)
    amplitude_b = 5 * np.sin(time) + 2
    amplitude_b = np.array([np.array([x]) for x in amplitude_b])
    amplitude_c = 3 * np.sin(time)
    amplitude_c = np.array([np.array([x]) for x in amplitude_c])
    amplitude_d = 3 * np.sin(time) + 2
    amplitude_d = np.array([np.array([x]) for x in amplitude_d])

    assert equals_signals_distance < m.dtw(amplitude_a, amplitude_b)[0]
    assert equals_signals_distance < m.dtw(amplitude_a, amplitude_c)[0]
    assert equals_signals_distance < m.dtw(amplitude_a, amplitude_d)[0]
    assert m.dtw(amplitude_a, amplitude_b)[0] < m.dtw(amplitude_a, amplitude_d)[0]
    assert m.dtw(amplitude_a, amplitude_c)[0] < m.dtw(amplitude_a, amplitude_d)[0]

    fig = plt.figure(figsize=(12, 4))
    _ = plt.plot(time, amplitude_a, label='A')
    _ = plt.plot(time, amplitude_b, label='B')
    _ = plt.title('DTW distance between A and B is %.2f' % equals_signals_distance)
    _ = plt.ylabel('Amplitude')
    _ = plt.xlabel('Time')
    _ = plt.legend()
    plt.show()


def example():
    import numpy as np
    import matplotlib.pyplot as plt
    y, sr = librosa.load(librosa.ex('brahms'), offset=10, duration=15)
    X = librosa.feature.chroma_cens(y=y, sr=sr)
    noise = np.random.rand(X.shape[0], 200)
    Y = np.concatenate((noise, noise, X, noise), axis=1)
    D, wp = librosa.sequence.dtw(X, Y, subseq=True)
    fig, ax = plt.subplots(nrows=2, sharex=True)
    img = librosa.display.specshow(D, x_axis='frames', y_axis='frames',
                                   ax=ax[0])
    ax[0].set(title='DTW cost', xlabel='Noisy sequence', ylabel='Target')
    ax[0].plot(wp[:, 1], wp[:, 0], label='Optimal path', color='y')
    ax[0].legend()
    fig.colorbar(img, ax=ax[0])
    ax[1].plot(D[-1, :] / wp.shape[0])
    ax[1].set(xlim=[0, Y.shape[1]], ylim=[0, 2],
              title='Matching cost function')


example()
# test_dwt()


# @staticmethod
# def dtw(x, y, dist=lambda x, y: norm(x - y, ord=2)):
#     from numpy import array, zeros, argmin, inf, equal, ndim
#     from scipy.spatial.distance import cdist
#     """
#     Computes Dynamic Time Warping (DTW) of two sequences.
#
#     :param array x: N1*M array
#     :param array y: N2*M array
#     :param func dist: distance used as cost measure
#
#     Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
#     """
#
#     def _traceback(D):
#         i, j = array(D.shape) - 2
#         p, q = [i], [j]
#         while ((i > 0) or (j > 0)):
#             tb = argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
#             if (tb == 0):
#                 i -= 1
#                 j -= 1
#             elif (tb == 1):
#                 i -= 1
#             else:  # (tb == 2):
#                 j -= 1
#             p.insert(0, i)
#             q.insert(0, j)
#         return array(p), array(q)
#
#     r, c = len(x), len(y)
#     D0 = zeros((r + 1, c + 1))
#     D0[0, 1:] = inf
#     D0[1:, 0] = inf
#     D1 = D0[1:, 1:]  # view
#
#     for i in range(r):
#         for j in range(c):
#             D1[i, j] = dist(x[i], y[j])
#
#     C = D1.copy()
#
#     for i in range(r):
#         for j in range(c):
#             D1[i, j] += min(D0[i, j], D0[i, j + 1], D0[i + 1, j])
#     if len(x) == 1:
#         path = zeros(len(y)), range(len(y))
#     elif len(y) == 1:
#         path = range(len(x)), zeros(len(x))
#     else:
#         path = _traceback(D0)
#     return D1[-1, -1] / sum(D1.shape), C, D1, path
