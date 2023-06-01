import digits_classifier as dc
import numpy as np
import matplotlib.pyplot as plt






def test_dwt():
    x = np.array([0, 0, 1, 1, 2, 4, 2, 1, 2, 0]).reshape(-1, 1)
    y = np.array([1, 1, 1, 2, 2, 2, 2, 3, 2, 0]).reshape(-1, 1)

    # plot x and y

    plt.plot(x)
    plt.plot(y)
    plt.show()
    classifier = dc.ClassifierHandler().get_pretrained_model()
    dist, cost, acc_cost, path = classifier.dtw(x, y)
    print('Minimum distance found:', dist)
    # print('Minimum distance found:', dist)
    # plt.imshow(acc.T, origin='lower', cmap=cm.gray, interpolation='nearest')
    # plot(path[0], path[1], 'w')
    # xlim((-0.5, acc.shape[0]-0.5))
    # ylim((-0.5, acc.shape[1]-0.5))

test_dwt()