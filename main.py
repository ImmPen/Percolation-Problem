import numpy as np
import matplotlib.pyplot as plt
import random


def get_proper_label(N, arr, i, j):
    r = 0
    if arr[i, j] != 0:
        r = arr[i, j]
        t = r
        t = -N[t]
        if t < 0:
          return r
        r = t
        t = -N[t]
        if t < 0:
            return r
        while t > 0:
            r = t
            t = -N[t]
        N[arr[i, j]] = int(-r)
    return r


def neighborhood(arr, i, j, labels):
    return set([get_proper_label(labels, arr, i - 1, j) if i > 0 else 0,\
                get_proper_label(labels, arr, i, j + 1) if j < (arr.shape[1] - 1) else 0,\
                get_proper_label(labels, arr, i + 1, j) if i < (arr.shape[0] - 1) else 0,\
                get_proper_label(labels, arr, i, j - 1) if j > 0 else 0])


def clusterize(arr, i, j, labels, incr):
    neighbor = neighborhood(arr, i, j, labels) - {0}
    if len(neighbor):
        tmp = min(neighbor)
        labels[tmp] += 1
        arr[i, j] = tmp
        for x in (neighbor - {tmp}):
            labels[tmp] += labels[x]
            labels[x] = -tmp
    else:
        incr += 1
        labels[incr] = 1
        arr[i, j] = incr
    return incr


def is_percolated(arr, labels):
    m, n = arr.shape
    bot = set()
    top = set()
    for x in range(n):
        if arr[m-1, x] != 0:
            bot.add(get_proper_label(labels, arr, m - 1, x))
        if arr[0, x] != 0:
            top.add(get_proper_label(labels, arr, 0, x))
    return not top.isdisjoint(bot)


if __name__ == '__main__':
    rng = np.random.default_rng(None)
    m = 20
    n = 20
    k = 0
    amount = 1000
    labels = dict()
    seed = np.arange(0, m * n)
    conc = np.array([])
    for i in range(amount):
        random.shuffle(seed)
        arr = np.zeros(shape=(m, n), dtype=int)
        k = 0
        for x in seed:
            k = clusterize(arr, x // n, x % n, labels, k)
            if is_percolated(arr, labels):
                break
        summ = 0
        for x in labels.values():
            if x > 0:
                summ += x
        #print(seed)
        #print(summ, m * n, summ / (m * n))
        conc = np.append(conc, summ / (m * n))
    ns, bins, patches = plt.hist(conc, 20)
    M = np.sum(ns * bins[:-1]) / amount
    D = np.sum(ns * (bins[:-1] - M) ** 2) / amount
    pos_y = max(ns)
    pos_x = min(bins)
    plt.text(pos_x, pos_y, 'Матрица M x N = {}x{}'.format(m, n))
    pos_y -= max(ns) // 10
    plt.text(pos_x, pos_y, 'Количество опытов - {}'.format(amount))
    pos_y -= max(ns) // 10
    plt.text(pos_x, pos_y, 'Среднее значение M = {:.4f}'.format(M))
    pos_y -= max(ns) // 10
    plt.text(pos_x, pos_y, r'Среднее квадратичное $\sigma^2$ = {:.4f}'.format(D))
    pos_y -= max(ns) // 10
    plt.text(pos_x, pos_y, r'отклонение $\sigma$ = {0:.4f}'.format(np.sqrt(D)).rjust(55))
    plt.title('Гистограмма')
    plt.xlabel('Концентрация')
    plt.ylabel('Количество')
    plt.grid(True)
    print(M)
    print(D)
    plt.show()
