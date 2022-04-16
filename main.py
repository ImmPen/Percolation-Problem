import numpy as np
import matplotlib.pyplot as plt


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


def clusterize0(arr, m, n, conc):
    k = 0
    cluster_labels = {}
    for i in range(m):
        for j in range(n):
            if arr[i, j] > conc:
                arr[i, j] = 0
            else:
                neighboor = np.zeros(2)
                if i > 0:
                    neighboor[0] = arr[i - 1, j] > 0
                if j > 0:
                    neighboor[1] = arr[i, j - 1] > 0

                if (not neighboor[0]) and (not neighboor[1]):
                    k += 1
                    arr[i, j] = k
                    cluster_labels[k] = 1
                else:
                    if neighboor[0]:
                        neighboor[0] = get_proper_label(cluster_labels, arr, i - 1, j)
                    if neighboor[1]:
                        neighboor[1] = get_proper_label(cluster_labels, arr, i, j - 1)
                    if neighboor[0] and neighboor[1] \
                        and neighboor[0] != neighboor[1]:
                        arr[i, j] = min(neighboor)
                        cluster_labels[arr[i, j]] = cluster_labels[neighboor[0]] + cluster_labels[neighboor[1]] + 1
                        for x in neighboor:
                            if x != min(neighboor):
                                cluster_labels[x] = int(-arr[i, j])
                    elif neighboor[0]:
                        arr[i, j] = neighboor[0]
                        cluster_labels[arr[i, j]] += 1
                    else:
                        arr[i, j] = neighboor[1]
                        cluster_labels[arr[i, j]] += 1
    arr = np.int32(arr)
    return cluster_labels


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
    labels = dict()
    seed = np.arange(0, m * n)
    conc = np.array([])
    for i in range(100):
        rng.shuffle(seed)
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
        print(seed)
        print(summ, m * n, summ / (m * n))
        conc = np.append(conc, summ / (m * n))
    n, bins, patches = plt.hist(conc, 20)
    M = np.sum(n * bins[:-1]) / 100
    D = np.sum(n * (bins[:-1] - M) ** 2) / (100 - 1)
    plt.text(0.45, 12, 'M x N = 20 x 20')
    plt.text(0.45, 11, 'Количество опытов - 100')
    plt.title('Гистограмма')
    plt.xlabel('Концетрация')
    plt.ylabel('Количество')
    plt.grid(True)
    print(M)
    print(D)
    plt.show()
