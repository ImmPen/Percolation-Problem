import numpy as np

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
