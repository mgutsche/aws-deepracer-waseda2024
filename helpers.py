import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from waypoints import waypoints


def get_eroded_indices(data, kernel):
    st_element = np.array(kernel).astype(bool)
    padded = np.pad(data.astype(bool), st_element.shape[0] // 2)
    windows = sliding_window_view(padded, window_shape=st_element.shape)
    return ~np.all(windows | ~st_element, axis=1)


def zero_runs(a):
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges


def plot_metric(m):
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.axis('equal')
    ax.scatter(waypoints[:-1, 2], waypoints[:-1, 3], s=2)  # inner line
    ax.scatter(waypoints[:-1, 4], waypoints[:-1, 5], s=2)  # outer line
    # ax.scatter(waypoints[:-1,0], waypoints[:-1,1], s=2)  # center line
    # metric on center line
    ax.scatter(x=waypoints[:, 0], y=waypoints[:, 1], c=m, cmap="plasma", vmin=m.min(), vmax=m.max(), s=30)
    for p, m in zip(waypoints[:, 0:2], m):
        if m != 0:
            ax.annotate(f"{m:.1f}", (p[0] + .02, p[1] + .03))

    plt.show()
