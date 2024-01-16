import math

import numpy as np

from helpers import get_eroded_indices, zero_runs
from waypoints import waypoints

# calculate relative angle between waypoints
degs = np.array([])
for wpt1, wpt2, wpt3 in zip(waypoints[:-2], waypoints[1:-1], waypoints[2:]):
    a1 = math.degrees(math.atan2(wpt1[1] - wpt2[1], wpt1[0] - wpt2[0]))
    a2 = math.degrees(math.atan2(wpt2[1] - wpt3[1], wpt2[0] - wpt3[0]))
    degs = np.append(degs, (a1 - a2 + 180) % 360 - 180)

# duplicate the first and last element to keep length consistent with waypoints
degs = np.append(degs, degs[-1])
degs = np.append(degs[0], degs)

# cleanup
degs[np.logical_and(degs < 4, degs > -4)] = 0  # threshold
degs[get_eroded_indices(degs, np.ones(3))] = 0  # erode in order to "shorten" curves

# additional interesting values
z = zero_runs(degs)

corners = np.array(list(zip(z[:-1, 1], z[1:, 0])))

mid_corner_pts = np.array(np.ceil(z[:-1, 1] + ((z[1:, 0] - z[:-1, 1]) / 2)), dtype=np.int16)

corner_dir = degs[mid_corner_pts]
corner_dir[corner_dir < 0] = -1
corner_dir[corner_dir > 0] = 1

straights = np.array([*list(zip(corners[:-1,1], corners[1:,0])), (corners[-1,1], corners[0,0])])

pseudo_straights = np.array(
    [*zip(mid_corner_pts[:-1], mid_corner_pts[1:]), (mid_corner_pts[-1], mid_corner_pts[0])])

pseudo_straight_max_length = np.max(np.abs(pseudo_straights[:, 1] - pseudo_straights[:, 0]))

