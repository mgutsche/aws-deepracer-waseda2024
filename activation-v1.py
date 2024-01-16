import math

import numpy as np

corner_weights = [0., 0., 0., 0., 0.,
                  0., 0., 0., 0., 0.,
                  0., 0., 0., 0., 0.,
                  0., 0., 0., 0., 0.,
                  0., 0., 0., 0., -0.05298124,
                  -0.12320451, -0.21419228, -0.33168991, -0.45528813, -0.55592047,
                  -0.65433926, -0.76911319, -0.88456759, -1., -0.87059673,
                  -0.79876319, -0.70303521, -0.54998881, -0.44852226, -0.36138406,
                  -0.27260266, -0.1208924, 0., 0., 0.,
                  0., 0., 0., 0., 0.,
                  0., 0., 0.17052701, 0.59630162, 1.,
                  0.70671317, 0., 0., 0., 0.,
                  0., 0., 0., 0., 0.,
                  0., 0., -0.21371, -1., -0.39991108,
                  0., 0., 0., 0., 0.,
                  0., 0., 0., 0., 0.,
                  0., -0.29170773, -0.54022915, -0.7554555, -0.87998036,
                  -1., -0.69238458, -0.46855806, -0.28076513, -0.08943792,
                  0., 0., 0., 0., 0.,
                  0., 0., 0., 0., 0.,
                  0., 0., 0., 0., 0.,
                  -0.12013132, -0.7214941, -1., -0.57455371, 0.,
                  0., 0., 0., 0., 0.,
                  0., 0., 0., 0.]

speed_weights = [0.95731166, 0.97199218, 0.98251486, 0.98975053, 0.99446759,
                 0.99733198, 0.99890718, 0.99965422, 0.9999317, 0.99999573,
                 1., 0.99999573, 0.9999317, 0.99965422, 0.99890718,
                 0.99733198, 0.99446759, 0.98975053, 0.98251486, 0.97199218,
                 0.95731166, 0.9375, 0.91148146, 0.87807783, 0.73004599,
                 0.53748125, 0.29185314, 0., 0., 0.,
                 0., 0., 0., 0., 0.,
                 0., 0., 0., 0.04045549, 0.25163187,
                 0.44669468, 0.7566152, 0.9999, 1., 0.9999,
                 0.9984, 0.9919, 0.9744, 0.9375, 0.8704,
                 0.7599, 0.5904, 0.00284598, 0., 0.,
                 0., 0.58396721, 0.70529823, 0.76284004, 0.78421272,
                 0.78914487, 0.78947368, 0.78914487, 0.78421272, 0.76284004,
                 0.70529823, 0.58396721, 0., 0., 0.,
                 0.37025853, 0.61163651, 0.75821083, 0.83881579, 0.87704307,
                 0.89124178, 0.8945184, 0.89473684, 0.8945184, 0.89124178,
                 0.87704307, 0.25540032, 0., 0., 0.,
                 0., 0., 0., 0.02886974, 0.58102417,
                 0.8704, 0.9375, 0.9744, 0.9919, 0.9984,
                 0.9999, 1., 0.9999, 0.9984, 0.9919,
                 0.9744, 0.9375, 0.8704, 0.7599, 0.5904,
                 0.10363735, 0., 0., 0., 0.31698654,
                 0.44368127, 0.55187487, 0.64346271, 0.72023769, 0.78389027,
                 0.83600847, 0.87807783, 0.91148146, 0.9375]

steering_weights = [0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 0.,
                    0., 0., 0., 0., 0.,
                    0., 0., 0., 0., -0.00688951,
                    -0.02291062, -0.05076348, -0.09389535, -0.15309953, -0.22538961,
                    -0.31047775, -0.41049072, -0.525517, -0.65555373, -0.76876328,
                    -0.86574233, -0.94114162, -0.9848075, -1., -0.98778902,
                    -0.9509473, -0.88157961, -0.78156665, -0.66654037, -0.53650364,
                    -0.42329408, -0.31942553, -0.22800513, -0.15648638, -0.09816201,
                    -0.05116881, -0.01572045, 0.02217478, 0.09971589, 0.22975262,
                    0.32165129, 0.32165129, 0.32165129, 0.32165129, 0.32165129,
                    0.32165129, 0.32165129, 0.32165129, 0.29947652, 0.2219354,
                    0.09189867, 0., -0.02779015, -0.15782688, -0.20983001,
                    -0.20983001, -0.20983001, -0.20983001, -0.20983001, -0.20983001,
                    -0.20983001, -0.20983001, -0.20983001, -0.18203986, -0.05200313,
                    0., -0.03793272, -0.10818235, -0.20641932, -0.32084909,
                    -0.45088582, -0.54092125, -0.601851, -0.63836078, -0.649991,
                    -0.649991, -0.649991, -0.61205828, -0.54180865, -0.44357168,
                    -0.32914191, -0.19910518, -0.10906975, -0.04813999, -0.01163021,
                    0., 0., 0., 0., 0.,
                    -0.01562148, -0.10944222, -0.23947895, -0.31419204, -0.31419204,
                    -0.31419204, -0.31419204, -0.31419204, -0.31419204, -0.31419204,
                    -0.31419204, -0.29857055, -0.20474982, -0.07471309, 0.,
                    0., 0., 0., 0.]


def reward_function(params):
    '''
    Example of using waypoints and heading to make the car point in the right direction
    '''

    # Read input variables
    waypoints = params['waypoints']
    closest_waypoints = params['closest_waypoints']
    heading = params['heading']
    steering_angle = params['steering_angle']

    # Initialize the reward with typical value
    reward = 2.0

    ########### custom ###############
    is_left_of_center: bool = params['is_left_of_center']
    speed = params['speed']
    distance_from_center = params['distance_from_center']
    track_width = params['track_width']

    corner_weight: float = corner_weights[closest_waypoints[1]]
    speed_weight: float = speed_weights[closest_waypoints[1]]
    steering_weight: float = steering_weights[closest_waypoints[1]]

    if (corner_weight > 0 and not is_left_of_center) or (corner_weight < 0 and is_left_of_center):
        # car is on the right side in the corner
        reward = reward + corner_weight

    if speed > speed_weight * 3:
        # car is going at least the right speed
        reward = reward + float(speed_weight)

    if distance_from_center > track_width / 3:
        reward *= 0.5

    if steering_weight > 0.4 and steering_angle > 3:
        # should turn right, turns left instead
        reward -= abs(corner_weight) * abs(steering_angle / 15)

    if steering_weight < -0.4 and steering_angle < -3:
        # should turn left, turns right instead
        reward -= abs(corner_weight) * abs(steering_angle / 15)

    if steering_weight > -0.15 and steering_weight < 0.15 and abs(steering_angle) < 5:
        # should go straight and does so
        reward += (1 - (abs(steering_angle) / 30)) * 0.5

    #### stock ####
    # Calculate the direction of the center line based on the closest waypoints
    next_point = waypoints[closest_waypoints[1]]
    prev_point = waypoints[closest_waypoints[0]]

    # Calculate the direction in radius, arctan2(dy, dx), the result is (-pi, pi) in radians
    track_direction = math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0])
    # Convert to degree
    track_direction = math.degrees(track_direction)

    # Calculate the difference between the track direction and the heading direction of the car
    direction_diff = abs(track_direction - heading)
    if direction_diff > 180:
        direction_diff = 360 - direction_diff

    # Penalize the reward if the difference is too large
    DIRECTION_THRESHOLD = 20.0
    if direction_diff > DIRECTION_THRESHOLD:
        reward *= 0.5

    return float(reward)