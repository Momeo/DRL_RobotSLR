import numpy as np


class SLR(object):
    sensor_info = None
    location = None
    orien = None
    n_sensors = None
    sensor_theta = None
    sensor_max = None

    def __init__(self, n_sensors, sensor_max=150):
        self.n_sensors = n_sensors
        self.sensor_theta = np.linspace(-np.pi / 2, np.pi / 2, n_sensors)
        self.location = np.array([0, 0])
        self.orien = 0
        self.sensor_max = sensor_max
        self.sensor_info = sensor_max + np.zeros((self.n_sensors, 3))

    def update(self, location, orien, obstacle_list):
        self.orien = rotation = orien
        self.location = cx, cy = location
        xs = cx + (np.zeros((self.n_sensors,)) + self.sensor_max) * np.cos(self.sensor_theta)
        ys = cy + (np.zeros((self.n_sensors,)) + self.sensor_max) * np.sin(self.sensor_theta)
        xys = np.array([[x, y] for x, y in zip(xs, ys)])  # shape (5 sensors, 2)

        # sensors
        tmp_x = xys[:, 0] - cx
        tmp_y = xys[:, 1] - cy
        # apply rotation
        rotated_x = tmp_x * np.cos(rotation) - tmp_y * np.sin(rotation)
        rotated_y = tmp_x * np.sin(rotation) + tmp_y * np.cos(rotation)
        # rotated x y
        self.sensor_info[:, -2:] = np.vstack([rotated_x + cx, rotated_y + cy]).T

        q = np.array([cx, cy])
        for si in range(len(self.sensor_info)):
            possible_sensor_distance = [self.sensor_max]
            possible_intersections = [self.sensor_info[si, -2:]]
            s = self.sensor_info[si, -2:] - q
            for obstacle in obstacle_list:
                boundary = obstacle.boundary
                for oi in range(len(boundary)):
                    p = boundary[oi]
                    r = boundary[(oi + 1) % len(boundary)] - boundary[oi]
                    flag, intersection, dist = self.collision(q, s, p, r)
                    if flag:
                        possible_intersections.append(intersection)
                        possible_sensor_distance.append(dist)
            distance = np.min(possible_sensor_distance)
            distance_index = np.argmin(possible_sensor_distance)
            self.sensor_info[si, 0] = distance
            self.sensor_info[si, -2:] = possible_intersections[int(distance_index)]


    @staticmethod
    def collision(q, s, p, r):
        if np.cross(r, s) != 0:  # may collision
            t = np.cross((q - p), s) / np.cross(r, s)
            u = np.cross((q - p), r) / np.cross(r, s)
            if 0 <= t <= 1 and 0 <= u <= 1:
                intersection = q + u * s
                dist = np.linalg.norm(u * s)
                return True, intersection, dist
        return False, None, None