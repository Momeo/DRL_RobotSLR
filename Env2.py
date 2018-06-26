"""
环境2：动态环境
"""

import numpy as np
import pyglet
from Sensor import SLR
import obstacle

pyglet.clock.set_fps_limit(10000)


class CarEnv(object):
    n_sensor = 10
    action_dim = 1
    state_dim = n_sensor
    viewer = None
    viewer_xy = (500, 500)
    sensor_max = 200.
    start_point = [450, 300]
    speed = 40.
    trajectory = None
    dt = 0.1
    obstacle_list = None
    obstacle_offset = None

    def __init__(self, discrete_action=False):
        self.is_discrete_action = discrete_action
        if discrete_action:
            self.actions = [-1, 0, 1]
        else:
            self.action_bound = [-1, 1]

        self.terminal = False
        # node1 (x, y, r, w, l),
        self.car_info = np.array([0, 0, 0, 20, 40], dtype=np.float64)   # car coordination
        self.sensor = SLR(self.n_sensor)

    def step(self, action):

        if self.is_discrete_action:
            action = self.actions[action]
        else:
            action = np.clip(action, *self.action_bound)[0]
        self.car_info[2] += action * np.pi/30  # max r = 6 degree
        self.car_info[:2] = self.car_info[:2] + \
                            self.speed * self.dt * np.array([np.cos(self.car_info[2]), np.sin(self.car_info[2])])
        self.trajectory.append(self.car_info[:2])
        for i in range(len(self.obstacle_list[:-1])):
            self.obstacle_list[i].boundary += self.obstacle_offset[i]
        self.obstacle_list, self.obstacle_offset = obstacle.dynamic_refresh(self.obstacle_list, self.obstacle_offset, self.car_info[:2])

        self._update_sensor()
        s = self._get_state()
        dist = self.euclidean_distance(self.trajectory[-1], self.trajectory[0])
        r = -0.1
        if dist>150:
            r = dist / 500
        if self.terminal:
            r = -1
        return s, r, self.terminal

    def reset(self):
        self.terminal = False
        if self.viewer:
            self.viewer.close()
        self.viewer = None
        self.obstacle_list, self.obstacle_offset = obstacle.generate_dynamic()
        self.start_point = obstacle.generate_unit(self.obstacle_list, 40)
        rotation = np.random.rand() * 2 * np.pi
        self.car_info[:3] = np.array([*self.start_point, rotation])
        obstacle_box_window = obstacle.Squ(np.array([250, 250]), 250)
        self.obstacle_list.append(obstacle_box_window)

        self.trajectory = []
        self.trajectory.append(self.start_point)
        self._update_sensor()
        return self._get_state()

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(*self.viewer_xy, self.car_info, self.sensor.sensor_info, self.obstacle_list)
        self.viewer.render(self.obstacle_list)

    def sample_action(self):
        if self.is_discrete_action:
            a = np.random.choice(list(range(3)))
        else:
            a = np.random.uniform(*self.action_bound, size=self.action_dim)
        return a

    def set_fps(self, fps=30):
        pyglet.clock.set_fps_limit(fps)

    def _get_state(self):
        s = self.sensor.sensor_info[:, 0].flatten()/self.sensor_max
        return s

    def _update_sensor(self):
        self.sensor.update(self.car_info[:2], self.car_info[2], self.obstacle_list)
        distance = np.min(self.sensor.sensor_info[:, 0])
        if distance < self.car_info[-1]/4:
            self.terminal = True

    @staticmethod
    def manhattan_distance(loc1, loc2):
        return np.sum(np.abs(loc1 - loc2))

    @staticmethod
    def euclidean_distance(loc1, loc2):
        return np.linalg.norm(loc1 - loc2)


class Viewer(pyglet.window.Window):
    color = {
        'background': [1]*3 + [1]
    }
    fps_display = pyglet.clock.ClockDisplay()

    def __init__(self, width, height, car_info, sensor_info, obstacle_list):
        super(Viewer, self).__init__(width, height, resizable=False, caption='test1', vsync=False)  # vsync=False to not use the monitor FPS
        self.set_location(x=80, y=10)
        pyglet.gl.glClearColor(*self.color['background'])

        self.car_info = car_info
        self.sensor_info = sensor_info
        self.batch = pyglet.graphics.Batch()
        background = pyglet.graphics.OrderedGroup(0)
        foreground = pyglet.graphics.OrderedGroup(1)

        self.sensors = []
        line_coord = [0, 0] * 2
        c = (73, 73, 73) * 2
        for i in range(len(self.sensor_info)):
            self.sensors.append(self.batch.add(2, pyglet.gl.GL_LINES, foreground, ('v2f', line_coord), ('c3B', c)))
        # TODO: Car
        car_box = [0, 0] * 4
        self.car = self.batch.add(4, pyglet.gl.GL_QUADS, foreground, ('v2f', car_box), ('c3B', (249, 86, 86) * 4))

        # TODO: 障碍物list
        self.obstacle = []
        for i in obstacle_list[:-1]:
            num = i.numPoint
            if num == 4:
                self.obstacle.append(self.batch.add(num, pyglet.gl.GL_QUADS, background, ('v2f', i.boundary.flatten()), ('c3B', (134, 181, 244) * num)))
            else:
                self.obstacle.append(self.batch.add(num, pyglet.gl.GL_POLYGON, background, ('v2f', i.boundary.flatten()),('c3B', (134, 181, 244) * num)))
        #self.obstacle.append(self.batch.add(4, pyglet.gl.GL_POLYGON, background, ('v2f', [0,480,20,480,20,500,0,500]),('c3B', (255, 255, 0) * 4)))

    def render(self, obstacle_list):
        pyglet.clock.tick()
        self._update(obstacle_list)
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()
        # self.fps_display.draw()

    def _update(self, obstacle_list):
        cx, cy, r, w, l = self.car_info

        # sensors
        for i, sensor in enumerate(self.sensors):
            sensor.vertices = [cx, cy, *self.sensor_info[i, -2:]]

        # car
        xys = [
            [cx + l / 2, cy + w / 2],
            [cx - l / 2, cy + w / 2],
            [cx - l / 2, cy - w / 2],
            [cx + l / 2, cy - w / 2],
        ]
        r_xys = []
        for x, y in xys:
            tempX = x - cx
            tempY = y - cy
            # apply rotation
            rotatedX = tempX * np.cos(r) - tempY * np.sin(r)
            rotatedY = tempX * np.sin(r) + tempY * np.cos(r)
            # rotated x y
            x = rotatedX + cx
            y = rotatedY + cy
            r_xys += [x, y]
        self.car.vertices = r_xys
        for i in range(len(obstacle_list[:-1])):
            self.obstacle[i].vertices = obstacle_list[i].boundary.flatten()


if __name__ == '__main__':
    np.random.seed(1)
    env = CarEnv()
    env.set_fps(60)
    for ep in range(20):
        s = env.reset()
        # for t in range(100):
        while True:
            env.render()
            s, r, done = env.step(env.sample_action())
            if done:
                break