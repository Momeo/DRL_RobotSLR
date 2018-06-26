import numpy as np


class polygon(object):
    location = None
    boundary = None
    r = 1
    numPoint = 4

    def __init__(self, boundary):
        self.boundary = boundary
        self.location = np.mean(boundary, 0)
        self.r = 50
        self.numPoint = len(self.boundary)


class Quad(object):
    location = None
    boundary = None
    r = 1
    numPoint = 4

    def __init__(self, x, y, w, h):
        self.boundary = np.array([[x, y],
                                 [x+w, y],
                                 [x+w, y+h],
                                 [x, y+h]])
        self.location = np.array([x+w/2, y+h/2])
        self.r = max(w/2, h/2)
        self.numPoint = len(self.boundary)


class Obstacle(object):
    location = None
    boundary = None
    r = 1

    def __init__(self, location, w):
        self.r *= w
        self.location = location
        self.boundary = self.boundary*w + location


class Squ(Obstacle):
    numPoint = 4
    x = np.array([0.5, 0.5, -0.5, -0.5])*2
    y = np.array([0.5, -0.5, -0.5, 0.5])*2

    boundary = np.vstack((x, y)).T
    r = 1

    def __init__(self, location, w):
        super(Squ, self).__init__(location, w)


class Cylinder(Obstacle):
    numPoint = 10
    theta = np.linspace(0, 2 * np.pi, numPoint)
    y = np.sin(theta)
    x = np.cos(theta)
    boundary = np.vstack((x, y)).T
    r = 1

    def __init__(self, location, w):
        super(Cylinder, self).__init__(location, w)


def generate(n):
    obstacle_list = []
    for i in range(n):
        w = np.random.randint(40, 60)
        loc = generate_unit(obstacle_list, w)
        obstacle = Squ(loc, w)
        obstacle_list.append(obstacle)
    return obstacle_list


def generate_env1():
    obstacle_list = []
    w = 60
    obstacle_list.append(Squ(np.array([125, 125]), w))
    obstacle_list.append(Squ(np.array([125, 375]), w))
    obstacle_list.append(Squ(np.array([375, 125]), w))
    obstacle_list.append(Squ(np.array([375, 375]), w))
    return obstacle_list


def generate_env2():
    obstacle_list = []
    obstacle_list.append(Quad(0, 0, 300, 30))
    obstacle_list.append(Quad(200, 200, 300, 30))
    obstacle_list.append(Quad(0, 350, 300, 30))
    return obstacle_list


def generate_env3():
    obstacle_list = []
    b = np.array([[0, 0],
                  [300, 0],
                  [300, 50],
                  [250, 50],
                  [250, 20],
                  [0, 20]])
    obstacle_list.append(polygon(b))

    return obstacle_list


def generate_env4():
    obstacle_list = []
    b = np.array([[0, 0],
                  [400, 150],
                  [400, 200],
                  [0, 50]])
    obstacle_list.append(polygon(b))
    b = np.array([[70, 200],
                  [500, 500],
                  [500, 550],
                  [70, 250]])
    obstacle_list.append(polygon(b))
    b = np.array([[0, 350],
                  [150, 350],
                  [150, 400],
                  [0, 400]])
    obstacle_list.append(polygon(b))
    b = np.array([[200, 400],
                  [400, 400],
                  [500, 500],
                  [200, 500]])
    obstacle_list.append(polygon(b))
    return obstacle_list


def generate_dynamic():
    obstacle_list = []
    w = 40
    obstacle_list.append(Squ(np.array([125, 125]), w))
    obstacle_list.append(Squ(np.array([125, 375]), w))
    obstacle_list.append(Squ(np.array([250, 250]), w))
    obstacle_list.append(Squ(np.array([250, 250]), w))
    obstacle_list.append(Squ(np.array([375, 125]), w))
    obstacle_list.append(Squ(np.array([375, 375]), w))
    obstacle_offset = np.random.randint(-3, 3, size=[len(obstacle_list), 2]).tolist()
    return obstacle_list, obstacle_offset


def generate_unit(obstacle_list, w, car_loc=None):
    if obstacle_list:
        Flag = True
        while Flag:
            Flag = False
            loc = np.random.randint(0, 500, size=2)
            for o in obstacle_list:
                if car_loc is None:
                    if np.linalg.norm(loc - o.location) < w+o.r:
                        Flag = True
                else:
                    if np.linalg.norm(loc - o.location) < w + o.r and np.linalg.norm(loc - car_loc) < w + o.r:
                        Flag = True
        return loc
    return np.random.randint(0, 500, size=2)


def dynamic_refresh(obstacle_list, obstacle_offset, car_loc):
    vertical_max = horizen_max = 500
    for j in range(len(obstacle_list[:-1])):
        i = obstacle_list[j]
        if (i.boundary[:, 0]<0).all() or (i.boundary[:,0]>vertical_max).all() or (i.boundary[:,1]<0).all() or (i.boundary[:,1]>horizen_max).all():
            loc = generate_unit(obstacle_list[:-1], 40, car_loc)
            obstacle_list[j] = Squ(loc, 40)
            obstacle_offset[j] = np.random.randint(-2, 2, size=[1, 2]).tolist()
    return obstacle_list, obstacle_offset
