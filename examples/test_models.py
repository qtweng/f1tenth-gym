import time
from f110_gym.envs.base_classes import Integrator
import yaml
import gym
import numpy as np
from argparse import Namespace

from numba import njit
import tensorflow as tf

from pyglet.gl import GL_POINTS

"""
Planner Helpers
"""


@njit(fastmath=False, cache=True)
def nearest_point_on_trajectory(point, trajectory):
    """
    Return the nearest point along the given piecewise linear trajectory.

    Same as nearest_point_on_line_segment, but vectorized. This method is quite fast, time constraints should
    not be an issue so long as trajectories are not insanely long.

        Order of magnitude: trajectory length: 1000 --> 0.0002 second computation (5000fps)

    point: size 2 numpy array
    trajectory: Nx2 matrix of (x,y) trajectory waypoints
        - these must be unique. If they are not unique, a divide by 0 error will destroy the world
    """
    diffs = trajectory[1:, :] - trajectory[:-1, :]
    l2s = diffs[:, 0]**2 + diffs[:, 1]**2
    # this is equivalent to the elementwise dot product
    # dots = np.sum((point - trajectory[:-1,:]) * diffs[:,:], axis=1)
    dots = np.empty((trajectory.shape[0]-1, ))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t < 0.0] = 0.0
    t[t > 1.0] = 1.0
    # t = np.clip(dots / l2s, 0.0, 1.0)
    projections = trajectory[:-1, :] + (t*diffs.T).T
    # dists = np.linalg.norm(point - projections, axis=1)
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp*temp))
    min_dist_segment = np.argmin(dists)
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment


@njit(fastmath=False, cache=True)
def first_point_on_trajectory_intersecting_circle(point, radius, trajectory, t=0.0, wrap=False):
    """
    starts at beginning of trajectory, and find the first point one radius away from the given point along the trajectory.

    Assumes that the first segment passes within a single radius of the point

    http://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
    """
    start_i = int(t)
    start_t = t % 1.0
    first_t = None
    first_i = None
    first_p = None
    trajectory = np.ascontiguousarray(trajectory)
    for i in range(start_i, trajectory.shape[0]-1):
        start = trajectory[i, :]
        end = trajectory[i+1, :]+1e-6
        V = np.ascontiguousarray(end - start)

        a = np.dot(V, V)
        b = 2.0*np.dot(V, start - point)
        c = np.dot(start, start) + np.dot(point, point) - \
            2.0*np.dot(start, point) - radius*radius
        discriminant = b*b-4*a*c

        if discriminant < 0:
            continue
        #   print "NO INTERSECTION"
        # else:
        # if discriminant >= 0.0:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0*a)
        t2 = (-b + discriminant) / (2.0*a)
        if i == start_i:
            if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break
        elif t1 >= 0.0 and t1 <= 1.0:
            first_t = t1
            first_i = i
            first_p = start + t1 * V
            break
        elif t2 >= 0.0 and t2 <= 1.0:
            first_t = t2
            first_i = i
            first_p = start + t2 * V
            break
    # wrap around to the beginning of the trajectory if no intersection is found1
    if wrap and first_p is None:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0], :]
            end = trajectory[(i+1) % trajectory.shape[0], :]+1e-6
            V = end - start

            a = np.dot(V, V)
            b = 2.0*np.dot(V, start - point)
            c = np.dot(start, start) + np.dot(point, point) - \
                2.0*np.dot(start, point) - radius*radius
            discriminant = b*b-4*a*c

            if discriminant < 0:
                continue
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0*a)
            t2 = (-b + discriminant) / (2.0*a)
            if t1 >= 0.0 and t1 <= 1.0:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            elif t2 >= 0.0 and t2 <= 1.0:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break

    return first_p, first_i, first_t


@njit(fastmath=False, cache=True)
def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    """
    Returns actuation
    """
    waypoint_y = np.dot(np.array(
        [np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2]-position)
    speed = lookahead_point[2]
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.
    radius = 1/(2.0*waypoint_y/lookahead_distance**2)
    steering_angle = np.arctan(wheelbase/radius)
    return speed, steering_angle


class PurePursuitPlanner:
    """
    Example Planner
    """

    def __init__(self, conf, wb):
        self.wheelbase = wb
        self.conf = conf
        self.load_waypoints(conf)
        self.max_reacquire = 20.

        self.drawn_waypoints = []

    def load_waypoints(self, conf):
        """
        loads waypoints
        """
        self.waypoints = np.loadtxt(
            conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)

    def render_waypoints(self, e):
        """
        update waypoints being drawn by EnvRenderer
        """

        # points = self.waypoints

        points = np.vstack(
            (self.waypoints[:, self.conf.wpt_xind], self.waypoints[:, self.conf.wpt_yind])).T

        scaled_points = 50.*points

        for i in range(points.shape[0]):
            if len(self.drawn_waypoints) < points.shape[0]:
                b = e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.]),
                                ('c3B/stream', [183, 193, 222]))
                self.drawn_waypoints.append(b)
            else:
                self.drawn_waypoints[i].vertices = [
                    scaled_points[i, 0], scaled_points[i, 1], 0.]

    def _get_current_waypoint(self, waypoints, lookahead_distance, position, theta):
        """
        gets the current waypoint to follow
        """
        wpts = np.vstack(
            (self.waypoints[:, self.conf.wpt_xind], self.waypoints[:, self.conf.wpt_yind])).T
        nearest_point, nearest_dist, t, i = nearest_point_on_trajectory(
            position, wpts)
        if nearest_dist < lookahead_distance:
            lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(
                position, lookahead_distance, wpts, i+t, wrap=True)
            if i2 == None:
                return None
            current_waypoint = np.empty((3, ))
            # x, y
            current_waypoint[0:2] = wpts[i2, :]
            # speed
            current_waypoint[2] = waypoints[i, self.conf.wpt_vind]
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            return np.append(wpts[i, :], waypoints[i, self.conf.wpt_vind])
        else:
            return None

    def plan(self, pose_x, pose_y, pose_theta, lookahead_distance, vgain):
        """
        gives actuation given observation
        """
        position = np.array([pose_x, pose_y])
        lookahead_point = self._get_current_waypoint(
            self.waypoints, lookahead_distance, position, pose_theta)

        if lookahead_point is None:
            return 4.0, 0.0

        speed, steering_angle = get_actuation(
            pose_theta, lookahead_point, position, lookahead_distance, self.wheelbase)
        speed = vgain * speed

        return steering_angle, speed


class DAVE:
    def __init__(self):
        print("Model")
        model_name = './f1_tenth_model'
        model = tf.keras.models.load_model(model_name+'.h5')
        # try:
        self.interpreter = tf.lite.Interpreter(
            model_path=model_name+'.tflite')  # ,num_threads = args.ncpu)
        # except ImportError:
        #    print(f'Error in importing model: {ImportError}')
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_details = self.interpreter.get_output_details()

    def callback(self, ldata):
        eighth = int(len(ldata)/8)
        ldata = np.array(ldata[eighth:-eighth]).astype(np.float32)
        ldata = np.expand_dims(ldata, axis=-1)
        ldata = np.expand_dims(ldata, axis=0)
        lidar_data = ldata
        return lidar_data

    def dnn_output(self, lidar_data):
        if lidar_data is None:
            return 0.
        lidar_data = self.callback(lidar_data)
        self.interpreter.set_tensor(self.input_index, lidar_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        servo = output[0, 0]
        speed = output[0, 1]
        return servo, speed


class FTG:
    def preprocess_lidar(self, ranges):
        """ Any preprocessing of the LiDAR data can be done in this function.
            Possible Improvements: smoothing of outliers in the data and placing
            a cap on the maximum distance a point can be.
        """
        # remove quadrant of LiDAR directly behind us
        eighth = int(len(ranges)/8)
        return np.array(ranges[eighth:-eighth])

    def get_angle(self, range_index, range_len):
        """ Calculate the angle that corresponds to a given LiDAR point and
            process it into a steering angle.
            Possible improvements: smoothing of aggressive steering angles
        """
        lidar_angle = (range_index - (range_len/2)) * self.radians_per_point
        steering_angle = np.clip(lidar_angle, np.radians(-90), np.radians(90))
        return steering_angle

    def process_lidar(self, ranges):
        """ Run the disparity extender algorithm!
            Possible improvements: varying the speed based on the
            steering angle or the distance to the farthest point.
        """
        self.radians_per_point = (2*np.pi)/len(ranges)
        proc_ranges = self.preprocess_lidar(ranges)
        # obstacle correction

        i = 0
        value = 0
        adj = 0
        # scan both left to right and right to left
        while i < len(proc_ranges) - 2:
            if i >= len(proc_ranges) - 2:
                break
            if adj > 0:
                proc_ranges[i] = value
                adj -= 1
            # check edge from small to large and extend
            elif (proc_ranges[i + 1] - proc_ranges[i]) > 1:
                # extend more if the distance is smaller
                adj = int(200 / (proc_ranges[i] + 1))
                value = proc_ranges[i]
            i += 1
        i = len(proc_ranges) - 1
        while i > 1:
            if i <= 1:
                break
            if adj > 0:
                proc_ranges[i] = value
                adj -= 1
            # check edge from small to large and extend
            elif (proc_ranges[i - 1] - proc_ranges[i]) > 1:
                # extend more if the distance is smaller
                adj = int(200 / (proc_ranges[i] + 1))
                value = proc_ranges[i]
            i -= 1
        steering_angle = self.get_angle(proc_ranges.argmax(), len(proc_ranges))
        speed = 2 + (0.5 * proc_ranges[proc_ranges.argmax()])
        return steering_angle, speed


def main():
    """
    main entry point
    """

    work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312,
            'tlad': 0.82461887897713965, 'vgain': 1.375}  # 0.90338203837889}

    with open('config_example_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    # planner = PurePursuitPlanner(conf, (0.17145+0.15875)) #FlippyPlanner(speed=0.2, flip_every=1, steer=10)
    ftg = FTG()
    dave = DAVE()

    def render_callback(env_renderer):
        # custom extra drawing function

        e = env_renderer

        # update camera to follow car
        x = e.cars[0].vertices[::2]
        y = e.cars[0].vertices[1::2]
        top, bottom, left, right = max(y), min(y), min(x), max(x)
        e.score_label.x = left
        e.score_label.y = top - 700
        e.left = left - 800
        e.right = right + 800
        e.top = top + 800
        e.bottom = bottom - 800

        # planner.render_waypoints(env_renderer)

    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1, timestep=0.01, integrator=Integrator.RK4, starts=np.array([[conf.sx, conf.sy, conf.stheta]]), checkpoints=[
                   (-1.043387605536935, 6.182245854975881), (-11.119006505707265, 12.196284277915074), (-35.28917215489346, 24.549809980247254), (-47.28584233778769, 7.649503337057018), (-34.33850049129457, -7.868348672295764), (-9.921228138520522, -7.54480924235594), (conf.sx, conf.sy)])
    env.add_render_callback(render_callback)

    obs = env.reset()
    env.render()

    laptime = 0.0
    start = time.time()
    done = False
    while not done:
        # actions[0] = planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], work['tlad'], work['vgain'])
        steer, speed = dave.dnn_output(obs)
        print(steer, speed)
        obs, step_reward, done, info = env.step(np.array([[steer, speed]]))
        print(step_reward, info['poses_x'][0], info['poses_y'][0])
        laptime += step_reward
        env.render(mode='human')

    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)


if __name__ == '__main__':
    main()
