import numpy as np
from sklearn.preprocessing import normalize
from moveit_msgs.srv import GetStateValidity, GetStateValidityRequest, GetStateValidityResponse
from moveit_msgs.srv import GetPositionFK, GetPositionFKRequest, GetPositionFKResponse
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray
import rospy
import time
from copy import copy

collision_srv = rospy.ServiceProxy('/check_state_validity', GetStateValidity)
collision_srv.wait_for_service()
collision_req = GetStateValidityRequest()
collision_req.robot_state.joint_state.name = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

fk_srv = rospy.ServiceProxy('/compute_fk', GetPositionFK)
fk_srv.wait_for_service()
fk_req = GetPositionFKRequest()
fk_req.robot_state.joint_state.name = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
fk_req.fk_link_names = ["tool0"]
fk_req.header.frame_id = "base_link"

marker_pub = rospy.Publisher('visualiation_markers', MarkerArray, queue_size=100)


def in_collision(state):
    collision_req.robot_state.joint_state.header.stamp = rospy.Time.now()
    collision_req.robot_state.joint_state.position = state
    res = collision_srv.call(collision_req)
    return not res.valid

def get_fk(state):
    fk_req.robot_state.joint_state.header.stamp = rospy.Time.now()
    fk_req.robot_state.joint_state.position = state
    res = fk_srv.call(fk_req)
    return res.pose_stamped[0].pose.position

def display_markers(points, colors):
    markerArray = MarkerArray()
    for i in range(len(points)):
        marker = Marker()
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.header.frame_id = "base_link"
        marker.id = i
        marker.pose.position = points[i]
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.color.r = colors[i][0]
        marker.color.g = colors[i][1]
        marker.color.b = colors[i][2]
        marker.color.a = 1
        markerArray.markers.append(marker)
    marker_pub.publish(markerArray)

# cost fns
def l2_cost_fn(second_states, final_states, goal_state):
    costs = []
    for second_state, final_state in zip(second_states, final_states):
        costs.append(np.linalg.norm(goal_state - final_state))
    return costs

def bc_similarity_cost_fn(states, chomp_state):
    costs = []
    for state in states:
        costs.append(np.linalg.norm(state - chomp_state))
    return costs

def bc_similarity_FK_cost_fn(states, chomp_state):
    costs = []
    chomp_fk = get_fk(chomp_state)
    for state in states:
        state_fk = get_fk(state)
        costs.append(np.linalg.norm(state_fk - chomp_fk))


class BCcontroller(object):
    def __init__(self, session, predicted_action, sdf_ph, state_ph):
        self.session = session
        self.predicted_action = predicted_action
        self.sdf_ph = sdf_ph
        self.state_ph = state_ph

    def action(self, state, goal_state, sdf):
        action = self.session.run(
                self.predicted_action, 
                feed_dict={self.sdf_ph: [sdf.data], self.state_ph: [state]}
            )[0]
        return (action, False)

class MPCcontroller_old(object):
    def __init__(self, 
                 session, 
                 predicted_action, 
                 sdf_ph, 
                 state_ph, 
                 arm_dimension, 
                 cost_fn=l2_cost_fn,
                 num_simulated_paths=20,
                 num_random=1, 
                 num_chomp=3):
        self.session = session
        self.predicted_action = predicted_action
        self.sdf_ph = sdf_ph
        self.state_ph = state_ph
        self.cost_fn = cost_fn
        self.arm_dimension = arm_dimension
        self.num_simulated_paths = num_simulated_paths
        self.num_random = num_random
        self.num_chomp = num_chomp

    def random_action(self, state, goal_state):
        # actions = np.random.uniform(-1, 1, size=(self.num_simulated_paths, self.arm_dimension))
        actions = []
        i = 0
        while i < self.num_simulated_paths:
            # action = np.random.uniform(-1, 1, size=self.arm_dimension)
            action = np.random.normal(goal_state - state, scale=2, size=self.arm_dimension)
            action = action / (4 * np.linalg.norm(action))

            if not in_collision(state + action):
                actions.append(action)
                i += 1
        actions = np.array(actions)
        # print actions
        # actions = normalize(actions, axis=1) / 4.
        return actions

    def action(self, state, goal_state, sdf):
        states = np.array([state] * self.num_simulated_paths)
        sdfs = np.array([sdf.data] * self.num_simulated_paths)

        initial_action_set = False
        for _ in range(self.num_random):
            a = time.time()
            actions = self.random_action(state, goal_state)
            print("random", time.time() - a)
            if not initial_action_set:
                initial_actions = actions
                second_states = states + actions
            states = states + actions
        a = time.time()
        for _ in range(self.num_chomp):
            actions = self.session.run(
                    self.predicted_action, 
                    feed_dict={self.sdf_ph: sdfs, self.state_ph: states}
                )
            states = states + actions
        print("chomp", time.time() - a)

        a = time.time()
        costs = self.cost_fn(second_states, states, goal_state)
        print("cost", time.time() - a)
        return initial_actions[np.argmin(costs)]

class MPCcontroller(object):
    def __init__(self, 
                 session, 
                 predicted_action, 
                 sdf_ph, 
                 state_ph, 
                 arm_dimension, 
                 cost_fn=bc_similarity_cost_fn,
                 num_simulated_paths=30,
                 horizon=7,
                 display_points=True):
        self.session = session
        self.predicted_action = predicted_action
        self.sdf_ph = sdf_ph
        self.state_ph = state_ph
        self.cost_fn = cost_fn
        self.arm_dimension = arm_dimension
        self.num_simulated_paths = num_simulated_paths
        self.horizon = horizon
        self.display_points = display_points

    def action(self, state, goal_state, sdf):
        original_state = copy(state)
        chomp_state = state
        chomp_trajectory = [get_fk(state)]
        chomp_actions = []
        a = time.time()
        for _ in range(self.horizon):
            action = self.session.run(
                    self.predicted_action, 
                    feed_dict={self.sdf_ph: [sdf.data], self.state_ph: chomp_state.reshape(1,6)}
                )[0]
            chomp_actions.append(action)
            chomp_state = chomp_state + action
            chomp_trajectory.append(get_fk(chomp_state))
        # print "CHOMP time:", time.time() - a

        a = time.time()
        states = np.array([state] * self.num_simulated_paths)
        trajectories = [states]
        actions = np.zeros(states.shape)
        for t in range(self.horizon):
            actions = []
            sigma = .001
            for i in range(self.num_simulated_paths):
                while True:
                    action = np.random.normal(chomp_actions[t], scale=sigma)
                    scale_factor = np.linalg.norm(chomp_actions[t], ord=np.inf) / np.linalg.norm(action, ord=np.inf)
                    action = action / scale_factor
                    if not in_collision(states[i] + action):
                        break
                    sigma *= 1.25
                actions.append(action)
            actions = np.array(actions)
            states = states + actions
            trajectories.append(states)
            if t == 0:
                initial_actions = actions
        trajectories = np.array(trajectories)
        # print "MPC time", time.time() - a

        a = time.time()
        costs = self.cost_fn(states, chomp_state)
        # print "cost", time.time() - a
        action = initial_actions[np.argmin(costs)]
        action_difference = np.linalg.norm(action - chomp_actions[0])

        if self.display_points:
            a = time.time()
            mpc_trajectory = []
            for state in trajectories[:,np.argmin(costs),:]:
                mpc_trajectory.append(get_fk(state))
            # mpc_final_point = get_fk(states[np.argmin(costs)])
            colors = ([[1,0,0]] * (self.horizon+1)) + ([[0,0,1]] * (self.horizon+1))
            # display_markers([chomp_final_point, mpc_final_point], [[1,0,0],[0,0,1]])
            points = chomp_trajectory + mpc_trajectory
            display_markers(points, colors)
            # print "Display time:", time.time() - a
        return action, action_difference > .01

if __name__ == "__main__":
    import rospy

    rospy.init_node('robot_controller')
    # in_collision([0, 0, 0, 0, 0, 0])
    # print "\n\n\n\n"
    # in_collision([0, .21, 0, 0, 0, 0])
    # print "\n\n\n\n"
    # in_collision([0, .43, 0, 0, 0, 0])
    fk = get_fk([0, 0, 0, 0, 0, 0])
    display_markers(fk, 0, [1.0,1.0,0])