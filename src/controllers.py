import numpy as np
from sklearn.preprocessing import normalize
from moveit_msgs.srv import GetStateValidity, GetStateValidityRequest, GetStateValidityResponse
import rospy
import time

collision_srv = rospy.ServiceProxy('/check_state_validity', GetStateValidity)
collision_srv.wait_for_service()
req = GetStateValidityRequest()

def in_collision(state):
    req.robot_state.joint_state.header.stamp = rospy.Time.now()
    req.robot_state.joint_state.name = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
    req.robot_state.joint_state.position = state
    res = collision_srv.call(req)
    # print res
    return not res.valid

# cost fns
def l2_cost_fn(second_states, final_states, goal_state):
    costs = []
    for second_state, final_state in zip(second_states, final_states):
        # if in_collision(second_state):
        #     costs.append(float('inf'))
        # else:
        costs.append(np.linalg.norm(goal_state - final_state))
    return costs

def bc_similarity_cost_fn(states, chomp_state):
    costs = []
    for state in states:
        costs.append(np.linalg.norm(state - chomp_state))
    return costs

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
        return action

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
            print "random", time.time() - a
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
        print "chomp", time.time() - a

        a = time.time()
        costs = self.cost_fn(second_states, states, goal_state)
        print "cost", time.time() - a
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
                 horizon=5):
        self.session = session
        self.predicted_action = predicted_action
        self.sdf_ph = sdf_ph
        self.state_ph = state_ph
        self.cost_fn = cost_fn
        self.arm_dimension = arm_dimension
        self.num_simulated_paths = num_simulated_paths
        self.horizon = horizon

    def action(self, state, goal_state, sdf):
        chomp_state = state
        chomp_actions = []
        a = time.time()
        for _ in range(self.horizon):
            action = self.session.run(
                    self.predicted_action, 
                    feed_dict={self.sdf_ph: [sdf.data], self.state_ph: chomp_state.reshape(1,6)}
                )[0]
            chomp_actions.append(action)
            chomp_state = chomp_state + action
        print "CHOMP time:", time.time() - a

        a = time.time()
        states = np.array([state] * self.num_simulated_paths)
        actions = np.zeros(states.shape)
        for t in range(self.horizon):
            actions = []
            sigma = .001
            for i in range(self.num_simulated_paths):
                while True:
                    action = np.random.normal(chomp_actions[t], scale=sigma)
                    if not in_collision(states[i] + action):
                        break
                    sigma *= 2
                actions.append(action)
            actions = np.array(actions)
            states = states + actions
            if t == 0:
                initial_actions = actions
        print "MPC time", time.time() - a

        a = time.time()
        costs = self.cost_fn(states, chomp_state)
        print "cost", time.time() - a
        # print chomp_actions[0]
        # print initial_actions[np.argmin(costs)], "\n"
        return initial_actions[np.argmin(costs)]

if __name__ == "__main__":
    import rospy

    rospy.init_node('robot_controller')
    in_collision([0, 0, 0, 0, 0, 0])
    print "\n\n\n\n"
    in_collision([0, .21, 0, 0, 0, 0])
    print "\n\n\n\n"
    in_collision([0, .43, 0, 0, 0, 0])