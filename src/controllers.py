import numpy as np
from sklearn.preprocessing import normalize

# cost fns
def l2_cost_fn(states, goal_state):
    costs = []
    for state in states:
        costs.append(np.linalg.norm(goal_state - state))
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

class MPCcontroller(object):
    def __init__(self, 
                 session, 
                 predicted_action, 
                 sdf_ph, 
                 state_ph, 
                 arm_dimension, 
                 cost_fn=l2_cost_fn,
                 num_simulated_paths=30,
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

    def random_action(self):
        actions = np.random.uniform(-1, 1, size=(self.num_simulated_paths, self.arm_dimension))
        actions = normalize(actions, axis=1) / 4.
        return actions

    def action(self, state, goal_state, sdf):
        states = np.array([state] * self.num_simulated_paths)
        sdfs = np.array([sdf.data] * self.num_simulated_paths)
        print sdfs.shape

        initial_action_set = False
        for _ in range(self.num_random):
            actions = self.random_action()
            if not initial_action_set:
                initial_actions = actions                
            states = states + actions
        for _ in range(self.num_chomp):
            actions = self.session.run(
                    self.predicted_action, 
                    feed_dict={self.sdf_ph: sdfs, self.state_ph: states}
                )
            states = states + actions

        costs = self.cost_fn(states, goal_state)
        return initial_actions[np.argmin(costs)]