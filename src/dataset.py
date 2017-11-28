import pickle
from os import listdir
import numpy as np

SDF_DIMENSION = (1.5,2,4)
SDF_RESOLUTION = .02

class SDF():
    def __init__(self):
        size = [int(dimension / SDF_RESOLUTION) for dimension in SDF_DIMENSION]
        self.data = np.zeros(size, dtype=np.uint8)

    def add_box(self, position, size):
        for i in np.arange(0, SDF_DIMENSION[0], SDF_RESOLUTION):
            if i < position[0] - size[0]/2. or i > position[0] + size[0]/2.:
                continue
            for j in np.arange(-SDF_DIMENSION[1]/2., SDF_DIMENSION[1]/2., SDF_RESOLUTION):
                if j < position[1] - size[1]/2. or j > position[1] + size[1]/2.:
                    continue
                for k in np.arange(-SDF_DIMENSION[2]/2., SDF_DIMENSION[2]/2., SDF_RESOLUTION):
                    if k < position[2] - size[2]/2. or k > position[2] + size[2]/2.:
                        continue
                    l = int(i/SDF_RESOLUTION)
                    m = int((SDF_DIMENSION[1]/2. + j)/SDF_RESOLUTION)
                    n = int((SDF_DIMENSION[2]/2. + k)/SDF_RESOLUTION)
                    self.data[l,m,n] = 1

def generate_dataset():
    def generate_sdf(i):
        file = str(i) + "_box_position.pkl"
        sdf = SDF()
        position = pickle.load(open("./data/trajectories/" + file, "rb"))
        sdf.add_box(position, (.5, .7, .1))
        return sdf.data

    def generate_state_action(i):
        file = str(i) + "_plan.pkl"
        trajectory = pickle.load(open("./data/trajectories/" + file, "rb"))
        trajectory = trajectory.joint_trajectory.points
        states, actions = [], []
        for j in range(len(trajectory) - 1):
            action = np.subtract(trajectory[j+1].positions, trajectory[j].positions)
            states.append(trajectory[j].positions)
            actions.append(action)
        states = np.array(states)
        actions = np.array(actions)
        return states, actions

    # def generate_dagger_data():
    #     files = listdir("./data/trajectories/")
    #     all_states, all_actions, sdfs, sdf_indices = [], [], [], []
    #     for i in range(int(len(files)/3)):
    #         file_start = "./data/dagger/" + str(i) + "_"
    #         states = np.load(file_start + "dagger_states.npy")
    #         actions = np.load(file_start + "dagger_actions.npy")
    #         box_position = np.load(file_start + "box_position.npy")
    #         sdf = SDF()
    #         sdf.add_box(box_position, (.5, .7, .1))

    #         sdfs.append(sdf.data)
    #         sdf_indices.append([i] * state.)

    files = listdir("./data/trajectories/")
    sdfs, states, actions, sdf_indices = [], [], [], []
    for i in range(int(len(files)/2)):
        sdfs.append(generate_sdf(i))
        state, action = generate_state_action(i)
        sdf_indices.append([i] * state.shape[0])
        states.append(state)
        actions.append(action)

    sdf_indices = np.hstack(sdf_indices)
    sdfs = np.array(sdfs)
    states = np.vstack(states)
    actions = np.vstack(actions)

    np.save("./data/dataset/sdfs.npy", sdfs)
    np.save("./data/dataset/sdf_indices.npy", sdf_indices)
    np.save("./data/dataset/states.npy", states)
    np.save("./data/dataset/actions.npy", actions)

def load_dataset():
    sdfs = np.load("./data/dataset/sdfs.npy")
    sdf_indices = np.load("./data/dataset/sdf_indices.npy")
    states = np.load("./data/dataset/states.npy")
    actions = np.load("./data/dataset/actions.npy")
    return sdfs, sdf_indices, states, actions

def display_chomp_trajectories():
    from geometry_msgs.msg import PoseStamped
    from main import display_trajectory
    import rospy

    # for i in range(30):
    for i in [60]:
        file = str(i) + "_plan.pkl"
        trajectory = pickle.load(open("./data/trajectories/" + file, "rb"))
        trajectory = trajectory.joint_trajectory.points
        file = str(i) + "_box_position.pkl"
        box_position = pickle.load(open("./data/trajectories/" + file, "rb"))
        display_trajectory(trajectory, box_position, iterations=1)

def check_for_empty_trajectories():
    files = listdir("./data/trajectories/")
    for file in files:
        if file.endswith("plan.pkl"):
            trajectory = pickle.load(open("./data/trajectories/" + file, "rb"))
            if len(trajectory.joint_trajectory.points) == 0:
                print(file)

if __name__ == "__main__":
    # display_chomp_trajectories()
    generate_dataset()
    # check_for_empty_trajectories()
