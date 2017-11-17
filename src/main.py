import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from random import shuffle
from dataset import generate_dataset, SDF
import pickle
import sys
import argparse
import time
from controllers import BCcontroller, MPCcontroller

SDF_DIMENSION = (1.5,2,4)
SDF_RESOLUTION = .02
# 6 for Fanuc, 7 for YuMi
ARM_DIMENSION = 6
LEARNING_RATE = 5e-4
ITERATIONS = 40
BATCH_SIZE = 20

def get_2d_model(sdf, state, num_actions, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        sdf_out = tf.layers.conv2d(sdf,     filters=32, kernel_size=8, strides=4, activation=tf.nn.relu)
        sdf_out = tf.layers.conv2d(sdf_out, filters=64, kernel_size=4, strides=2, activation=tf.nn.relu)
        sdf_out = tf.layers.conv2d(sdf_out, filters=64, kernel_size=3, strides=1, activation=tf.nn.relu)
        
        flattened = tf.contrib.layers.flatten(sdf_out)
        out = tf.concat((flattened, state), axis=1)

        out = tf.layers.dense(out, 256,         activation=tf.nn.relu)
        out = tf.layers.dense(out, 256,         activation=tf.nn.relu)
        out = tf.layers.dense(out, num_actions, activation=None)
    return out

def get_3d_model(sdf, state, num_actions, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        sdf_out = tf.expand_dims(sdf, -1)
        sdf_out = tf.layers.conv3d(sdf_out, filters=32, kernel_size=8, strides=4, activation=tf.nn.relu)
        sdf_out = tf.layers.conv3d(sdf_out, filters=32, kernel_size=4, strides=2, activation=tf.nn.relu)
        sdf_out = tf.layers.conv3d(sdf_out, filters=64, kernel_size=3, strides=1, activation=tf.nn.relu)
        
        flattened = tf.contrib.layers.flatten(sdf_out)
        out = tf.concat((flattened, state), axis=1)

        out = tf.layers.dense(out, 256,         activation=tf.nn.relu)
        out = tf.layers.dense(out, 256,         activation=tf.nn.relu)
        out = tf.layers.dense(out, num_actions, activation=None)
    return out

# def get_model(sdf, state, num_actions, scope, reuse=False):
#     with tf.variable_scope(scope, reuse=reuse):
#         sdf_out = tf.expand_dims(sdf, -1)
#         print sdf_out.get_shape()
#         sdf_out = tf.layers.conv3d(sdf_out, filters=32, kernel_size=8, strides=4, activation=tf.nn.relu)
#         print sdf_out.get_shape()
#     return sdf_out

def set_up():
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    session = tf.Session(config=tf_config)

    size = [dimension / SDF_RESOLUTION for dimension in SDF_DIMENSION]
    sdf_ph = tf.placeholder(tf.uint8, [None] + size)
    sdf_ph_float = tf.cast(sdf_ph, tf.float32)
    state_ph = tf.placeholder(tf.float32, [None, ARM_DIMENSION])

    predicted_action = get_3d_model(sdf_ph_float, state_ph, ARM_DIMENSION, "policy", reuse=False)
    action = tf.placeholder(tf.float32, [None, ARM_DIMENSION])

    loss = tf.reduce_mean(tf.square(predicted_action - action))
    update_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    
    session.__enter__()
    tf.global_variables_initializer().run()

    return session, loss, update_op, predicted_action, action, sdf_ph, state_ph
    
def train(session, loss, update_op, action, sdf_ph, state_ph):
    saver = tf.train.Saver()
    sdfs, sdf_indices, states, actions = generate_dataset(use_numpy=True)
    indices = np.arange(states.shape[0])

    for i in range(ITERATIONS):
        shuffle(indices)
        batch = indices[:BATCH_SIZE]
        sdf_batch = sdfs[sdf_indices[batch]]

        if i % 10 == 0:
            current_loss = session.run(loss, feed_dict={
                sdf_ph: sdf_batch, 
                state_ph: states[batch], 
                action: actions[batch]
            })
            print("Loss at iteration", i, ": ", current_loss)
        session.run(update_op, feed_dict={
                sdf_ph: sdf_batch, 
                state_ph: states[batch], 
                action: actions[batch]
            })
    saver.save(session, "./models/model.ckpt")

def load_model(session):
    saver = tf.train.Saver()
    saver.restore(session, "./models/model.ckpt")

def evaluate(session, 
             predicted_action, 
             sdf_ph, 
             state_ph, 
             controller, 
             horizon=100, 
             box_position=np.array([1,0,1]), 
             state=np.array([0.0] * 6), 
             goal_state=np.array([-0.0974195, 1.3523, 0.682611, 0.156142, 0.675658, -0.122225])):
    from geometry_msgs.msg import PoseStamped
    from replanning_demo import RobotController, add_obstacle
    import rospy
    rospy.init_node('robot_controller')
    robot_controller = RobotController()
    add_obstacle(box_position)

    sdf = SDF()
    sdf.add_box(box_position, (.5, .7, .1))

    states = []
    start_time = time.time()
    for _ in range(horizon):
        robot_controller.publish_joints(state)
        state = state + controller.action(state, goal_state, sdf)
        states.append(states)
    print "Execution time:", time.time() - start_time
    return np.array(states)

def display_trajectory(trajectory, box_position=np.array([1,0,1]), iterations=10):
    from geometry_msgs.msg import PoseStamped
    from replanning_demo import RobotController, add_obstacle
    import rospy

    rospy.init_node('robot_controller')
    robot_controller = RobotController()
    add_obstacle(box_position)

    for _ in range(iterations):
        raw_input("Press enter to display trajectory")
        for point in trajectory:
            if hasattr(point, "positions"):
                point = point.positions
            robot_controller.publish_joints(point)
            rospy.sleep(.05)

def visualize_sdf():
    sdfs, sdf_indices, states, actions = generate_dataset()
    plt.gray()
    # for scale in np.linspace(0,1,20):
    for i in range(30):
        scale = 12/20.
        img = sdfs[i][int(scale*SDF_DIMENSION[0]/SDF_RESOLUTION)]
        plt.imshow(img)
        plt.show()
    sys.exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-visualize_sdf', '-v', action='store_true')
    parser.add_argument('-retrain_model', '-r', action='store_true')
    parser.add_argument('-load_model', '-l', action='store_true')
    args = parser.parse_args()

    if args.visualize_sdf:
        visualize_sdf()
    session, loss, update_op, predicted_action, action, sdf_ph, state_ph = set_up()
    if args.load_model:
        load_model(session)
    if args.retrain_model:
        train(session, loss, update_op, action, sdf_ph, state_ph)
    else:
        box_position=np.array([1,.25,1])
        # controller = MPCcontroller(session, predicted_action, sdf_ph, state_ph, ARM_DIMENSION)
        controller = BCcontroller(session, predicted_action, sdf_ph, state_ph)
        trajectory = evaluate(session, predicted_action, sdf_ph, state_ph, controller, box_position=box_position)
        # display_trajectory(trajectory, box_position=box_position)

        # steps = []
        # for y_pos in np.linspace(-.5, .5, 30):
        #     box_position = np.array([1, y_pos, 1])
        #     trajectory = evaluate(session, predicted_action, sdf_ph, state_ph, horizon=10, box_position=box_position)
        #     steps.append(trajectory[-1])
        # steps = np.vstack(steps)
        # np.save("./dagger/steps.npy", steps)