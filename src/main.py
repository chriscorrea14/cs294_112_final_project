import tensorflow as tf
import numpy as np
from random import shuffle
from dataset import generate_dataset, SDF, generate_numpy_dataset
import pickle

SDF_DIMENSION = (3,3,4)
SDF_RESOLUTION = .02
# 6 for Fanuc, 7 for YuMi
ARM_DIMENSION = 6
LEARNING_RATE = 5e-4
ITERATIONS = 40
BATCH_SIZE = 20
RETRAIN = False

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
    sdfs, sdf_indices, states, actions = generate_numpy_dataset()
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
    saver.save(session, "../models/model.ckpt")

def load_model(session):
    saver = tf.train.Saver()
    saver.restore(session, "../models/model.ckpt")

def evaluate(session, predicted_action, sdf_ph, state_ph):
    # position = np.random.normal(scale=0.1, size=3) + np.array([1,0,1])
    position = np.array([1,.3,1])
    print("box position:", position)
    sdf = SDF()
    sdf.add_box(position, (.5, .7, .1))

    states = []
    state = np.array([0.0] * 6)
    for _ in range(100):
        a = session.run(predicted_action, feed_dict={sdf_ph: [sdf.data], state_ph: [state]})[0]
        state = state + a
        states.append(state)

    states = np.array(states)
    # print(states)
    # file = "tmp.pkl"
    # with open(file, 'wb') as output:
    #     pickle.dump(states, output, pickle.HIGHEST_PROTOCOL)
    while True:
        raw_input("Press enter to display trajectory")
        display_trajectory(states)

def display_trajectory(trajectory):
    from geometry_msgs.msg import PoseStamped
    from replanning_demo import RobotController
    import rospy
    # file = "tmp.pkl"
    # trajectory = pickle.load(open(file, "rb"))
    rospy.init_node('robot_controller')
    robot_controller = RobotController()

    pose = PoseStamped()
    # pose.header.frame_id = robot.get_planning_frame()
    box_position = [0.80702869, 0.05520982, 0.93043837]
    pose.pose.position.x = box_position[0]
    pose.pose.position.y = box_position[1]
    pose.pose.position.z = box_position[2]
    pose.pose.orientation.w = 1
    pose.pose.orientation.x = 0
    pose.pose.orientation.y = 0
    pose.pose.orientation.z = 0
    robot_controller.scene.add_box("aaa", pose, size=(0.5, .7, 0.1))
    rospy.sleep(2)

    for point in trajectory:
        robot_controller.publish_joints(point)
        rospy.sleep(.1)

if __name__ == "__main__":
    session, loss, update_op, predicted_action, action, sdf_ph, state_ph = set_up()
    if RETRAIN:
        train(session, loss, update_op, action, sdf_ph, state_ph)
    else:
        load_model(session)
        evaluate(session, predicted_action, sdf_ph, state_ph)