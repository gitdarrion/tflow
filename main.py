import os
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

""" Prints object information. """
node1 = tf.constant(3, dtype=tf.float32)
node2 = tf.constant(4.0)
print(node1, node2)

""" Prints tensor values. """
session = tf.Session()
print(session.run([node1, node2]))

""" Operations. """
node3 = tf.add(node1, node2)
print("node3: ", node3)
print("session.run(node3): ", session.run(node3))
