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
# Operations with TF functions.
node3 = tf.add(node1, node2)
print("node3: ", node3)
print("session.run(node3): ", session.run(node3))

# Operations with placeholders.
a = tf.placeholder(dtype=tf.float32)
b = tf.placeholder(dtype=tf.float32)
adder_node = a + b
print(session.run(adder_node, {a: 3, b: 4}))
print(session.run(adder_node, {a: [1,3], b: [2,4]}))
