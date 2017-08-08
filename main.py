import tensorflow as tf

""" Prints object information. """
node1 = tf.constant(3, dtype=tf.float32)
node2 = tf.constant(4.0)
print(node1, node2)

""" Prints tensor values. """
session = tf.Session()
print(session.run([node1, node2]))
