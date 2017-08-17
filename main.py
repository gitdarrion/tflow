import os
import tensorflow as tf


os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

""" Print object information. """

node1 = tf.constant(3, dtype=tf.float32)
node2 = tf.constant(4.0)
print(node1, node2)

""" Print tensor values. """

session = tf.Session()
print(session.run([node1, node2]))

""" Execute operations on tensors. """

### FUNCTIONS ###
node3 = tf.add(node1, node2)
print("node3: ", node3)
print("session.run(node3): ", session.run(node3))

### PLACEHOLDERS ###
a = tf.placeholder(dtype=tf.float32)
b = tf.placeholder(dtype=tf.float32)
adder_node = a + b
print(session.run(adder_node, {a: 3, b: 4}))
print(session.run(adder_node, {a: [1,3], b: [2,4]}))
add_and_triple = adder_node * 3
print(session.run(add_and_triple, {a: 3, b: 4.5}))

### VARIABLES ###
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(dtype=tf.float32)
linear_model = W * x + b
init = tf.global_variables_initializer()
session.run(init)
print(session.run(linear_model, {x:[1,2,3,4]}))

### REDUCE SUM / ERROR ###
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(session.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
