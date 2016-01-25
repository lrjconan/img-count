import tensorflow as tf
import numpy as np

G = np.array([[0, 3, 3, 0, 0, 0],
  [0, 0, 2, 3, 0, 0],
  [0, 0, 0, 0, 2, 0],
  [0, 0, 0, 0, 4, 2],
  [0, 0, 0, 0, 0, 3],
  [0, 0, 0, 0, 0, 0]])

with tf.Session() as sess:
    result = tf.user_ops.max_flow(G)
    print result.eval()

