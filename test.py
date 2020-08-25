import numpy as np
import tensorflow as tf
import time
import scipy.io as sio





sess=tf.Session()
saver = tf.train.import_meta_graph('./model/model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./model'))


graph = tf.get_default_graph()
x = graph.get_tensor_by_name("input_data:0")
keep_prob = graph.get_tensor_by_name("keep_prob:0")
y = graph.get_tensor_by_name("MLP_decoder/decode_output:0")


load_fn = './data/Segundo.mat'
load_data = sio.loadmat(load_fn)
test_data = load_data['data']
test_data = np.array(test_data)
img_band = test_data.shape[2]
test_data_shape = test_data.shape
test_data = test_data.reshape([test_data_shape[0]*test_data_shape[1], img_band])
test_data = 2*((test_data-test_data.min()) /
                    (test_data.max()-test_data.min()))-1


reconstruct_result = sess.run(y, feed_dict={x: test_data, keep_prob: 1})


save_fn = './result/reconstruct_result.mat'
sio.savemat(save_fn, {'reconstruct_result': reconstruct_result})


