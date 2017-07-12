import tensorflow as tf
import dataset
import numpy as np
sess = tf.Session()
saver = tf.train.import_meta_graph('my_test_model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./'))
graph = tf.get_default_graph()

y_pred = graph.get_tensor_by_name("y_pred:0")

#your test path
test_path='test'

img_size = 128

#more than one class can be added according to your requirement
classes = ['pan', 'npan']
num_classes = len(classes)

#num_channels is the number of test images in the test folder
num_channels = 3
img_size_flat = img_size * img_size * num_channels
test_images, test_ids = dataset.read_test_set(test_path, img_size,classes)
x= graph.get_tensor_by_name("x:0")

x_batch = test_images.reshape(num_channels, img_size_flat)
y_true = graph.get_tensor_by_name("y_true:0")
y_test_images = np.zeros((num_channels, 2))
feed_dict_testing = {x: x_batch, y_true: y_test_images}
print(sess.run(y_pred, feed_dict=feed_dict_testing))
