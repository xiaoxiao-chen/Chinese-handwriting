import os
import numpy as np
import struct
import PIL.Image
 
import scipy.misc
from sklearn.utils import shuffle
import tensorflow as tf
import time
import pandas as pd
from pandas import DataFrame


data_dir = './hanwriting'
train_data_x = pd.read_csv(os.path.join(data_dir, 'train_X.csv'))
train_data_y = pd.read_csv(os.path.join(data_dir, 'train_Y.csv'))
test_data_x = pd.read_csv(os.path.join(data_dir, 'test_X.csv'))
test_data_y = pd.read_csv(os.path.join(data_dir, 'test_Y.csv'))


# shuffle样本--打乱样本顺序  #(33505,4096)(33505,140)   #(8380,4096)(8380,140)
train_data_x, train_data_y = shuffle(train_data_x, train_data_y, random_state=0)
test_data_x, test_data_y = shuffle(test_data_x, test_data_y, random_state=0)

batch_size = 128
num_batch = len(train_data_x) // batch_size #261次
 
 
X = tf.placeholder(tf.float32, [None, 64*64])
Y = tf.placeholder(tf.float32, [None, 140])
keep_prob = tf.placeholder(tf.float32)
 
def chinese_hand_write_cnn():
	x = tf.reshape(X, shape=[-1, 64, 64, 1])
	# 3 conv layers
	w_c1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
	b_c1 = tf.Variable(tf.zeros([32]))
	conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
	pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	
	w_c2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
	b_c2 = tf.Variable(tf.zeros([64]))
	conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(pool1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
	pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	
	# fully connect layer
	w_d = tf.Variable(tf.random_normal([8*32*64, 1024], stddev=0.01))
	b_d = tf.Variable(tf.zeros([1024]))
	dense = tf.reshape(pool2, [-1, w_d.get_shape().as_list()[0]])
	dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
	dense = tf.nn.dropout(dense, keep_prob)

	w_out = tf.Variable(tf.random_normal([1024, 140], stddev=0.01))
	b_out = tf.Variable(tf.zeros([140]))
	out = tf.add(tf.matmul(dense, w_out), b_out)
	return out

def train_hand_write_cnn():
	output = chinese_hand_write_cnn()

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=Y))
	optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

	accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1)), tf.float32))

	## TensorBoard
	#tf.scalar_summary("loss", loss)
	#tf.scalar_summary("accuracy", accuracy)
	#merged_summary_op = tf.merge_all_summaries()

	saver=tf.train.Saver(max_to_keep=4)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		## 命令行执行 tensorboard --logdir=./log  打开浏览器访问http://0.0.0.0:6006
		#summary_writer = tf.train.SummaryWriter('./log', graph=tf.get_default_graph())

		for e in range(50):#50次迭代，轮数
			for i in range(num_batch):  #261次，每次128个样本
				batch_x = train_data_x[i*batch_size : (i+1)*batch_size]
				batch_y = train_data_y[i*batch_size : (i+1)*batch_size]
				_, loss_ ,acc_ = sess.run([optimizer, loss, accuracy], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.5})
				saver.save(sess,'model/handwriting_model',global_step=e)

				#_, loss_, summary = sess.run([optimizer, loss, merged_summary_op], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.5})
				## 每次迭代都保存日志
				#summary_writer.add_summary(summary, e*num_batch+i)
				#print(e*num_batch+i, loss_)

				if (e*num_batch+i) % 100 == 0:
					# 每100次输出一次训练数据上的损失值、准确率
					print("step {},train_loss {}, train_acc{}".format(e*num_batch+i, loss_, acc_))

					print('test_acc %g' %  sess.run(accuracy,feed_dict={X: test_data_x[:500], Y: test_data_y[:500], keep_prob: 1.}))


start=time.time()
train_hand_write_cnn()
end=time.time()
print('训练用时：{}'.format(end-start))

# 训练完后，在测试集上进行测试
#print('test accuracy %g' %  sess.run(accuracy,feed_dict={X: test_data_x[:500], Y: test_data_y[:500], keep_prob: 1.}))
