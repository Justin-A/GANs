import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../mnist/data/", one_hot = True)
X = tf.placeholder(tf.float32, [None, 28 * 28]) # MNIST = 28 * 28
Z = tf.placeholder(tf.float32, [None, 128]) # Random Noise

# Generator
G_Weight1 = tf.Variable(tf.random_normal([128, 256], stddev = 0.01))
G_Weight2 = tf.Variable(tf.random_normal([256, 28 * 28], stddev = 0.01))
G_Bias1 = tf.Variable(tf.zeros([256]))
G_Bias2 = tf.Variable(tf.zeros([28 * 28]))

def generator(noise):
	G_Hidden_Layer = tf.nn.relu(tf.matmul(noise, G_Weight1) + G_Bias1)
	output = tf.nn.sigmoid(tf.matmul(G_Hidden_Layer, G_Weight2) + G_Bias2)
	return output

# Discriminator
D_Weight1 = tf.Variable(tf.random_normal([28 * 28, 256], stddev = 0.01))
D_Weight2 = tf.Variable(tf.random_normal([256, 1], stddev = 0.01))
D_Bias1 = tf.Varaible(tf.zeros([256]))
D_Bias2 = tf.Variable(tf.zeros([1]))

def discriminator(inputs):
	D_Hidden_Layer = tf.nn.relu(tf.matmul(inputs, D_Weight1) + D_Bias1)
	output = tf.nn.sigmoid(tf.matmul(D_Hidden_Layer, D_Weight2) + D_Bias2)
	return output

# Main
G = generator(Z)
Loss_G = -tf.reduce_mean(tf.log(discriminator(X)) + tf.log(1 - discriminator(G)))
Loss_D = -tf.reduce_mean(tf.log(discriminator(G)))

Train_D = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(loss_D, var_list = [D_Weight1, D_Bias1, D_Weight2, D_Bias2])
Train_G = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(loss_G, var_list = [G_Weight1, G_Bias1, G_Weight2, G_Bias2])

session = tf.Session()
sess.run(tf.globaal_variables_initializer())

# Training & Testing
noise_test = np.random.noraml(size = (10, 128))
batch_size = 200
epochs = 200
for epoch in range(epochs):
	for i in range(int(mnist.train.num_examples / batch_size)):
		batch_xs, _ = mnist.train.next_batch(batch_size)
		noise = np.random.normal(size = (batch_size, 128))
		sess.run(train_D, feed_dict = {X: batch_xs, Z: noise})
		sess.run(train_G, feed_dict = {Z: noise})

	if epoch == 0 or (epoch + 1) % 10 == 0:
		samples = sess.run(G, feed_dict = {Z:noise_test})

		fig, ax = plt.subplots(1, 10, figsize = (10, 1))
		for i in range(10):
			ax[i].set_axis_off()
			ax[i].imshow(np.reshape(samples[i], (28, 28)))
		plt.savefig('samples_ex/{}.png'.format(str(epoch).zfill(3)), bbox_inches = 'tight')
		plt.close(fig)