import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


X = tf.placeholder(tf.float32, shape=[None, 2200])

D_W1 = tf.Variable(xavier_init([2200, 128]))
D_b1 = tf.Variable(tf.zeros(shape=[128]))

D_W2 = tf.Variable(xavier_init([128, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]

saver = tf.train.Saver()

Z = tf.placeholder(tf.float32, shape=[None, 2200])

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit

D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(Z)

#D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
# G_loss = -tf.reduce_mean(tf.log(D_fake))

# Alternative losses:
# -------------------
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)

mb_size = 64

sess = tf.Session()
saver = tf.train.Saver(max_to_keep=5)
sess.run(tf.global_variables_initializer())



i = 0

def discriminator_train(X_mb, samples):
    if not os.path.exists('/tmp/GAN'):
        os.makedirs('/tmp/GAN')
    for it in range(10000):
        #X_mb, _ = mnist.train.next_batch(mb_size)
        #X_mb = sess.run(X, feed_dict={X: sample_Z(64, 2200)})
        #samples = sess.run(Z, feed_dict={Z: sample_Z(64, 2200)})
        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: samples})
    #saver = tf.train.Saver(max_to_keep=5)
    save_path = saver.save(sess, "/tmp/GAN/GAN_preference_based_model.ckpt")
    print("Model saved in path: %s" % save_path)
    D_real_probability, _ = sess.run([D_real, D_logit_real], feed_dict={X: X_mb})
    D_fake_probability, _ = sess.run([D_real, D_logit_real], feed_dict={X: samples})
    return (np.array(D_real_probability), np.array(D_fake_probability))

def discriminator_test(X_mb, samples):
    for it in range(5000):
        if it % 1000 == 0:
            # generated data
            #samples = sess.run(Z, feed_dict={Z:sample_Z(16, 784)})
            # training data
            #X_mb, _ = mnist.train.next_batch(16)
            D_real_probability, _ = sess.run([D_real, D_logit_real], feed_dict={X: X_mb})
            D_fake_probability, _ = sess.run([D_real, D_logit_real], feed_dict={X: samples})
            #print('Iter: {}'.format(it))
            #print('D_real', D_real_probability,'D_fake', D_fake_probability)
            #save_path = saver.save(sess, "/tmp/vanillaGAN_model.ckpt")
            #print("Model saved in path: %s" % save_path)

        #X_mb, _ = mnist.train.next_batch(mb_size)
        #X_mb = sess.run(X, feed_dict={X: sample_Z(64, 2200)})
        #samples = sess.run(Z, feed_dict={Z: sample_Z(64, 2200)})
        #_, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(mb_size, 2200)})
        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: samples})
    #saver = tf.train.Saver(max_to_keep=5)
    save_path = saver.save(sess, "/tmp/GAN/GAN_preference_based_model.ckpt")
    print("Model re-saved in path: %s, iters" % save_path, it)
    #tf.reset_default_graph()
    saver.restore(sess, "/tmp/GAN/GAN_preference_based_model.ckpt")
    print("Preference based Model restored.")
    return (np.array(D_real_probability), np.array(D_fake_probability))

#samples = sess.run(Z, feed_dict={Z: sample_Z(64, 2200)})
#X_mb = sess.run(X, feed_dict={X: sample_Z(64, 2200)})
#X_mb, _ = mnist.train.next_batch(64)
#discriminator_train(X_mb, samples)