#!/usr/bin/env python

import tensorflow as tf
import math
import os
import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data

# Define parameters
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_integer('epoch_number', None, 'Number of epochs to run trainer.')
flags.DEFINE_integer("batch_size", 128,
                     "indicates batch size in a single gpu, default is 1024")
flags.DEFINE_integer("thread_number", 1, "Number of thread to read data")
flags.DEFINE_integer("min_after_dequeue", 100,
                     "indicates min_after_dequeue of shuffle queue")
flags.DEFINE_string("output_dir", "./tensorboard/",
                    "indicates training output")
flags.DEFINE_string("model", "deep",
                    "Model to train, option model: deep, linear")
flags.DEFINE_string("optimizer", "sgd", "optimizer to import")
flags.DEFINE_integer('hidden1', 10, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 20, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('steps_to_validate', 10,
                     'Steps to validate and print loss')
flags.DEFINE_string("mode", "train",
                    "Option mode: train, train_from_scratch, inference")
# For distributed
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

# Hyperparameters
learning_rate = FLAGS.learning_rate
batch_size = FLAGS.batch_size



def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)


def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    

def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        print "ps pod join"
        server.join()
    elif FLAGS.job_name == "worker":
        print "wk pod join"
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):
            mnist = input_data.read_data_sets('mnist_data',one_hot=True)
            n_batch = mnist.train.num_examples // batch_size
            print "kaishi jisuan"
            x = tf.placeholder(tf.float32,[None,784])
            y = tf.placeholder(tf.float32,[None,10])
            x_image = tf.reshape(x,[-1,28,28,1])

            W_conv1 = weight_variable([5,5,1,32])
            b_conv1 = bias_variable([32])

            h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
            h_pool1 = max_pool_2x2(h_conv1)

            W_conv2 = weight_variable([5,5,32,64])
            b_conv2 = bias_variable([64])

            h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
            h_pool2 = max_pool_2x2(h_conv2)

            W_fc1 = weight_variable([7*7*64,1024])
            b_fc1 = bias_variable([1024])

            h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)

            keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

            W_fc2 = weight_variable([1024,10])
            b_fc2 = bias_variable([10])

            prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)

            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
            
            global_step = tf.Variable(0, name='global_step', trainable=False)
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy,global_step=global_step)


            correct_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

            saver = tf.train.Saver()
            init_op = tf.global_variables_initializer()

        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                 logdir="./checkpoint/",
                                 init_op=init_op,
                                 saver=saver,
                                 global_step=global_step,
                                 save_model_secs=60)

        with sv.managed_session(server.target) as sess:
            step = 0
            while not sv.should_stop() and step < 3:
                print "jiedia jisuan "
                # Get coordinator and run queues to read data
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord,sess=sess)

                try:
                    while not coord.should_stop():
                        # Run train op
                        for batch in range(n_batch):         
                            batch_xs,batch_ys =  mnist.train.next_batch(batch_size)
                            _,loss_value, step=sess.run([train_step,cross_entropy,global_step],feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.8})
                        step = step // n_batch
                        test_acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
                        
                        print "Time " + str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))) +"Iter " + str(step) + ", Testing Accuracy= " + str(test_acc)
                        
                except tf.errors.OutOfRangeError:
                    print("Done training after reading all data")
                finally:
                    coord.request_stop()

                # Wait for threads to exit
                coord.join(threads)


if __name__ == "__main__":
    tf.app.run()