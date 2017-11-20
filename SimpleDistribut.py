#!/usr/bin/env python

import tensorflow as tf
import math
import os
import numpy as np
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
            w = tf.Variable(tf.zeros([784,10]))
            b = tf.Variable(tf.zeros(10))
            prediction = tf.nn.softmax(tf.matmul(x,w)+b)
            loss = tf.reduce_mean(tf.square(y-prediction))
            global_step = tf.Variable(0, name='global_step', trainable=False)
            train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss,global_step=global_step)
            
            correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
            
            saver = tf.train.Saver()
            init_op = tf.global_variables_initializer()
            
            tf.summary.scalar('loss', loss)
            tf.summary.scalar('accuracy', accuracy)

            summary_op = tf.summary.merge_all()

        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                 logdir="./checkpoint/",
                                 init_op=init_op,
                                 summary_op=summary_op,
                                 saver=saver,
                                 global_step=global_step,
                                 save_model_secs=60)

        with sv.managed_session(server.target) as sess:
            step = 0
            while not sv.should_stop() and step < 70:
                print "jiedia jisuan "
                # Get coordinator and run queues to read data
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord,sess=sess)

                try:
                    while not coord.should_stop():
                        # Run train op
                        for batch in range(n_batch):         
                            batch_xs,batch_ys =  mnist.train.next_batch(batch_size)
                            _,loss_value, step=sess.run([train_step,loss,global_step],feed_dict={x:batch_xs,y:batch_ys})
                        step = step // n_batch
                        test_acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
                        
                        print "Iter " + str(step) + ", Testing Accuracy= " + str(test_acc)
                        coord.request_stop()
                        
                except tf.errors.OutOfRangeError:
                    print("Done training after reading all data")
                finally:
                    coord.request_stop()

                # Wait for threads to exit
                coord.join(threads)


if __name__ == "__main__":
    tf.app.run()