import tensorflow as tf
from mnist_read import read, package_data
import cv2
import numpy as np

learning_rate = 0.01
training_epochs = 60
batch_size = 100
display_step = 1

def ascii_show(image, size=28):
    with open("./scope.txt", "a") as wf:

        for y in range(28):
             row = ""
             for x in image[y,:]:
                 row += '{0: <6}'.format(x)
             wf.write(row)
             wf.write("\n")
             # print(row)


def inference(x):
    init = tf.constant_initializer(value=0)
    W = tf.get_variable("W", [784, 10],
                        initializer=init)
    b = tf.get_variable("b", [10],
                        initializer=init)
    output = tf.nn.softmax(tf.matmul(x, W) + b)

    w_hist = tf.summary.histogram("weights", W)
    b_hist = tf.summary.histogram("biases", b)
    y_hist = tf.summary.histogram("output", output)

    return output


def loss(output, y):
    dot_product = y * tf.log(output)

    # Reduction along axis 0 collapses each column into a single
    # value, whereas reduction along axis 1 collapses each row
    # into a single value. In general, reduction along axis i
    # collapses the ith dimension of a tensor to size 1.
    xentropy = -tf.reduce_sum(dot_product, axis=1)

    loss = tf.reduce_mean(xentropy)

    return loss


def training(cost, global_step):
    tf.summary.scalar("cost", cost)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(cost, global_step=global_step)

    return train_op


def evaluate(output, y):
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar("validation error", (1.0 - accuracy))

    return accuracy




if __name__ == '__main__':
    # Parameters

    with tf.Graph().as_default():
        # mnist data image of shape 28*28=784
        x = tf.placeholder("float", [None, 784])
        # 0-9 digits recognition => 10 classes
        y = tf.placeholder("float", [None, 10])
        output = inference(x)
        cost = loss(output, y)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = training(cost, global_step)
        eval_op = evaluate(output, y)
        summary_op = tf.summary.merge_all()
        # summary_op = tf.merge_all_summaries()
        saver = tf.train.Saver()
        sess = tf.Session()
        data = read(path='MNIST_data')
        data_number = next(data)
        # summary_writer = tf.train.SummaryWriter("logistic_logs/",
        #                                         graph_def=sess.graph_def)
        summary_writer = tf.summary.FileWriter("logistic_logs/",
                                               graph_def=sess.graph_def)

        # init_op = tf.initialize_all_variables()
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        # Training cycle
        for epoch in range(training_epochs-1):
            avg_cost = 0.
            total_batch = int(data_number / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                # one batch of data for validation data set
                mbatch_x, mbatch_y = package_data(data, batch_size)
                # Fit training using batch data
                feed_dict = {x: mbatch_x, y: mbatch_y}
                # row, col = mbatch_x.shape
                # for i in range(row):
                #     img = mbatch_x[i, :]
                #     ascii_show(img.reshape(28,28))


                sess.run(train_op, feed_dict=feed_dict)
                # Compute average loss
                # minibatch_cost = sess.run(cost,
                #                           feed_dict=feed_dict)

                # print(sess.run(cost, feed_dict=feed_dict))
                avg_cost += sess.run(cost, feed_dict=feed_dict) / total_batch
                # print(avg_cost)

            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost =", "{:.9f}".format(avg_cost))
                validation_images, validation_labels = package_data(data, batch_size)
                accuracy = sess.run(eval_op, feed_dict={x: validation_images, y: validation_labels})

                print("Validation Error:", (1 - accuracy))

                summary_str = sess.run(summary_op, feed_dict={x: mbatch_x, y: mbatch_y})
                summary_writer.add_summary(summary_str, sess.run(global_step))

                saver.save(sess, "logistic_logs/model-checkpoint", global_step=global_step)
        print("Optimization Finished!")
        test_data = read(dataset='testing', path='MNIST_data')
        size = next(test_data)
        test_images, test_labels = package_data(test_data, size)
        test_feed_dict = {
            x: test_images,
            y: test_labels
        }
        accuracy = sess.run(eval_op, feed_dict=test_feed_dict)
        print("Test Accuracy:", accuracy)