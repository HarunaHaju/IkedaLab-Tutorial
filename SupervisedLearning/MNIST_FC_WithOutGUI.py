import tensorflow as tf
import numpy as np
import READ_MNIST


# training iterations
TRAINING_ITER_NUM = 500

BATCH_SIZE = 128

# test iterations
TEST_ITER_NUM = 10

# data and labels
train_images = READ_MNIST.load_train_images()
train_labels = READ_MNIST.load_train_labels()
test_images = READ_MNIST.load_test_images()
test_labels = READ_MNIST.load_test_labels()


def add_layer(inputs, in_size, out_size, activation_function=None):
    # define the weight and bias
    weight = tf.Variable(tf.random_normal([in_size, out_size]))
    bias = tf.Variable(tf.random_normal([1, out_size]))

    # define the layer, if activation function is none, output the layer directly
    layer = tf.matmul(inputs, weight) + bias
    if activation_function is None:
        return layer
    else:
        return activation_function(layer)


if __name__ == '__main__':
    # define placeholder for input data x and true output data y
    # None means the number of input is not fixed, you can input many x and y
    x_input = tf.placeholder(tf.float32, [None, 784])
    y_true = tf.placeholder(tf.float32, [None, 10])

    # define the network
    # 784 ->  10
    layer_1 = add_layer(x_input, 784, 10, activation_function=None)

    # define the loss function
    entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_1, labels=y_true)
    loss_function = tf.reduce_mean(entropy)

    # define the learning rate
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(1.0, global_step, 100, 0.96, staircase=False)

    # define the optimal
    optimal = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function, global_step=global_step)

    # define the initializer and session
    init = tf.global_variables_initializer()
    sess = tf.Session()

    # initialize
    sess.run(init)

    for i in range(TRAINING_ITER_NUM):
        # random get index
        batch_index = np.random.randint(0, len(train_labels), [BATCH_SIZE])

        loss = 0

        # define the data
        x_data = np.zeros([BATCH_SIZE, 784])
        y_data = np.zeros([BATCH_SIZE, 10])

        for index in range(BATCH_SIZE):
            # get image and label, then add it into x_data and y_data
            x_ = train_images[batch_index[index]].reshape([1, 784])

            # y is one hot data
            # for example, if a image is 5
            # the y is [0 0 0 0 0 1 0 0 0 0]
            y_ = np.zeros([1, 10])
            y_[0, train_labels[batch_index[index]]] = 1

            # add the image and label into x_data and y_data
            x_data[index] = x_
            y_data[index] = y_

        # training
        _, loss = sess.run([optimal, loss_function], feed_dict={x_input: x_data, y_true: y_data})

        # get average loss
        loss /= BATCH_SIZE

        # output the training details
        print('Training Epoch {0}, loss: {1}'.format(i, loss))

    # test on test set, and calculate the accuracy
    accuracy = 0
    for i in range(len(test_labels)):
        result = sess.run(layer_1, feed_dict={x_input: test_images[i].reshape([1, 784])})
        result = np.argmax(result[0])
        if result == test_labels[i]:
            accuracy += 1

    # output the accuracy
    print("-------------------------------")
    print("Accuracy on test set: {} %".format(accuracy * 100.0 / len(test_labels)))
    print("-------------------------------")

    # test the network on test set
    # random get some data and test the network
    test_index = np.random.randint(0, len(test_images), [TEST_ITER_NUM])

    print("test_epoch \t true_label \t predict_label \t result")
    for i in range(TEST_ITER_NUM):

        # predict by network
        predict_label = sess.run(layer_1, feed_dict={x_input: test_images[test_index[i]].reshape([1, 784])})
        predict_label = np.argmax(predict_label[0])
        result = predict_label == test_labels[test_index[i]]
        print(str(i) + " \t\t " + str(test_labels[test_index[i]])
              + " \t\t " + str(predict_label) + " \t\t " + str(result))
