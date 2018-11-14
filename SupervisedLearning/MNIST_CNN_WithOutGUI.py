import tensorflow as tf
import numpy as np
import READ_MNIST


# training iterations
TRAINING_ITER_NUM = 1000

BATCH_SIZE = 64

# test iterations
TEST_ITER_NUM = 10

# data and labels
train_images = READ_MNIST.load_train_images()
train_labels = READ_MNIST.load_train_labels()
test_images = READ_MNIST.load_test_images()
test_labels = READ_MNIST.load_test_labels()


# function to build a conv layer
def conv2d(x, w, b, activation_function=tf.nn.relu):
    return activation_function(tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME') + b)


# function to build a max pool layer
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# function to build a fc layer
def fc(x, w, b, activation_function=tf.nn.relu):
    return activation_function(tf.matmul(x, w) + b)


if __name__ == '__main__':
    # define placeholder for input data x and true output data y
    # None means the number of input is not fixed, you can input many x and y
    x_original_data = tf.placeholder(tf.float32, [None, 784])
    y_true = tf.placeholder(tf.float32, [None, 10])

    # data is [None, 784], we have to reshape it to [None, 28, 28]
    # the 1 in [None, 28, 28, 1] means the channel of image is 1
    # the gray image's channel is 1, RGB image's channel is 3, RGBA image's channel image is 4
    x_input = tf.reshape(x_original_data, [-1, 28, 28, 1])

    '''
        define the network
        28*28*1 image -> 32 conv core with 5*5*1 shape (max_pool) -> 64 conv core with 5*5*32 shape (max_pool)
        -> fc 1024 -> fc 10 output
    '''

    # 32 conv cores, shape is 5*5*1
    w_conv1 = tf.Variable(tf.random_normal([5, 5, 1, 32]))
    b_conv1 = tf.Variable(tf.random_normal([32]))
    layer_conv1 = conv2d(x_input, w_conv1, b_conv1, activation_function=tf.nn.relu)
    layer_pool1 = max_pool_2x2(layer_conv1)

    # 64 conv cores, shape is 5*5*1
    w_conv2 = tf.Variable(tf.random_normal([5, 5, 32, 64]))
    b_conv2 = tf.Variable(tf.random_normal([64]))
    layer_conv2 = conv2d(layer_pool1, w_conv2, b_conv2, activation_function=tf.nn.relu)
    layer_pool2 = max_pool_2x2(layer_conv2)

    # flat the layer 2
    layer_pool2_flat = tf.reshape(layer_pool2, [-1, 7*7*64])

    # full connected layer,
    w_fc1 = tf.Variable(tf.random_normal([7*7*64, 1024]))
    b_fc1 = b_conv1 = tf.Variable(tf.random_normal([1024]))
    layer_fc1 = fc(layer_pool2_flat, w_fc1, b_fc1, activation_function=tf.nn.relu)

    # output layer
    w_fc2 = tf.Variable(tf.random_normal([1024, 10]))
    b_fc2 = tf.Variable(tf.random_normal([10]))
    predict = tf.matmul(layer_fc1, w_fc2) + b_fc2

    # define the loss function
    entropy = tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=y_true)
    loss_function = tf.reduce_mean(entropy)

    # define the optimal
    optimal = tf.train.AdamOptimizer(0.001).minimize(loss_function)

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
        _, loss = sess.run([optimal, loss_function], feed_dict={x_original_data: x_data, y_true: y_data})

        # get average loss
        loss /= BATCH_SIZE

        # output the training details
        print('Training Epoch {0}, loss: {1}'.format(i, loss))

    # test on test set, and calculate the accuracy
    accuracy = 0
    for i in range(len(test_labels)):
        result = sess.run(predict, feed_dict={x_original_data: test_images[i].reshape([1, 784])})
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
        predict_label = sess.run(predict, feed_dict={x_original_data: test_images[test_index[i]].reshape([1, 784])})
        predict_label = np.argmax(predict_label[0])
        result = predict_label == test_labels[test_index[i]]
        print(str(i) + " \t\t " + str(test_labels[test_index[i]])
              + " \t\t " + str(predict_label) + " \t\t " + str(result))
