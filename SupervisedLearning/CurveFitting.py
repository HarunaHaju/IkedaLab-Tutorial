import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math

ITER_NUM = 3000

# Define the data
# y = sin(x) + noise
# 0 <= noise <= 0.05
x_data = np.linspace(-math.pi, math.pi, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.sin(x_data) + noise


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
    x_input = tf.placeholder(tf.float32, [None, 1])
    y_true = tf.placeholder(tf.float32, [None, 1])

    # define the network
    # you could change the parameters here and see the result
    layer_1 = add_layer(x_input, 1, 10, activation_function=tf.nn.tanh)
    prediction = add_layer(layer_1, 10, 1, activation_function=tf.nn.tanh)

    # define the loss function
    loss_function = tf.reduce_mean(tf.square(y_true - prediction))

    # define the learning rate
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(1.0, global_step, 100, 0.96, staircase=False)

    # define the optimal
    # you could try different optimizer here and see the result
    optimal = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function, global_step=global_step)
    # optimal = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss_function, global_step=global_step)
    # optimal = tf.train.FtrlOptimizer(learning_rate).minimize(loss_function, global_step=global_step)

    # define the initializer and session
    init = tf.global_variables_initializer()
    sess = tf.Session()

    # initialize
    sess.run(init)

    # plot the data
    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    ax.scatter(x_data, y_data)

    # plot the loss
    loss_fig = fig.add_subplot(2, 1, 2)
    loss_fig.set_title("Loss")

    # interactive mode on, could dynamic refresh the picture
    plt.ion()
    plt.show()

    # define the loss history
    loss_history = []

    for i in range(ITER_NUM):
        # training
        _, loss = sess.run([optimal, loss_function], feed_dict={x_input: x_data, y_true: y_data})

        # add loss to history
        loss_history.append(loss)

        # output the training details
        print('Epoch {0}: {1}'.format(i, loss))

        if i % 10 == 0:
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass

            # get the prediction value
            prediction_value = sess.run(prediction, feed_dict={x_input: x_data})

            # plot the prediction value
            lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
            ax.set_title("Fitting Result at Step:"+str(i))

            # plot the loss
            loss_fig.plot(np.arange(len(loss_history)), loss_history, 'r-')

            plt.pause(0.05)

    # close interactive mode
    plt.ioff()
    plt.show()
