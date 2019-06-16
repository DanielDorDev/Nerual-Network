import random
import numpy as np
import numpy
#   import matplotlib.pyplot as plt


# Daniel Dor 205581770
# Function for relu derivative, div = 1 if > 0, , as learned.
def relu_derivative(x):
    return np.maximum(0, np.sign(x))


# Relu as learned.
def relu(x):
    return np.maximum(x, 0)


# Soft max with subtract of max (prevent high values).
def soft_max(x):
    exp_x = np.exp(x - np.max(x, axis=1)[:, np.newaxis])
    return exp_x / np.sum(exp_x, axis=1)[:, np.newaxis]


# Pixel normalize to [-1, 1]
def pixel_norm(vector):
    return (vector - 128) / 128


# NN object, will be used for prediction, will be trained by the user.
class NeuralNetwork:

    # Init the number of classification classes, shape of the input.
    def __init__(self, class_count, shape):

        # Prevent deterministic random, and init weights, bias, reset them.
        random.seed(a=None, version=2)
        self.class_num = class_count
        self.w1 = 0
        self.w2 = 0
        self.b1 = 0
        self.b2 = 0
        self.reset(shape)

    # Train the nn, given epoch times, and learning rate.
    def train(self, x, y, epoch, lr):

        # Create vector for output layer.
        y_len = y.shape[0]
        label_vector = np.zeros((y_len, self.class_num))
        for i in range(y_len):
            label_vector[i, y[i]] = 1

        # Save for self graph plot(check development of the neural network).
        #   itr_num = []
        #   abc_percent = []

        # Init batch size, root of number of examples.
        batch_size = np.floor(np.sqrt(y.shape[0]))

        # Create zip, for shuffling.
        data = list(zip(x, y))

        # Vector length.
        x_len = x.shape[1]

        # Every epoch shuffle data, create batch, backprop the batch(update accrue in that function).
        for i in range(epoch):
            random.shuffle(data)
            for batch in np.array_split(data, batch_size):
                batch_len = batch.shape[0]
                x_batch, y_batch = zip(*batch)

                # pixel the data first, for [-1, 1] values, and shape them.
                self.backprop(pixel_norm(np.reshape(a=x_batch, newshape=(batch_len, x_len)).astype(np.float64)),
                              np.reshape(a=y_batch, newshape=batch_len), lr)

    # For prediction, self check, predict 5000 first values(the most far from the batched range)
            """
            answer = nn.predict(x[:5000])
            train_num = x[:5000].shape[0]
            correct = train_num - np.count_nonzero(answer != y[:5000])
            print(correct / train_num)
            itr_num.append(i)
            abc_percent.append(correct / train_num)

        plt.scatter(itr_num, abc_percent)
        plt.title("Acc grow")
        plt.xlabel("itr")
        plt.ylabel("acc")
        plt.show()
"""
    # Reset values, bias won't change, the weights init by random.
    # I read in article, that np.sqrt(2 / x_shape) is good way to set the random, and 512 gives good results.
    def reset(self, x_shape):
        self.w1 = np.random.randn(x_shape, 35) * np.sqrt(2 / x_shape)
        self.w2 = np.random.randn(35, self.class_num) * np.sqrt(2 / x_shape)
        self.b1 = 0
        self.b2 = 0

    # Predict, use feedforward, take the y_hat result and return arg max values.
    def predict(self, x):
        layer1, y_hat = self.feedforward(pixel_norm(x.astype(np.float64)))
        return y_hat.argmax(axis=1)

    # Feed forward, calculate h1, h output, and return the results.
    def feedforward(self, x):
        layer1 = relu(np.dot(x, self.w1) + self.b1)
        y_hat = soft_max(np.dot(layer1, self.w2) + self.b2)
        return layer1, y_hat

    # Calculate gradient, and update weights.
    def backprop(self, x, y, lr_rate):

        # Create vector for output layer.
        y_len = y.shape[0]
        label_vector = np.zeros((y_len, self.class_num))
        for i in range(y_len):
            label_vector[i, y[i]] = 1

        # Calculate layer outputs.
        a1, y_hat = self.feedforward(x)

        # Calculate the backprop, div values for weight update.
        dz2 = y_hat - label_vector
        ah = np.dot(dz2, self.w2.T)
        dzh = relu_derivative(np.dot(x, self.w1) + self.b1)

        # lr divided by the number of samples in batch.
        lr = lr_rate / y_len

        self.w1 += - lr * np.dot(x.T, dzh * ah)  # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/dw1
        self.b1 += - lr * np.sum(ah * dzh, axis=0)  # dL/dz2 * dz2/dh1 * dh1/dz1 * 1 (dz1/db1 = 1)
        self.w2 += - lr * np.dot(a1.T, dz2)  # dL/dz2 * dz2/dw2
        self.b2 += - lr * np.sum(dz2, axis=0)  # dL/dz2 * 1 (dz2/db2 = 1)


if __name__ == "__main__":
    # Upload data, and create nn, 10 classes, with pixel vector size.
    train_x = numpy.loadtxt("train_x", dtype=numpy.uint8)
    train_y = numpy.loadtxt("train_y", dtype=numpy.uint8)

    nn = NeuralNetwork(10, train_x.shape[1])

    # Train data, i saw that 25 epochs is enough for good result.
    # After i got to stable gradient, i try to get to the lowest point(try to fit even better for random data).
    nn.train(train_x, train_y, 40, 0.1)
    np.savetxt('test_y', nn.predict(numpy.loadtxt("test_x", dtype=numpy.uint8)), fmt='%d')
    print("Finished")

