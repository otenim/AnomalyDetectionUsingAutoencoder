from keras.datasets import mnist, fashion_mnist
from models import load_model
import numpy as np
import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math

curdir = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('--optimizer', choices=['adam','sgd','adagrad'], default='adam')
parser.add_argument('--loss', choices=['mean_squared_error', 'binary_crossentropy'], default='mean_squared_error')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--test_samples', type=int, default=50)
parser.add_argument('--result', default=os.path.join(curdir, 'result.png'))

def main(args):

    # prepare normal dataset (Mnist)
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train / 255. # normalize into [0,1]
    x_test = x_test / 255.

    # prapare abnormal dataset (Fashion Mnist)
    (_, _), (x_abnormal, _) = fashion_mnist.load_data()
    x_abnormal = x_abnormal / 255.

    # sample args.test_samples images from eaech of x_test and x_abnormal
    perm = np.random.permutation(args.test_samples)
    x_test = x_test[perm][:args.test_samples]
    x_abnormal = x_abnormal[perm][:args.test_samples]

    # train each model and test their capabilities of anomaly deteciton
    model_names = ['autoencoder', 'deep_autoencoder', 'convolutional_autoencoder']
    for model_name in model_names:
        # instantiate model
        model = load_model(model_name)

        # reshape input data according to the model's input tensor
        if model_name == 'convolutional_autoencoder':
            x_train = x_train.reshape(-1,28,28,1)
            x_test = x_test.reshape(-1,28,28,1)
            x_abnormal = x_abnormal.reshape(-1,28,28,1)
        elif model_name == 'autoencoder' or model_name == 'deep_autoencoder':
            x_train = x_train.reshape(-1,28*28)
            x_test = x_test.reshape(-1,28*28)
            x_abnormal = x_abnormal.reshape(-1,28*28)
        else:
            raise ValueError('Unknown model_name %s was given' % model_name)

        # compile model
        model.compile(optimizer=args.optimizer, loss=args.loss)

        # train on only normal training data
        model.fit(
            x=x_train,
            y=x_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

        # test
        x_concat = np.concatenate([x_test, x_abnormal], axis=0)
        losses = []
        for x in x_concat:
            # compule loss for each test sample
            x = np.expand_dims(x, axis=0)
            loss = model.test_on_batch(x, x)
            losses.append(loss)

        # plot
        plt.plot(range(len(losses)), losses, linestyle='-', linewidth=1, label=model_name)

        # delete model for saving memory
        del model

    # create graph
    plt.legend(loc='best')
    plt.grid()
    plt.xlabel('sample index')
    plt.ylabel('loss')
    plt.savefig(args.result)
    plt.clf()




if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
