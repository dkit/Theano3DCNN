from Layers import HiddenLayer, LeNetConvPoolLayer
from theano.tensor import TensorType
import theano
from logistic_sgd import LogisticRegression
import numpy as np

T = theano.tensor
floatX = theano.config.floatX
intX = 'int32'
dtensor5Float = TensorType(floatX, (False,)*5)
dtensor5Int = TensorType(intX, (False,)*5)
dtensor5IntVector = TensorType(intX, (False,))

class Network(object):
    def __init__(self, input_shape, convolution_layer_shapes, pooling_layer_shapes, n_classes, learning_rate, rng):
        self.rng = rng
        self.index = T.ivector()
        self.input_shape = input_shape
        self.convolution_layer_shapes = convolution_layer_shapes
        self.pooling_layer_shapes = pooling_layer_shapes
        self.layers = []
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.build_network()

    def build_network(self):
        # allocate symbolic variables for the data
        self.x = dtensor5Float('x')  # the data is presented as rasterized images
        self.y = dtensor5IntVector('y') # and again

        self.training_data = theano.shared(np.zeros(self.input_shape,dtype=floatX), borrow=True)
        self.training_data_labels = theano.shared(np.zeros((self.input_shape[0], ),dtype=dtensor5IntVector), borrow=True)
        self.testing_data = theano.shared(np.zeros(self.input_shape,dtype=floatX), borrow=True)
        self.testing_data_labels = theano.shared(np.zeros((self.input_shape[0], ),dtype=dtensor5IntVector), borrow=True)

        input_to_layer = self.x
        input_shape_to_layer = self.input_shape

        for i in range(0, len(self.convolution_layer_shapes)):
            print("Setting input layer shape to")
            print(input_shape_to_layer);
            layer = LeNetConvPoolLayer(
                self.rng,
                input=input_to_layer,
                image_shape=input_shape_to_layer,
                filter_shape=self.convolution_layer_shapes[i],
                poolsize=self.pooling_layer_shapes[i]
            )	
            self.layers.append(layer)
            input_to_layer = layer.output
            input_shape_to_layer = layer.output_shape()

        layer2_input = input_to_layer.flatten(2)

        # construct a fully-connected sigmoidal layer
        self.layer2 = HiddenLayer(
            self.rng,
            input=layer2_input,
            n_in=np.prod(input_shape_to_layer[1:]),
            n_out=500,
            activation=T.tanh
        )

        # classify the values of the fully-connected sigmoidal layer
        self.layer3 = LogisticRegression(input=self.layer2.output, n_in=500, n_out=self.n_classes)

        #self.vy = dtensor5IntVector('vy') # and again
        self.batch_x = dtensor5Float('batch_x')  # the data is presented as rasterized images
        self.batch_y = dtensor5IntVector('batch_y') # and again

        self.validate_model = theano.function(
            inputs=[],
            outputs=self.layer3.errors(self.y),
            givens={
                self.x: self.testing_data,
                self.y: self.testing_data_labels
            }
        )

        # create a list of all model parameters to be fit by gradient descent
        self.params = self.layer3.params + self.layer2.params
        for i in self.layers:
            self.params += i.params
        #self.params = self.layers[0].params

        # the cost we minimize during training is the NLL of the model
        self.cost = self.layer3.negative_log_likelihood(self.y)

        # create a list of gradients for all model parameters
        self.grads = T.grad(self.cost, self.params)
        # train_model is a function that updates the model parameters by
        # SGD Since this model has many parameters, it would be tedious to
        # manually create an update rule for each model parameter. We thus
        # create the updates list by automatically looping over all
        # (params[i], grads[i]) pairs.
        self.updates = [
            (param_i, param_i - self.learning_rate * grad_i)
            for param_i, grad_i in zip(self.params, self.grads)
        ]	

        self.train_model = theano.function(
            inputs=[],
            outputs=self.cost,
            updates=self.updates,
            givens={
                self.x: self.training_data,
                self.y: self.training_data_labels
            }
        )

    def train(self, training_data_set, training_data_set_labels):
        self.training_data.set_value(training_data_set)
        self.training_data_labels.set_value(training_data_set_labels)
        self.minibatch_avg_cost = self.train_model()
        return self.minibatch_avg_cost
        
    def validate(self, testing_data_set, testing_data_set_labels):
        self.testing_data.set_value(testing_data_set)
        self.testing_data_labels.set_value(testing_data_set_labels)
        self.minibatch_avg_cost = self.validate_model()
        return self.minibatch_avg_cost
