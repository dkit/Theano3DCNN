#!/usr/bin/env python
import numpy as np
import theano
import theano.tensor.nnet.conv3d2d
from theano.tensor.signal import pool
from maxpool3d import max_pool_3d

T = theano.tensor

class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2, 2), buildFullLayer=True):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor5
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 5
        :param filter_shape: (number of filters, time, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 5
        :param image_shape: (batch size, time, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 3
        :param poolsize: the downsampling (pooling) factor (time #rows, #cols)
        """
        self.input = input
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.rng = rng
        if buildFullLayer:
            self.buildFullLayer()
		
    def calculate_conv_layer_shape(self):
        filter_shape = self.filter_shape
        image_shape = self.image_shape
        
        # This code was adapted from conv3d3d.py
        # Not all code paths have been tested!
        # just valid valid valid
        border_mode = ['valid', 'valid', 'valid']
        # reshape the output to restore its original size
        # shape = Ns, Ts, Nf, Tf, W-Wf+1, H-Hf+1
        if border_mode[1] == 'valid':
            new_dim = [
                image_shape[0],  # Ns
                image_shape[1],  # Ts
                filter_shape[0],  # Nf
                filter_shape[1],  # Tf
                image_shape[3] - filter_shape[3] + 1,
                image_shape[4] - filter_shape[4] + 1
            ]
        elif border_mode[1] == 'full':
            new_dim = [
                image_shape[0],  # Ns
                image_shape[1],  # Ts
                filter_shape[0],  # Nf
                filter_shape[1],  # Tf
                image_shape[3] + filter_shape[3] - 1,
                image_shape[4] + filter_shape[4] - 1
            ]
        elif border_mode[1] == 'same':
            raise NotImplementedError()
        else:
            raise ValueError('invalid border mode', border_mode[1])

        # now sum out along the Tf to get the output
        # but we have to sum on a diagonal through the Tf and Ts submatrix.
        if border_mode[0] == 'valid':
            if filter_shape[1] != 1:
                new_dim[1] = new_dim[1]-new_dim[3]+1
                del new_dim[3]
            else:  # for Tf==1, no sum along Tf, the Ts-axis of the output is unchanged!
               new_dim = [
                    image_shape[0],
                    image_shape[1],
                    filter_shape[0],
                    image_shape[3] - filter_shape[3] + 1,
                    image_shape[4] - filter_shape[4] + 1,
                ]
        elif border_mode[0] == 'full':
            if filter_shape[1] != 1:
                # pad out_tmp with zeros to have full convolution
                new_dim = [
                    image_shape[0],  # Ns
                    image_shape[1] + 2 * (filter_shape[1] - 1),  # Ts
                    filter_shape[0],  # Nf
                    filter_shape[1],  # Tf
                    image_shape[3] + filter_shape[3] - 1,
                    image_shape[4] + filter_shape[4] - 1
                ]
                new_dim[1] = new_dim[1]-new_dim[3]+1
                del new_dim[3]
            else:  # for tf==1, no sum along tf, the ts-axis of the output is unchanged!
               new_dim = [
                    image_shape[0],
                    image_shape[1],
                    filter_shape[0],
                    image_shape[3] + filter_shape[3] - 1,
                    image_shape[4] + filter_shape[4] - 1
                ]
        elif border_mode[0] == 'same':
            raise NotImplementedError('sequence border mode', border_mode[0])
        else:
            raise ValueError('invalid border mode', border_mode[1])
        return new_dim
            
    def output_shape(self, filter_shape=None, image_shape=None, poolsize=None):
        if filter_shape is None: filter_shape = self.filter_shape
        if image_shape is None:  image_shape = self.image_shape
        if poolsize is None:     poolsize = self.poolsize
        
        batchsize = image_shape[0]
        time = (image_shape[1]-filter_shape[1]) // poolsize[0] + 1
        channels = filter_shape[0]
        width = (image_shape[3]-filter_shape[3]) // poolsize[1] + 1
        height = (image_shape[4]-filter_shape[4]) // poolsize[2] + 1
        
        layer_shape = self.calculate_conv_layer_shape()
        layer_shape[1] = layer_shape[1]// poolsize[0]
        layer_shape[3] = layer_shape[3]// poolsize[1]
        layer_shape[4] = layer_shape[4]// poolsize[2]
        return layer_shape
        
    def buildFullLayer(self, rng=None, input=None, filter_shape=None, image_shape=None, poolsize=None):
        if rng is None:          rng = self.rng
        if input is None:        input = self.input
        if filter_shape is None: filter_shape = self.filter_shape
        if image_shape is None:  image_shape = self.image_shape
        if poolsize is None:     poolsize = self.poolsize
        
        self.conv_out = self.createConvLayer(rng, filter_shape, image_shape, poolsize)
        self.pooled_out = self.createMaxPoolLayer(self.conv_out, poolsize)
        self.output = self.createOutput(self.pooled_out)

    def createOutput(self, pooled_out=None):
        if pooled_out is None: pooled_out = self.pooled_out
        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        return T.tanh(pooled_out + self.b.dimshuffle('x','x',0,'x','x'))
		
    def createMaxPoolLayer(self, conv_out=None, poolsize=None):
        if conv_out is None: conv_out = self.conv_out
        if poolsize is None: poolsize = self.poolsize
        # pool each feature map individually, using maxpooling
        #pooled_out = pool.pool_2d(
        self.pooled_out = max_pool_3d(
            input=conv_out.dimshuffle(0,2,1,3,4),
            ds=poolsize,
            ignore_border=True
        )
        return self.pooled_out.dimshuffle(0,2,1,3,4)
		
    def createConvLayer(self, rng, filter_shape, image_shape, poolsize):
        if rng is None:          rng = self.rng
        if filter_shape is None: filter_shape = self.filter_shape
        if image_shape is None:  image_shape = self.image_shape
        if poolsize is None:     poolsize = self.poolsize

        # there are "time * num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "time num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (np.prod(filter_shape[0:1]) * np.prod(filter_shape[3:]) //
                   np.prod(poolsize))
        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        conv_out = T.nnet.conv3d2d.conv3d(
			signals=self.input,  # Ns, Ts, C, Hs, Ws
			filters=self.W,      # Nf, Tf, C, Hf, Wf
			signals_shape=image_shape,
			filters_shape=filter_shape,
			border_mode='valid')

        # store parameters of this layer
        self.params = [self.W, self.b]
		
        return conv_out
	
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]

