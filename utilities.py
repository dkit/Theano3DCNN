import numpy as np
import theano

def random_matrix(shape, np_rng, name=None):
	return theano.shared(np.require(np_rng.randn(*shape), dtype=theano.config.floatX),
			borrow=True, name=name)
