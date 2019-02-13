import numpy as np
from cs231n.layers import *
#from cs231n.layer_utils import *

class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        #self.normalization = normalization
        #self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        
        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        sig = weight_scale
        for i in range(self.num_layers - 1):
          self.params['W' + str(i+1)] = np.random.normal(0.0, sig, [input_dim, hidden_dims[i]])
          self.params['b' + str(i+1)] = np.zeros([hidden_dims[i]])
          input_dim = hidden_dims[i]
        self.params['W' + str(self.num_layers)] = np.random.normal(0.0, sig, [input_dim, num_classes])
        self.params['b' + str(self.num_layers)] = np.zeros([num_classes])  
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        
        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def affine_forward(self, x, w, b):
        """
        Computes the forward pass for an affine (fully-connected) layer.

        The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
        examples, where each example x[i] has shape (d_1, ..., d_k). We will
        reshape each input into a vector of dimension D = d_1 * ... * d_k, and
        then transform it to an output vector of dimension M.

        Inputs:
        - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
        - w: A numpy array of weights, of shape (D, M)
        - b: A numpy array of biases, of shape (M,)

        Returns a tuple of:
        - out: output, of shape (N, M)
        - cache: (x, w, b)
        """
        out = None
        ###########################################################################
        # TODO: Implement the affine forward pass. Store the result in out. You   #
        # will need to reshape the input into rows.                               #
        ###########################################################################
        ninput = x.shape[0]
        #print("nshape:",ninput)
        q = np.reshape(x,[ninput,-1]) #input goes from (2,4,5,6) to (2,120), weights are (120,3)
        out = np.dot(q,w) + b #output is of (2,3)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        cache = (x, w, b)
        return out, cache


    def affine_backward(self, dout, cache):
        """
        Computes the backward pass for an affine layer.

        Inputs:
        - dout: Upstream derivative, of shape (N, M)
        - cache: Tuple of:
          - x: Input data, of shape (N, d_1, ... d_k)
          - w: Weights, of shape (D, M)
          - b: Biases, of shape (M,)

        Returns a tuple of:
        - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
        - dw: Gradient with respect to w, of shape (D, M)
        - db: Gradient with respect to b, of shape (M,)
        """
        x, w, b = cache
        dx, dw, db = None, None, None
        ###########################################################################
        # TODO: Implement the affine backward pass.                               #
        ###########################################################################
        ninp = x.shape[0]
        q1 = np.reshape(x,[ninp, -1]) #Reshape each input
        wt = w.T
        dx = np.dot(dout, wt) #Calculate dx = dout . w
        dx = np.reshape(dx, x.shape)
        q1t = q1.T
        dw = np.dot(q1t,dout) #Calculate dw = x . dout
        db = np.sum(dout, axis=0) #Calculate db = dout
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return dx, dw, db


    def relu_forward(self, x):
        """
        Computes the forward pass for a layer of rectified linear units (ReLUs).

        Input:
        - x: Inputs, of any shape

        Returns a tuple of:
        - out: Output, of the same shape as x
        - cache: x
        """
        out = None
        ###########################################################################
        # TODO: Implement the ReLU forward pass.                                  #
        ###########################################################################
        out = np.maximum( x, 0)
        
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        cache = x
        return out, cache


    def relu_backward(self, dout, cache):
        """
        Computes the backward pass for a layer of rectified linear units (ReLUs).

        Input:
        - dout: Upstream derivatives, of any shape
        - cache: Input x, of same shape as dout

        Returns:
        - dx: Gradient with respect to x
        """
        dx, x = None, cache
        ###########################################################################
        # TODO: Implement the ReLU backward pass.                                 #
        ###########################################################################
        r_x = x>=0
        dx = dout * r_x 
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return dx
   
    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        # if self.use_dropout:
            # self.dropout_param['mode'] = mode
        # if self.normalization=='batchnorm':
            # for bn_param in self.bn_params:
                # bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        fc_cache = {}
        relu_cache = {}
        inpt = X.shape[0]

        q = np.reshape(X, [inpt, -1])
        
        for i in range(self.num_layers-1):
          fc_out, fc_cache[str(i+1)] = self.affine_forward(q, self.params['W' + str(i+1)], self.params['b'+str(i+1)]) 
          relu_out, relu_cache[str(i+1)] = self.relu_forward(fc_out)    
          q = relu_out    
        
        scores, fc_cache[str(self.num_layers)] = self.affine_forward(q, self.params['W'+str(self.num_layers)], self.params['b'+str(self.num_layers)])        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dx_soft = softmax_loss(scores, y)
        Wsqr = np.sum(np.square(self.params['W'+str(self.num_layers)]))
        loss += 0.5*self.reg *(Wsqr)

        dx_b, dw_b, db_b = self.affine_backward(dx_soft, fc_cache[str(self.num_layers)])
        grads['W'+str(self.num_layers)] = dw_b + self.reg*self.params['W'+str(self.num_layers)]
        grads['b'+str(self.num_layers)] = db_b
        
        for i in range(self.num_layers - 1, 0, -1):
          dx_relu = self.relu_backward(dx_b, relu_cache[str(i)])
          dx_b, dw_b, db_b = self.affine_backward(dx_relu, fc_cache[str(i)])  
          grads['W' + str(i)] = dw_b + self.reg * self.params['W' + str(i)]
          grads['b' + str(i)] = db_b
          loss += 0.5 * self.reg * (np.sum(np.square(self.params['W' + str(i)])))  
  
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads