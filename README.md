# CS231n-library-fnn

### An implementation of Assignment-2 Stanford Course CS231n on Convolutional Neural Networks

>fully_connected_net.py :
 A file compiled for overview of the implementation.
 
Has a class FullyConnectedNet that takes and argument 

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
        
This is to understand the working of a multi-layered perceptron.

The same has been implemented within the **cs231n** library 
* Just download and place the cs23n folder within your local directory.
* Example call to create a MLP:
  
```
import cs231n.classifiers.fc_net as fc

learning_rate = 3.113669e-04 
weight_scale = 2.461858e-02
model = fc.FullyConnectedNet([100, 100, 100, 100],
                weight_scale=weight_scale, dtype=np.float64)
solver = Solver(model, data,
                print_every=100, num_epochs=10, batch_size=250,
                update_rule='adam',
                optim_config={
                  'learning_rate': learning_rate,
                }
         )
solver.train()

best_model = model

```

## Implementation 

* Libraries used:
```
import numpy as np
from cs231n.layers import *
```

What we want to develop looks like this:

<pre>

{affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax}
where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.
</pre>

* Weights are initialized from a normal distribution centered at 0 with standard deviation equal to weight_scale. Biases should be initialized to zero.                                                                         

> loss(self, X, y=None):

* **Uses the following functions to create a feed forward network:**

> def affine_forward(self, x, w, b):

* Computes the forward pass for an affine (fully-connected) layer.
* The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
        examples, where each example x[i] has shape (d_1, ..., d_k). We will
        reshape each input into a vector of dimension D = d_1 * ... * d_k, and
        then transform it to an output vector of dimension M.
* Inputs:

        - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
        
        - w: A numpy array of weights, of shape (D, M)
        
        - b: A numpy array of biases, of shape (M,)
        
* Returns a tuple of:

        - out: output, of shape (N, M)
    
        - cache: (x, w, b)

> def relu_forward(self, x):

* Computes the forward pass for a layer of rectified linear units (ReLUs).

* Input:

        - x: Inputs, of any shape
        
* Returns a tuple of:

        - out: Output, of the same shape as x
        
        - cache: x

* **Compute loss and gradient for the fully-connected net using the following:**

> softmax_loss(scores, y)

* to compute loss of the network.


> def affine_backward(self, dout, cache):

* Computes the backward pass for an affine layer.

* Inputs:

        - dout: Upstream derivative, of shape (N, M)
        
        - cache: Tuple of:
        
          - x: Input data, of shape (N, d_1, ... d_k)
          
          - w: Weights, of shape (D, M)
          
          - b: Biases, of shape (M,)
          
*  Returns a tuple of:

        - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
        
        - dw: Gradient with respect to w, of shape (D, M)
        
        - db: Gradient with respect to b, of shape (M,

> def relu_forward(self, x):

* Computes the forward pass for a layer of rectified linear units (ReLUs).

* Input:

        - x: Inputs, of any shape
        
* Returns a tuple of:

        - out: Output, of the same shape as x
        
        - cache: x
        
        
        
