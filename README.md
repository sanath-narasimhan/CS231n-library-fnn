# CS231n-library-fnn

### An implementation of Assignment-2 Stanford Course CS231n on Convolutional Neural Networks

>fully_connected_net.py

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


