# Train on Mobile Phone (Snapdragon 835)
## 1. Explanation :

Train a batch of data and test the model with testing dataset.
The MNIST dataset is trained for 2 times.

The training process is accelerated by off-loading the MatMul operation from CPU
to GPU in TensorFlow framework.

## 2. Train MNIST dataset on mobile CPU

| Model Name |  Overall Accuracy (%)  | Training Time (s) | Batch Size |
| :---       | :---                   | :---              | :---       |
| MLP        | 79.1111                | 10.9938           | 100        |
| DNN        | 94.8323                | 207.205           | 1000       |

## 3. Train MNIST dataset on mobile GPU

| Model Name |  Overall Accuracy (%)  | Training Time (s) | Batch Size |
| :---       | :---                   | :---              | :---       |
| MLP        | 79.3232                | 57.1262           | 100        |
| DNN        | 96.7778                | 509.021           | 1000       |

## 4. Instructions:

Compile project by running script `./train_mnist_android.sh`

Run the following commands on Android:

### 1) MLP
`./train_mnist --graphName=mnist_mlp.pb --batchSize=100`
#### * Change batchSize=100 for better accuracy
### 2) DNN
`./train_mnist --graphName=mnist_dnn.pb --dropProb=0.8 --batchSize=1000`
