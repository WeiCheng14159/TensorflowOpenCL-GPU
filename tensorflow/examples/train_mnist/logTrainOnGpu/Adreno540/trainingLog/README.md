# Training and Testing on Mobile Phone (Snapdragon 835)
## 1. Explanation :

Train a batch of data and test the model with testing dataset.
The MNIST dataset is trained for 2 times.

## 2. Train MNIST dataset on mobile GPU

| Model Name |  Overall Accuracy (%)  |
| :---       | :---                   |
| MLP        | 84.25                  |
| DNN        | 95.37                  |

### MLP training accuracy:

![MLP training results](https://github.com/supernovaremnant/TensorflowOpenCL-GPU/blob/feat-trainMNIST-full-speed/tensorflow/examples/train_mnist/logTrainOnGpu/Adreno540/trainingLog/mlp/Screen%20Shot%202018-03-27%20at%2010.33.57%20AM.png)

### DNN training accuracy:

![DNN training results](https://github.com/supernovaremnant/TensorflowOpenCL-GPU/blob/feat-trainMNIST-full-speed/tensorflow/examples/train_mnist/logTrainOnGpu/Adreno540/trainingLog/dnn/Screen%20Shot%202018-03-27%20at%207.33.52%20AM.png)

### MLP model structure:

![MLP training results](https://github.com/supernovaremnant/TensorflowOpenCL-GPU/blob/feat-trainMNIST-full-speed/tensorflow/examples/train_mnist/logTrainOnGpu/Adreno540/trainingLog/mlp/graph_large_attrs_key%3D_too_large_attrs%26limit_attr_size%3D1024%26run%3D.png)

### DNN model structure:

![DNN training results](https://github.com/supernovaremnant/TensorflowOpenCL-GPU/blob/feat-trainMNIST-full-speed/tensorflow/examples/train_mnist/logTrainOnGpu/Adreno540/trainingLog/dnn/graph_large_attrs_key%3D_too_large_attrs%26limit_attr_size%3D1024%26run%3D.png)

## 3. Instructions:

Compile project by running script `./train_mnist_android.sh`

For GPU results, run the following commands on Android

### 1) MLP
`./train_and_test_mnist --graphName=mnist_mlp.pb --batchSize=100`
#### * Change batchSize=100 for better accuracy
### 2) DNN
`./train_and_test_mnist --graphName=mnist_dnn.pb --batchSize=1000 --dropProb=0.8`
