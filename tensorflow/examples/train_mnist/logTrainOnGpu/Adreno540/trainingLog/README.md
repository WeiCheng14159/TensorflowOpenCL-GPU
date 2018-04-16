# Training and Testing on Mobile Phone (Snapdragon 835)
## 1. Explanation :

Train a batch of data and test the model with testing dataset.

## 2. Train MNIST dataset on mobile GPU

| Model Name |  Overall Accuracy (%)  |
| :---       | :---                   |
| MLP        | 79.46                  |
| DNN        | 92.38                  |

### MLP training accuracy:

![MLP training results](/tensorflow/examples/train_mnist/logTrainOnGpu/Adreno540/trainingLog/mlp/Screen_Shot_2018-03-29_at_4.14.59_AM.png)

### DNN training accuracy:

![DNN training results](/tensorflow/examples/train_mnist/logTrainOnGpu/Adreno540/trainingLog/dnn/Screen_Shot_2018-03-29_at_4.38.32_AM.png)

### MLP model structure:

![MLP training results](/tensorflow/examples/train_mnist/logTrainOnGpu/Adreno540/trainingLog/mlp/graph_large_attrs_key%3D_too_large_attrs%26limit_attr_size%3D1024%26run%3D.png)

### DNN model structure:

![DNN training results](/tensorflow/examples/train_mnist/logTrainOnGpu/Adreno540/trainingLog/dnn/graph_large_attrs_key%3D_too_large_attrs%26limit_attr_size%3D1024%26run%3D.png)

## 3. Instructions:

Compile project by running script `./train_mnist_android.sh`

For GPU results, run the following commands on Android

### 1) MLP
`./train_and_test_mnist --graphName=mnist_mlp.pb --batchSize=100 --iteration=3`
#### * batchSize=100 for better accuracy
### 2) DNN
`./train_and_test_mnist --graphName=mnist_dnn.pb --batchSize=1000 --dropProb=0.8 --iteration=3`
