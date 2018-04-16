# Training and Testing on Mobile Phone (Snapdragon 835)
## 1. Explanation :

Train a batch of data and test the model with testing dataset.

## 2. Train MNIST dataset on mobile GPU

| Model Name |  Overall Accuracy (%)  |
| :---       | :---                   |
| MLP        | 85.01                  |
| DNN        | 95.72                  |

### MLP training accuracy:

![MLP training results](/tensorflow/examples/train_mnist/logTrainOnGpuFp16/Adreno540/trainingLog/mlp/Screen_Shot_2018-04-16_at_6.44.00_PM.png)

### DNN training accuracy:

![DNN training results](/tensorflow/examples/train_mnist/logTrainOnGpuFp16/Adreno540/trainingLog/dnn/Screen_Shot_2018-04-16_at_7.29.11_PM.png)

## 3. Instructions:

Compile project by running script `./train_mnist_android.sh`

For GPU results, run the following commands on Android

### 1) MLP
`./train_and_test_mnist --graphName=mnist_mlp.pb --batchSize=100 --iteration=3`
#### * batchSize=100 for better accuracy
### 2) DNN
`./train_and_test_mnist --graphName=mnist_dnn.pb --batchSize=1000 --dropProb=0.8 --iteration=3`
