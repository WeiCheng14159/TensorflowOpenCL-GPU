# Training and Testing on Mobile Phone (Snapdragon 835)
# 1. Explanation :

Train a batch of data and test the model with testing dataset.
The MNIST dataset is trained for 2 times.

# 2. Train MNIST dataset on mobile CPU

| Model Name |  Overall Accuracy (%)  |
| :---       | :---                   |
| MLP        | 83.95                  |
| DNN        | 94.46                  |

# 3. Instructions:

Compile project by running script `./train_mnist_android.sh`

For GPU results, run the following commands on Android

1) MLP `./train_and_test_mnist --graphName=mnist_mlp.pb --batchSize=100`
### * Change batchSize=100 for better accuracy
2) DNN `./train_and_test_mnist --graphName=mnist_dnn.pb --batchSize=1000 --dropProb=0.8`
