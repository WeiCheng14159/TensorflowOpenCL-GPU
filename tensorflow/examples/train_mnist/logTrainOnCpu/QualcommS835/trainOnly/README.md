# Training on Mobile Phone (Snapdragon 835)
# 1. Explanation :

Train TF model on mobile CPU.

# 2. Train MNIST dataset on mobile CPU

| Model Name |  Overall Accuracy (%)  | Training Time (s) | Batch Size |
| :---       | :---                   | :---              | :---       |
| MLP        | 79.1111                | 10.9938           | 100        |
| DNN        | 94.8323                | 207.205           | 1000       |

# 3. Instructions:

Compile project by running script `./train_mnist_android.sh`

Run the following commands on Android:

1) MLP `./train_mnist --graphName=mnist_mlp.pb --batchSize=100`
### * Change batchSize=100 for better accuracy
2) DNN `./train_mnist --graphName=mnist_dnn.pb --dropProb=0.8 --batchSize=1000`
