// Tensorflow trainer

#include <vector>
#include <stdexcept>

#include "tensorflow/core/lib/core/stringpiece.h"

// MNIST reader helper
#include "mnist/mnist_reader.hpp"

using tensorflow::string;
using namespace std;

class mnistReader{

public:
  // Constructor
  mnistReader(string path);

  // Training Batch
  void getTrainingBatch(int beginIdx, int batchSize,
    vector<float>* batchImgVec, vector<long int>* batchLabelVec);

  // Testing Batch
  void getTestingBatch(int beginIdx, int batchSize,
    vector<float>* batchImgVec, vector<long int>* batchLabelVec);

  // getTrainingDataSize
  int getTrainingDataSize();

  // getTestingDataSize
  int getTestingDataSize();

  // getImgSize
  int getImgSize();
private:
  mnist::MNIST_dataset<vector, vector<uint8_t>, uint8_t> dataset;
  int trainDataSize;
  int testDataSize;
  int imgSize;
};
