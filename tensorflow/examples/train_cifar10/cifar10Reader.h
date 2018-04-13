// Tensorflow trainer

#include <vector>
#include <stdexcept>

#include "tensorflow/core/lib/core/stringpiece.h"

// MNIST reader helper
#include "cifar/cifar10_reader.hpp"

using tensorflow::string;
using namespace std;

class cifar10Reader{

public:
  // Constructor
  cifar10Reader();

  // Training Batch
  void getTrainingBatch(int beginIdx, int batchSize,
    vector<float>* batchImgVec, vector<float>* batchLabelVec);

  // Testing Batch
  void getTestingBatch(int beginIdx, int batchSize,
    vector<float>* batchImgVec, vector<float>* batchLabelVec);

  // getTrainingDataSize
  int getTrainingDataSize();

  // getTestingDataSize
  int getTestingDataSize();

  // getImgSize
  int getImgSize();
private:
  cifar::CIFAR10_dataset<vector, vector<uint8_t>, uint8_t> dataset;
  int trainDataSize;
  int testDataSize;
  int imgSize;
};
