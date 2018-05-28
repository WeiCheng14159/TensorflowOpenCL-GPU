#include "cifar10Reader.h"

using tensorflow::string;
using namespace std;

// Constructor
cifar10Reader::cifar10Reader(){
  dataset = cifar::read_dataset<vector, vector, uint8_t, uint8_t>();
  trainDataSize = dataset.training_images.size();
  testDataSize = dataset.test_images.size();
  // CIFAR10 dataset consists of square images with dimension 32*32
  imgSize = 32;
}

// Training Batch
// Notice that if beginIdx > `trainDataSize`, beginIdx' = (beginIdx % trainDataSize)
void cifar10Reader::getTrainingBatch(int beginIdx, int batchSize,
  vector<float>* batchImgVec, vector<float>* batchLabelVec)
{
    // Boundary checking
    if( beginIdx + batchSize > trainDataSize){
      beginIdx = beginIdx % trainDataSize;
    }

    for( auto idx = beginIdx; idx < beginIdx + batchSize; idx++ )
    {
        vector<uint8_t> vecImg = dataset.training_images[ idx ];

        for( auto pixel = 0 ; pixel < vecImg.size() ; pixel ++ ){
          batchImgVec->push_back( static_cast<float>( vecImg[pixel]/2 ) );
        }

        uint8_t vecLabel = dataset.training_labels[ idx ];

        float oneHotLabelVec[10] = {0,0,0,0,0,0,0,0,0,0};
        oneHotLabelVec[vecLabel] = 1.0f;

        batchLabelVec->insert(batchLabelVec->end(), oneHotLabelVec, oneHotLabelVec+10);
    }
}

// Testing Batch
void cifar10Reader::getTestingBatch(int beginIdx, int batchSize,
  vector<float>* batchImgVec, vector<float>* batchLabelVec)
{
  // Boundary checking
  if( beginIdx + batchSize > testDataSize){
    throw std::invalid_argument( "Index out of bound " );
  }

  for( auto idx = beginIdx; idx < beginIdx + batchSize; idx++ )
  {
      vector<uint8_t> vecImg = dataset.test_images[ idx ];
      for( auto pixel = 0 ; pixel < vecImg.size() ; pixel ++ ){
        batchImgVec->push_back( static_cast<float>( vecImg[pixel] ) );
      }

      uint8_t vecLabel = dataset.test_labels[ idx ];

      float oneHotLabelVec[10] = {0,0,0,0,0,0,0,0,0,0};
      oneHotLabelVec[vecLabel] = 1.0f;

      batchLabelVec->insert(batchLabelVec->end(), oneHotLabelVec, oneHotLabelVec+10);
  }
}

// getTrainingDataSize
int cifar10Reader::getTrainingDataSize(){
  return trainDataSize;
}

// getTestingDataSize
int cifar10Reader::getTestingDataSize(){
  return testDataSize;
}

// getImgSize
int cifar10Reader::getImgSize(){
  return imgSize;
}
