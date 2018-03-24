#include "mnistReader.h"

using tensorflow::string;
using namespace std;

// Constructor
mnistReader::mnistReader(string path){
  dataset = mnist::read_dataset<vector, vector, uint8_t, uint8_t>(path);
  trainDataSize = dataset.training_images.size();
  testDataSize = dataset.test_images.size();
  // MNIST handwritten dataset consists of square images with dimension 28*28
  imgSize = sqrt( dataset.training_images[0].size() );
}

// Training Batch
void mnistReader::getTrainingBatch(int beginIdx, int batchSize,
  vector<float>* batchImgVec, vector<long int>* batchLabelVec)
{
    // Boundary checking
    if( beginIdx + batchSize > trainDataSize){
      throw std::invalid_argument( "Index out of bound " );
    }

    for( auto idx = beginIdx; idx < beginIdx + batchSize; idx++ )
    {
        vector<uint8_t> vecImg = dataset.training_images[ idx ];
        for( auto pixel = 0 ; pixel < vecImg.size() ; pixel ++ ){
          batchImgVec->push_back( static_cast<float>( vecImg[pixel] ) );
        }

        uint8_t vecLabel = dataset.training_labels[ idx ];
        batchLabelVec->push_back( static_cast<long int>(vecLabel) );
    }
}

// Testing Batch
void mnistReader::getTestingBatch(int beginIdx, int batchSize,
  vector<float>* batchImgVec, vector<long int>* batchLabelVec)
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
      batchLabelVec->push_back( static_cast<long int>(vecLabel) );
  }
}

// getTrainingDataSize
int mnistReader::getTrainingDataSize(){
  return trainDataSize;
}

// getTestingDataSize
int mnistReader::getTestingDataSize(){
  return testDataSize;
}

// getImgSize
int mnistReader::getImgSize(){
  return imgSize;
}