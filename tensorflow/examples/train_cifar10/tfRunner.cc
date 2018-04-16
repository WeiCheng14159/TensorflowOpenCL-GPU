// Tensorflow trainer

#include "tfRunner.h"

using tensorflow::int32;
using tensorflow::string;
using tensorflow::Status;
using tensorflow::Tensor;

using namespace std;
using namespace tensorflow;

tfRunner::tfRunner(const string& initOps,
  const string& trainOps, const string& costOps, const string& graph)
{
  initOpsName = initOps;
  trainingOpsName = trainOps;
  costOpsName = costOps;
  graphName = graph;
};

// Init Input & Output Tensor
void tfRunner::tensorInit(int batchSize, int inputSize)
{
  inputTensor = Tensor( DT_FLOAT, TensorShape( { batchSize, inputSize } ));
  outputTensor = Tensor( DT_FLOAT, TensorShape( { batchSize, 10       } ));
  dropoutTensor = Tensor( DT_FLOAT, TensorShape( { 1 } ) );
}

// Copy data to Tensors
void tfRunner::copyToTensor( const vector<float>& inputVector,
    const vector<float>& outputVector, const vector<float>& dropoutProb )
{
  copy_n(inputVector.begin(), inputVector.size(),
    inputTensor.flat<float>().data() );
  copy_n(outputVector.begin(), outputVector.size(),
    outputTensor.flat<float>().data() );
  copy_n(dropoutProb.begin(), dropoutProb.size(),
    dropoutTensor.flat<float>().data() );
}

// Init session
void tfRunner::sessInit(unique_ptr<Session>& sess)
{
  TF_CHECK_OK( sess->Run( {}, {}, {initOpsName}, nullptr ) );
}
// Session Training
void tfRunner::sessionTrain(unique_ptr<Session>& sess, const string& inputOpsName,
    const string& outputOpsName, const string& dropoutOpsName )
{
  if ( graphName == "cifar10_mlp.pb" ){
    TF_CHECK_OK( sess->Run( { {inputOpsName, inputTensor},
      {outputOpsName, outputTensor} }, {}, {trainingOpsName}, nullptr) );
  }else if( graphName == "cifar10_cnn.pb" ){
    TF_CHECK_OK( sess->Run( { {inputOpsName, inputTensor},
      {outputOpsName, outputTensor}, {dropoutOpsName, dropoutTensor} }, {},
      {trainingOpsName}, nullptr) );
  }else if( graphName == "cifar10_dnn.pb" ){
    TF_CHECK_OK( sess->Run( { {inputOpsName, inputTensor},
      {outputOpsName, outputTensor}, {dropoutOpsName, dropoutTensor} }, {},
      {trainingOpsName}, nullptr) );
  }else{
    LOG(ERROR) << graphName << " Not Supported";
  }
}
// Session Testing
double tfRunner::sessionTest(unique_ptr<Session>& sess, const string& inputOpsName,
    const string& outputOpsName, const string& dropoutOpsName )
{

  double accu = 0.0f;

  // Results
  vector<Tensor> outputs;

  if ( graphName == "cifar10_mlp.pb" ){
    TF_CHECK_OK( sess->Run( { {inputOpsName, inputTensor},
      {outputOpsName, outputTensor} }, {costOpsName}, {}, &outputs) );
  }else if( graphName == "cifar10_cnn.pb" ){
    TF_CHECK_OK( sess->Run( { {inputOpsName, inputTensor},
      {outputOpsName, outputTensor}, {dropoutOpsName, dropoutTensor} }, {costOpsName},
      {}, &outputs) );
  }else if( graphName == "cifar10_dnn.pb" ){
    TF_CHECK_OK( sess->Run( { {inputOpsName, inputTensor},
      {outputOpsName, outputTensor}, {dropoutOpsName, dropoutTensor} }, {costOpsName},
      {}, &outputs) );
  }else{
    LOG(ERROR) << graphName << " Not Supported";
  }

  return double ( outputs[0].scalar<float>()(0) );
}
