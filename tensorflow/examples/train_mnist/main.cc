
/*
By Cheng Wei on 2018/Jan/24
==============================================================================*/
// A simple program trainging a MNIST TF model using TF C++ API

#include <vector>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

#include "tfRunner.h"
#include "util.h"
#include "mnistReader.h"

// These are all common classes it's handy to reference with no namespace.
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;
using namespace tensorflow;
using namespace std;

int main(int argc, char* argv[]) {

  string root_dir         = "/data/local/tmp/";
  string graphName        = "mnist_mlp.pb";
  string mnistDir         = root_dir + "MNIST_data/";
  string inputOpsName     = "input";
  string outputOpsName    = "output";
  string accuOpsName      = "test";
  string trainOpsName     = "adam_optimizer/train";
  string dropoutOpsName   = "Dropout/Placeholder";
  int32  input_width      = 28;
  int32  input_height     = 28;
  int32  batchSize        = 50;
  int32  maxSteps         = 100000;
  vector<float> dropProb  = { 0.5 } ;

  vector<Flag> flag_list = {
      Flag("root_dir",      &root_dir,      "Binary Root Directory"),
      Flag("graphName",     &graphName,     "Graph To Be Executed"),
      Flag("mnistDir",      &mnistDir,      "MNIST Dataset Directory"),
      Flag("inputOpsName",  &inputOpsName,  "Input Ops Name"),
      Flag("outputOpsName", &outputOpsName, "Output Ops Name"),
      Flag("accuOpsName",   &accuOpsName,   "Cost Ops Name"),
      Flag("trainOpsName",  &trainOpsName,  "Train Ops Name"),
      Flag("dropoutOpsName",&dropoutOpsName,"Dropout Ops Name"),
      Flag("batchSize",     &batchSize,     "Training & Testing Batch Size"),
      Flag("maxSteps",      &maxSteps,      "Maximum Number of Taining Steps"),
      Flag("dropProb",      &dropProb[0],   "Drop-out Layer (if any) Probability"),
  };

  string usage = Flags::Usage(argv[0], flag_list);
  const bool parse_result = Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    LOG(ERROR) << usage;
    return -1;
  }

  // We need to call this to set up global state for TensorFlow.
  port::InitMain(argv[0], &argc, &argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return -1;
  }

  LOG(INFO) << "[Root directory] = " << root_dir ;

  // Prepare MNIST dataset
  LOG(INFO) << "[MNIST Dataset Directory] = " << mnistDir ;

  mnistReader mnist = mnistReader(mnistDir);

  LOG(INFO) << "[MNIST Dataset] Num of Training Images = " << mnist.getTrainingDataSize();
  LOG(INFO) << "[MNIST Dataset] Num of Training Labels = " << mnist.getTrainingDataSize();
  LOG(INFO) << "[MNIST Dataset] Num of Test     Images = " << mnist.getTestingDataSize();
  LOG(INFO) << "[MNIST Dataset] Num of Test     Labels = " << mnist.getTestingDataSize();
  LOG(INFO) << "[MNIST Dataset] Input Image Size       = " << mnist.getImgSize();

  input_width  = mnist.getImgSize();
  input_height = mnist.getImgSize();

  // Load TF model.
  unique_ptr<Session> session;
  string graph_path = io::JoinPath(root_dir, graphName);
  Status load_graph_status = LoadGraph(graph_path, &session);
  if (!load_graph_status.ok()) {
    LOG(ERROR) << load_graph_status;
    return -1;
  }
  LOG(INFO) << "[TF Model File Loaded From Directory] = " << graph_path ;

  tfRunner runner = tfRunner("init", trainOpsName, accuOpsName, graphName);

  runner.sessInit(session);

  runner.tensorInit(batchSize, input_width*input_height);

  for( auto beginIdx = 0 ; beginIdx < mnist.getTrainingDataSize() - batchSize;
    beginIdx = beginIdx + batchSize )
  {

    LOG(INFO) << beginIdx << " trained.";

    // If the number of training steps > maxSteps then stop training
    if ( beginIdx > maxSteps ){ break; }

    // image vector with dimension { 1, batchSize x input_width x input_height }
    vector<float> batchTrainImgFloatVec;
    // label vector with dimension { 1, batchSize }
    vector<long int> batchTrainLabelInt64Vec;

    mnist.getTrainingBatch(beginIdx, batchSize, &batchTrainImgFloatVec, &batchTrainLabelInt64Vec);

    runner.copyToTensor(batchTrainImgFloatVec, batchTrainLabelInt64Vec, dropProb);

    runner.sessionTrain(session, inputOpsName, outputOpsName, dropoutOpsName);

  } // End of Training Batch Loop

  vector<float> avg_accu;

  for( auto beginIdx = 0 ; beginIdx < mnist.getTestingDataSize() - batchSize;
    beginIdx = beginIdx + batchSize )
  { // Testing Batch Loop

    LOG(INFO) << beginIdx << " tested.";

    // image vector with dimension { 1, batchSize x input_width x input_height }
    vector<float> batchTestImgFloatVec;
    // label vector with dimension { 1, batchSize }
    vector<long int> batchTestLabelInt64Vec;

    mnist.getTestingBatch(beginIdx, batchSize, &batchTestImgFloatVec, &batchTestLabelInt64Vec);

    // No drop out layer when testing
    dropProb[0] = 1.0f;

    runner.copyToTensor(batchTestImgFloatVec, batchTestLabelInt64Vec, dropProb);

    double acc = runner.sessionTest(session, inputOpsName, outputOpsName, dropoutOpsName);

    avg_accu.push_back( acc );
    LOG(INFO) << "Accuracy " << acc * 100 << "\%";

  } // End of Testing Batch Loop

  LOG(INFO) << "Overall testing accuracy " << 100 * accumulate(
    avg_accu.begin(), avg_accu.end(), 0.0f) / avg_accu.size() << "\%";

} // End of main
