
/*
By Cheng Wei on 2018/Jan/24
==============================================================================*/
// A simple program trainging a CIFAR10 TF model using TF C++ API

#include <vector>
#include <chrono>

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
#include "cifar10Reader.h"
#include "tensorboard_logger.h"

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
  string graphName        = "cifar10_mlp.pb";
  string cifarDir         = root_dir + "cifar-10/";
  string inputOpsName     = "input";
  string outputOpsName    = "output";
  string accuOpsName      = "test";
  string trainOpsName     = "adam_optimizer/train";
  string dropoutOpsName   = "Dropout/Placeholder";
  int32  input_width      = 32;
  int32  input_height     = 32;
  int32  batchSize        = 50;
  int32  maxSteps         = 1000000;
  float  iteration        = 1.0f;
  vector<float> dropProb  = { 0.5 } ;

  long int timeStamp = std::chrono::duration_cast<std::chrono::milliseconds>
  ( std::chrono::system_clock::now().time_since_epoch() ).count();
  string logFileName      = root_dir + "events.out.tfevents." + to_string(timeStamp)
     + ".wei.local";

  vector<Flag> flag_list = {
      Flag("root_dir",      &root_dir,      "Binary Root Directory"),
      Flag("graphName",     &graphName,     "Graph To Be Executed"),
      Flag("cifarDir",      &cifarDir,      "CIFAR10 Dataset Directory"),
      Flag("inputOpsName",  &inputOpsName,  "Input Ops Name"),
      Flag("outputOpsName", &outputOpsName, "Output Ops Name"),
      Flag("accuOpsName",   &accuOpsName,   "Cost Ops Name"),
      Flag("trainOpsName",  &trainOpsName,  "Train Ops Name"),
      Flag("dropoutOpsName",&dropoutOpsName,"Dropout Ops Name"),
      Flag("batchSize",     &batchSize,     "Training & Testing Batch Size"),
      Flag("maxSteps",      &maxSteps,      "Maximum Number of Taining Steps"),
      Flag("iteration",     &iteration,     "Number of Iteration to Traing the Whole Dataset"),
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

  // Prepare CIFAR10 dataset
  LOG(INFO) << "[CIFAR10 Dataset Directory] = " << cifarDir ;

  cifar10Reader cifar = cifar10Reader();

  LOG(INFO) << "[CIFAR10 Dataset] Num of Training Images = " << cifar.getTrainingDataSize();
  LOG(INFO) << "[CIFAR10 Dataset] Num of Training Labels = " << cifar.getTrainingDataSize();
  LOG(INFO) << "[CIFAR10 Dataset] Num of Test     Images = " << cifar.getTestingDataSize();
  LOG(INFO) << "[CIFAR10 Dataset] Num of Test     Labels = " << cifar.getTestingDataSize();
  LOG(INFO) << "[CIFAR10 Dataset] Input Image Size       = " << cifar.getImgSize();

  input_width  = cifar.getImgSize();
  input_height = cifar.getImgSize();

  TensorBoardLogger logger( logFileName.c_str() );

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

  runner.tensorInit(batchSize, input_width*input_height*3);

  for( auto beginIdx = 0 ; beginIdx < cifar.getTrainingDataSize()*iteration - batchSize;
    beginIdx = beginIdx + batchSize )
  {

    LOG(INFO) << beginIdx << " trained.";

    // If the number of training steps > maxSteps then stop training
    if ( beginIdx > maxSteps ){ break; }

    // image vector with dimension { 1, batchSize x input_width x input_height }
    vector<float> batchTrainImgFloatVec;
    // label vector with dimension { 1, batchSize }
    vector<float> batchTrainLabelFloatVec;

    cifar.getTrainingBatch(beginIdx, batchSize, &batchTrainImgFloatVec, &batchTrainLabelFloatVec);

    runner.copyToTensor(batchTrainImgFloatVec, batchTrainLabelFloatVec, dropProb);

    runner.sessionTrain(session, inputOpsName, outputOpsName, dropoutOpsName);

    // Do overall testing for each 1000 data trained
    if( beginIdx % (10*batchSize) == 0 )
    {
      vector<double> avg_accu;

      for( auto beginIdx = 0 ; beginIdx < cifar.getTestingDataSize() - batchSize;
        beginIdx = beginIdx + batchSize )
      { // Testing Batch Loop

        // LOG(INFO) << beginIdx << " tested.";

        // image vector with dimension { 1, batchSize x input_width x input_height }
        vector<float> batchTestImgFloatVec;
        // label vector with dimension { 1, batchSize }
        vector<float> batchTestLabelFloatVec;

        cifar.getTestingBatch(beginIdx, batchSize, &batchTestImgFloatVec, &batchTestLabelFloatVec);

        // No drop out layer when testing
        dropProb[0] = 1.0f;

        runner.copyToTensor(batchTestImgFloatVec, batchTestLabelFloatVec, dropProb);

        double acc = runner.sessionTest(session, inputOpsName, outputOpsName, dropoutOpsName);

        avg_accu.push_back( acc );
        // LOG(INFO) << "Accuracy " << acc * 100 << "\%";

      } // End of Testing Batch Loop

      auto acc = 100 * accumulate( avg_accu.begin(), avg_accu.end(), 0.0f) / avg_accu.size();

      LOG(INFO) << "Overall testing accuracy " << acc << "\%";

      logger.add_scalar("accurarcy", beginIdx, acc);
    }

  } // End of Training Batch Loop

} // End of main
