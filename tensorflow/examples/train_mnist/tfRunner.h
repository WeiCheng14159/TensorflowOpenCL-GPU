// Tensorflow Runner
#include <vector>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/standard_ops.h"
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

using namespace tensorflow;
using namespace std;
using tensorflow::int32;
using tensorflow::string;
using tensorflow::Status;
using tensorflow::Tensor;

class tfRunner{
public:

  tfRunner(const string& initOps,
    const string& trainOps, const string& costOps, const string& graph);

  // Init Input & Output Tensor
  void tensorInit(int batchSize, int inputSize);

  // Copy data to Tensors
  void copyToTensor( const vector<float>& inputVector,
    const vector<float>& outputVector, const vector<float>& dropoutProb );

  // Init session
  void sessInit(unique_ptr<Session>& sess);

  // Session Training
  void sessionTrain(unique_ptr<Session>& sess, const string& inputOpsName,
    const string& outputOpsName, const string& dropoutOpsName );

  // Session Testing
  double sessionTest(unique_ptr<Session>& sess, const string& inputOpsName,
      const string& outputOpsName, const string& dropoutOpsName );

private:
  // Used Tensor
  Tensor inputTensor;
  Tensor outputTensor;
  Tensor dropoutTensor;

  // Tensorflow graph name
  string graphName;

  // Tensorflow init ops
  string initOpsName;

  // Tensorflow training ops
  string trainingOpsName;

  // Tensorflow testing ops
  string costOpsName;

};
