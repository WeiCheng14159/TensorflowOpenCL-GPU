
/*
By Cheng Wei on 2018/Jan/24
==============================================================================*/
// A simple program trainging a MNIST TF model using TF C++ API

#include <fstream>
#include <utility>
#include <vector>
#include <iostream>
#include <math.h>

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

// MNIST reader helper
#include "mnist/mnist_reader.hpp"

// These are all common classes it's handy to reference with no namespace.
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;
using namespace tensorflow;

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status LoadGraph(const string& graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session) {
  tensorflow::GraphDef graph_def;
  Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'");
  }
  session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return Status::OK();
}

int main(int argc, char* argv[]) {

  string root_dir         = "/data/local/tmp/";
  string graph_name       = "mnist_100_mlp.pb";
  string mnist_dir        = root_dir + "MNIST_data/";
  string input_layer      = "input";
  string output_layer     = "output";
  int32  input_width      = 28;
  int32  input_height     = 28;
  int32  batch_size       = 10;

  std::vector<Flag> flag_list = {
      Flag("root_dir",      &root_dir,
        "interpret graph file names relative to this directory"),
      Flag("graph_name",    &graph_name,    "graph to be executed"),
      Flag("mnist_dir",     &mnist_dir,     "MNIST dataset directory"),
      Flag("input_layer",   &input_layer,   "name of input layer"),
      Flag("output_layer",  &output_layer,  "name of output layer"),
      Flag("batch_size",    &batch_size,    "training batch size"),
  };

  string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    LOG(ERROR) << usage;
    return -1;
  }

  // We need to call this to set up global state for TensorFlow.
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return -1;
  }

  std::cout << "[Root directory] = " << root_dir << std::endl;

  // Prepare MNIST dataset
    std::cout << "[MNIST Dataset Directory] = " << mnist_dir << std::endl;

    // Load MNIST data
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
        mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(mnist_dir);

    std::cout << "[MNIST Dataset] Num of Training Images = " <<
      dataset.training_images.size() << std::endl;
    std::cout << "[MNIST Dataset] Num of Training Labels = " <<
      dataset.training_labels.size() << std::endl;
    std::cout << "[MNIST Dataset] Num of Test     Images = " <<
      dataset.test_images.size() << std::endl;
    std::cout << "[MNIST Dataset] Num of Test     Labels = " <<
      dataset.test_labels.size() << std::endl;

    input_width  = std::sqrt( dataset.training_images[0].size() );
    input_height = std::sqrt( dataset.training_images[0].size() );

    std::cout << "[MNIST Dataset] Input Image Size       = " <<
      input_width << "x" << input_height << std::endl;

  // Load and initialize TF model.
  std::unique_ptr<tensorflow::Session> session;
  string graph_path = tensorflow::io::JoinPath(root_dir, graph_name);
  Status load_graph_status = LoadGraph(graph_path, &session);
  if (!load_graph_status.ok()) {
    LOG(ERROR) << load_graph_status;
    return -1;
  }
  std::cout << "[TF Model File Loaded From Directory] = " << graph_path << std::endl;

  // Load MNIST training data into TF Tensors
    // batch_tensor with dimenstion { batch_size, input_width * input_height }
    // = { batch_size, 28*28 }
    tensorflow::Tensor batch_tensor(tensorflow::DT_FLOAT,
      tensorflow::TensorShape({batch_size, input_width * input_height}));

    for( auto batch_idx = 0 ; batch_idx < dataset.training_images.size() / batch_size;
      batch_idx ++ ){ // Batch loop

      // batch vector with dimension { 1, batch_size x input_width x input_height }
      std::vector<float> batch_vec;

      for ( auto batch_itr = 0 ; batch_itr < batch_size ; batch_itr ++ ){ // Within a Batch

        // vec with dim { 1, input_width * input_height }
        std::vector<uint8_t> vec = dataset.training_images[ batch_idx * batch_size + batch_itr ];
        for( auto pixel = 0 ; pixel < vec.size() ; pixel ++ ){
          // Cast image data from uint_8 -> float
          batch_vec.push_back( (float)( vec[pixel] ) );
        }
      } // Within a Batch

      // Copy a single batch of data to TF Tensor
      std::copy_n( batch_vec.begin(), batch_vec.size(), batch_tensor.flat<float>().data() );

      // Check if Tensor is initialized
      if( !batch_tensor.IsInitialized() ){
        LOG(ERROR) << "Tensor not initialized" ;
        return -1;
      }

      // Input Tensor info
      std::cout << "Batch " << batch_idx << " loaded " << batch_tensor.DebugString()
        << std::endl;

  } // End of Batch loop

} // End of main
