
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
#define ROOT_DIR "/data/local/tmp"

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

int main() {

  string root_dir = ROOT_DIR;
  std::cout << "Root directory: " << ROOT_DIR << std::endl;
  // Prepare MNIST dataset
    string mnist_dir = root_dir + "/MNIST_data/";
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
    std::cout << "[MNIST Dataset] Input Image Size       = " <<
      std::sqrt( dataset.training_images[0].size() )         << "x" <<
      std::sqrt( dataset.training_images[0].size() )         << std::endl;

  // Load and initialize the model.
  string graph_dir = "/mnist_100_mlp.pb";
  std::unique_ptr<tensorflow::Session> session;
  string graph_path = tensorflow::io::JoinPath(root_dir, graph_dir);
  Status load_graph_status = LoadGraph(graph_path, &session);
  if (!load_graph_status.ok()) {
    LOG(ERROR) << load_graph_status;
    return -1;
  }
  std::cout << "[TF Model File Loaded Dir] = " << graph_dir << std::endl;

  // Load MNIST training data into TF Tensors
  std::vector<Tensor> img_tensors;

  // for( auto i = 0 ; i < training_images.size() )


  // Actually run the image through the model.
  // std::vector<Tensor> outputs;
  // Status run_status = session->Run({{input_layer, resized_tensor}},
  //                                  {output_layer}, {}, &outputs);
  // if (!run_status.ok()) {
  //   LOG(ERROR) << "Running model failed: " << run_status;
  //   return -1;
  // }

}
