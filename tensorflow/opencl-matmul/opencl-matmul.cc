#include "tensorflow/core/public/session.h"
#include "tensorflow/core/graph/default_device.h"

using namespace tensorflow;
using namespace std;

int main(int argc, char* argv[]) {

    std::string graph_definition = "matmul.pb";
    Session* session;
    GraphDef graph_def;
    SessionOptions opts;
    std::vector<Tensor> outputs; // Store outputs
    TF_CHECK_OK(ReadBinaryProto(Env::Default(), graph_definition, &graph_def));

    // Set graph options
    graph::SetDefaultDevice("/cpu:0", &graph_def);

    // create a new session
    TF_CHECK_OK(NewSession(opts, &session));

    // Load graph into session
    TF_CHECK_OK(session->Create(graph_def));

    // X matrix has shape 1 * mat_size
    // X matrix has shape mat_size * 1
    int mat_size = 10000;
    Tensor Tx(DT_FLOAT, TensorShape({1 ,mat_size}));
    Tensor Ty(DT_FLOAT, TensorShape({mat_size ,1}));

    // x_ & y_ are Eigen::Tensor
    auto x_ = Tx.matrix<float>().setRandom();
    auto y_ = Ty.matrix<float>().setRandom();

    // Compute matrix multiplaction result
    std::vector<Tensor> output;
    TF_CHECK_OK(session->Run({{"x", Tx}, {"y", Ty}}, {"matmul"}, {}, &outputs)); // Get cost
    cout << "GPU result: " << outputs[0].scalar<float>()(0) << endl;

    // Check on CPU
    auto mat = x_ * y_ ;
    cout << "CPU result: " << mat.sum() << endl ;

    session->Close();
    delete session;
    return 0;
}
