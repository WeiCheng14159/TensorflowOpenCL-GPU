// sample program can be found on  https://tebesu.github.io/posts/Training-a-TensorFlow-graph-in-C++-API

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/graph/default_device.h"

using namespace tensorflow;
using namespace std;

int main(int argc, char* argv[]) {

  if( argc != 2){
    cerr << "usage [Batch Size]" << endl;
  }

    string graph_definition = "mlp.pb";
    Session* session;
    GraphDef graph_def;
    SessionOptions opts;
    vector<Tensor> outputs; // Store outputs
    TF_CHECK_OK(ReadBinaryProto(Env::Default(), graph_definition, &graph_def));

    // create a new session
    TF_CHECK_OK(NewSession(opts, &session));

    // Load graph into session
    TF_CHECK_OK(session->Create(graph_def));

    // Initialize our variables
    TF_CHECK_OK(session->Run({}, {}, {"init_all_vars_op"}, nullptr));

    // Read in batch size from command line
    int batch_size = atoi( argv[1] );

    // Tensor initializaiotn
    Tensor x(DT_FLOAT, TensorShape({batch_size, 256}));
    Tensor y(DT_FLOAT, TensorShape({batch_size, 10}));

    // Convert TensorFlow Tensor -> Eigen Matrix
    auto _XTensor = x.matrix<float>();
    auto _YTensor = y.matrix<float>();

    // Initialize Matrix size with random number
    _XTensor.setRandom();
    _YTensor.setRandom();

    float cost = 0.0;

    TF_CHECK_OK(session->Run({{"x", x}, {"y", y}}, {"cost"}, {}, &outputs)); // Get cost
    float initial_cost = outputs[0].scalar<float>()(0);

    int iter = 0;
    cout << "initial_cost: " << initial_cost << endl;
    do{
        TF_CHECK_OK(session->Run({{"x", x}, {"y", y}}, {"cost"}, {}, &outputs)); // Get cost
        cost = outputs[0].scalar<float>()(0);
        cout << "Iteration: " << iter << ", Cost: " <<  cost << endl;
        TF_CHECK_OK(session->Run({{"x", x}, {"y", y}}, {}, {"train"}, nullptr)); // Train
        outputs.clear();
        iter++;
    }while( cost >= 0.01 * initial_cost );

    cout << "Final cost: " << cost << endl;

    return 0;
}
