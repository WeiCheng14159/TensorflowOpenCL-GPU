#include <string>
#include <sstream>

#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/public/session.h"

using tensorflow::string;
using namespace tensorflow;

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status LoadGraph(const string& graph_file_name,
                 unique_ptr<Session>* session) {
  GraphDef graph_def;
  Status load_graph_status =
      ReadBinaryProto(Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'");
  }
  session->reset(NewSession(SessionOptions()));
  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return Status::OK();
}

template <typename T>
std::string to_string(T value)
{
    std::ostringstream os ;
    os << value ;
    return os.str() ;
}
