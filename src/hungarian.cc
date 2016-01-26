// Implements the Hungarian algorithm.
// Input is a 2-D weight matrix w_{i, j}.
// Output is a matching M_{i, j}, and vertex covers u_{i, 0}, v_{0, j}.

#include <deque>
#include <iostream>
#include <limits>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"

using namespace tensorflow;

typedef Eigen::Matrix<float, -1, -1, Eigen::RowMajor> MatrixXfR;

REGISTER_OP("Hungarian")
    .Input("weights: float")
    .Output("matching: float")
    .Output("cover_x: float")
    .Output("cover_y: float");

class HungarianOp : public OpKernel {
 public:
  explicit HungarianOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    const auto& shape = input_tensor.shape();

    // Create an output tensor
    Tensor* matching_tensor = NULL;
    Tensor* cover_x_tensor = NULL;
    Tensor* cover_y_tensor = NULL;
    TensorShape shape_x;
    shape_x.AddDim(shape.dim_size(0));
    shape_x.AddDim(1);
    TensorShape shape_y;
    shape_y.AddDim(1);
    shape_y.AddDim(shape.dim_size(1));
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, shape, &matching_tensor));
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, shape_x, &cover_x_tensor));
    OP_REQUIRES_OK(context,
                   context->allocate_output(2, shape_y, &cover_y_tensor));

    const auto& inp = CopyInput(input_tensor);
    MatrixXfR cover_x = MatrixXfR::Zero(shape.dim_size(0), 1);
    MatrixXfR cover_y = MatrixXfR::Zero(1, shape.dim_size(1));
    MatrixXfR matching = MatrixXfR::Zero(shape.dim_size(0), shape.dim_size(1));

    MinWeightedBipartiteCover(inp, &matching, &cover_x, &cover_y);
    CopyOutput(matching, matching_tensor);
    CopyOutput(cover_x, cover_x_tensor);
    CopyOutput(cover_y, cover_y_tensor);
  }

 private:
  MatrixXfR CopyInput(const Tensor& tensor) {
    const auto& shape = tensor.shape();
    MatrixXfR copy =
        Eigen::Map<MatrixXfR>((float*)tensor.tensor_data().data(),
                              shape.dim_size(0), shape.dim_size(1));

    return copy;
  }

  void CopyOutput(const MatrixXfR& output, Tensor* output_tensor) {
    auto output_matrix = output_tensor->matrix<float>();
    const auto& shape = output_tensor->shape();
    for (int i = 0; i < shape.dim_size(0); ++i) {
      for (int j = 0; j < shape.dim_size(1); ++j) {
        output_matrix(i, j) = output(i, j);
      }
    }
  }

  bool Augment(const MatrixXfR& capacity, MatrixXfR& flow,
               MatrixXfR& residual) {
    int n = residual.outerSize();
    int s = 0;
    int t = n - 1;

    std::deque<int> q;
    q.push_back(s);

    bool* mark = (bool*)calloc(n, sizeof(bool));
    int* p = (int*)calloc(n, sizeof(int));
    bool found = false;

    for (int v = 0; v < n; ++v) {
      p[v] = -1;
    }

    while (q.size() > 0) {
      int v = q.front();
      q.pop_front();
      mark[v] = true;
      if (v == t) {
        found = true;
        break;
      }
      for (int u = 0; u < n; ++u) {
        if (!mark[u] && residual(v, u) > 0) {
          q.push_back(u);
          p[u] = v;
        }
      }
    }

    if (found) {
      float b = capacity.maxCoeff();
      int v = t;
      while (p[v] != -1) {
        b = MIN(b, residual(p[v], v));
        v = p[v];
      }

      v = t;
      while (p[v] != -1) {
        if (capacity(p[v], v) > 0) {
          flow(p[v], v) += b;
        } else {
          flow(v, p[v]) -= b;
        }
        residual(p[v], v) -= b;
        residual(v, p[v]) += b;
        v = p[v];
      }
    }

    delete mark;
    delete p;

    return found;
  }

  MatrixXfR MaxFlow(const MatrixXfR& capacity) {
    int n = capacity.outerSize();
    MatrixXfR flow = MatrixXfR::Zero(n, n);
    MatrixXfR residual(capacity);
    while (Augment(capacity, flow, residual));

    return flow;
  }

  void MaxBipartiteMatching(const MatrixXfR& graph, MatrixXfR* matching) {
    int n_X = graph.outerSize();
    int n_Y = graph.innerSize();
    int n = n_X + n_Y + 2;
    MatrixXfR capacity = MatrixXfR::Zero(n, n);
    int s = 0;
    int t = n_X + n_Y + 1;
    int x_start = 1;
    int y_start = n_X + 1;
    MatrixXfR ones = MatrixXfR::Constant(n, n, 1.0);
    capacity.block(x_start, y_start, n_X, n_Y) = graph.block(0, 0, n_X, n_Y);
    capacity.block(s, x_start, 1, n_X) = ones.block(s, x_start, 1, n_X);
    capacity.block(y_start, t, n_Y, 1) = ones.block(y_start, t, n_Y, 1);
    // LOG(INFO) << "reformed graph: \n" << capacity;

    MatrixXfR flow_max = MaxFlow(capacity);
    // LOG(INFO) << "max flow: \n" << flow_max;

    // MatrixXfR matching = MatrixXfR::Zero(n_X, n_Y);
    matching->block(0, 0, n_X, n_Y) =
        flow_max.block(x_start, y_start, n_X, n_Y);
    // LOG(INFO) << "matching: \n" << *matching;
    // LOG(INFO) << "saturate: " << IsBipartiteMatchingSaturate(*matching);
  }

  bool IsBipartiteMatchingSaturate(const MatrixXfR& matching) {
    int n_X = matching.outerSize();
    int n_Y = matching.innerSize();

    if (n_X >= n_Y) {
      // Each vertex in Y needs to match to vertex in X.
      for (int j = 0; j < n_Y; ++j) {
        float sum = 0;
        for (int i = 0; i < n_X; ++i) {
          sum += matching(i, j);
        }
        if (sum == 0) {
          return false;
        }
      }
      return true;
    } else {
      // Each vertex in X needs to match to vertex in Y.
      for (int i = 0; i < n_X; ++i) {
        float sum = 0;
        for (int j = 0; j < n_Y; ++j) {
          sum += matching(i, j);
        }
        if (sum == 0) {
          return false;
        }
      }
      return true;
    }
  }

  void GetSetBipartiteNeighbours(const std::set<int>& set,
                                 const MatrixXfR& graph,
                                 std::set<int>* neighbours) {
    neighbours->clear();
    int n_Y = graph.innerSize();
    for (std::set<int>::iterator it = set.begin(); it != set.end(); ++it) {
      int v = *it;
      for (int u = 0; u < n_Y; ++u) {
        if (graph(v, u) > 0) {
          neighbours->insert(u);
        }
      }
    }
  }

  bool SetEqual(const std::set<int>& a, const std::set<int>& b) {
    if (a.size() != b.size()) {
      return false;
    }
    for (std::set<int>::iterator it = a.begin(); it != a.end(); ++it) {
      if (b.find(*it) == a.end()) {
        return false;
      }
    }
    return true;
  }

  void PrintSet(const std::set<int>& s) {
    std::cout << "{";
    for (std::set<int>::iterator it = s.begin(); it != s.end(); ++it) {
      std::cout << *it << ", ";
    }
    std::cout << "}" << std::endl;
  }

  int GetMatchedX(int y, const MatrixXfR& matching) {
    int n_X = matching.outerSize();
    for (int u = 0; u < n_X; ++u) {
      if (matching(u, y) == 1.0) {
        return u;
      }
    }
    return -1;
  }

  int GetMatchedY(int x, const MatrixXfR& matching) {
    int n_Y = matching.innerSize();
    for (int v = 0; v < n_Y; ++v) {
      if (matching(x, v) == 1.0) {
        return v;
      }
    }
    return -1;
  }

  MatrixXfR GetEqualityGraph(const MatrixXfR& weights, const MatrixXfR& cover_x,
                             const MatrixXfR& cover_y) {
    int n_X = weights.outerSize();
    int n_Y = weights.innerSize();
    MatrixXfR equality = MatrixXfR::Zero(n_X, n_Y);
    for (int x = 0; x < n_X; ++x) {
      for (int y = 0; y < n_Y; ++y) {
        // LOG(INFO) << "x: " << x << " y: " << y << " cx: " << cover_x(x, 0)
        //           << " cy: " << cover_y(0, y) << " w: " << weights(x, y);
        if (cover_x(x, 0) + cover_y(0, y) - weights(x, y) == 0.0) {
          equality(x, y) = 1.0;
        }
      }
    }
    return equality;
  }

  void MinWeightedBipartiteCover(const MatrixXfR& weights, MatrixXfR* matching,
                                 MatrixXfR* cover_x, MatrixXfR* cover_y) {
    int n_X = weights.outerSize();
    int n_Y = weights.innerSize();
    MatrixXfR maxCoeff = weights.rowwise().maxCoeff();
    MatrixXfR& c_x = *cover_x;
    MatrixXfR& c_y = *cover_y;
    MatrixXfR& M = *matching;
    for (int x = 0; x < n_X; ++x) {
      c_x(x, 0) = maxCoeff(x, 0);
    }
    for (int y = 0; y < n_Y; ++y) {
      c_y(0, y) = 0.0f;
    }
    for (int x = 0; x < n_X; ++x) {
      for (int y = 0; y < n_Y; ++y) {
        M(x, y) = 0.0f;
      }
    }
    // LOG(INFO) << "initial cover x: \n" << c_x;
    // LOG(INFO) << "initial cover y: \n" << c_y;

    MatrixXfR equality(n_X, n_Y);
    std::set<int> S;
    std::set<int> T;
    bool next_match = true;
    // int i = 0;

    while (true) {
      // LOG(INFO) << "-----------------------------";
      // LOG(INFO) << "iteration " << i;
      equality = GetEqualityGraph(weights, c_x, c_y);
      // LOG(INFO) << "equality graph: \n" << equality;
      if (next_match) {
        MaxBipartiteMatching(equality, matching);
        // LOG(INFO) << "new matching: \n" << M;
        if (IsBipartiteMatchingSaturate(M)) {
          // LOG(INFO) << "found solution, exit";
          // LOG(INFO) << "-----------------------------";
          return;
        }

        for (int u = 0; u < n_X; ++u) {
          if (GetMatchedY(u, M) == -1) {
            S.insert(u);
            break;
          }
        }
      }

      std::set<int> N_S;
      GetSetBipartiteNeighbours(S, equality, &N_S);
      // LOG(INFO) << "S: ";
      // PrintSet(S);
      // LOG(INFO) << "T: ";
      // PrintSet(T);
      // LOG(INFO) << "N_S: ";
      // PrintSet(N_S);

      if (SetEqual(N_S, T)) {
        // LOG(INFO) << "N_S == T";
        // LOG(INFO) << "Update cover";
        float a = std::numeric_limits<float>::max();
        for (std::set<int>::iterator it = S.begin(); it != S.end(); ++it) {
          int x = *it;
          for (int y = 0; y < n_Y; ++y) {
            if (T.find(y) == T.end()) {
              a = MIN(a, c_x(x, 0) + c_y(0, y) - weights(x, y));
            }
          }
        }
        // LOG(INFO) << "a: " << a;
        for (std::set<int>::iterator it = S.begin(); it != S.end(); ++it) {
          int x = *it;
          c_x(x, 0) -= a;
        }
        for (std::set<int>::iterator it = T.begin(); it != T.end(); ++it) {
          int y = *it;
          c_y(0, y) += a;
        }
        // LOG(INFO) << "cover x: \n" << c_x;
        // LOG(INFO) << "cover y: \n" << c_y;
      } else {
        // LOG(INFO) << "N_S != T";
        while (N_S.size() > T.size()) {
          int y;
          for (std::set<int>::iterator it = N_S.begin(); it != N_S.end();
               ++it) {
            y = *it;
            if (T.find(y) == T.end()) {
              // LOG(INFO) << "pick y in N_S not in T: " << y;
              break;
            }
          }

          int z = GetMatchedX(y, M);
          if (z == -1) {
            // LOG(INFO) << "y unmatched, look for matching";
            next_match = true;
            break;
          } else {
            // LOG(INFO) << "y matched, increase S and T";
            next_match = false;
            S.insert(z);
            for (int v = 0; v < n_Y; ++v) {
              if (equality(z, v) > 0.0) {
                N_S.insert(v);
              }
            }
            T.insert(y);
            // LOG(INFO) << "S: ";
            // PrintSet(S);
            // LOG(INFO) << "T: ";
            // PrintSet(T);
            // LOG(INFO) << "N_S: ";
            // PrintSet(N_S);
          }
        }
      }

      // LOG(INFO) << "end of iteration";
      // LOG(INFO) << "-----------------------------";
      // i++;
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("Hungarian").Device(DEVICE_CPU), HungarianOp);
