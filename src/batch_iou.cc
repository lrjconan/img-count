// Computes IOU score in batch.
// Input is two 4-d tensors [B, N, H, W], [B, M, H, W].
// Output is 3-d tensor [B, N, M].

#include <deque>
#include <iostream>
#include <limits>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"

using namespace tensorflow;

REGISTER_OP("BatchIou")
    .Input("pred: float")
    .Input("gt: float")
    .Output("score: float");

typedef Eigen::Matrix<float, -1, -1, Eigen::RowMajor> MatrixXfR;

class BatchIouOp : public OpKernel {
 public:
  explicit BatchIouOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& pred_tensor = context->input(0);
    const Tensor& gt_tensor = context->input(1);

    Tensor* output_tensor = NULL;
    TensorShape shape_out;
    shape_out.AddDim(pred_tensor.shape().dim_size(0));
    shape_out.AddDim(pred_tensor.shape().dim_size(1));
    shape_out.AddDim(gt_tensor.shape().dim_size(1));

    OP_REQUIRES_OK(context,
                   context->allocate_output(0, shape_out, &output_tensor));
    ComputeBatchIou(pred_tensor, gt_tensor, output_tensor);
  }

 private:
  MatrixXfR Union(const MatrixXfR& a, const MatrixXfR& b) {
    return (a + b) - a.cwiseProduct(b);
  }

  MatrixXfR Intersection(const MatrixXfR& a, const MatrixXfR& b) {
    return a.cwiseProduct(b);
  }
  
  float ComputeIou(const MatrixXfR& a, const MatrixXfR& b) {
    return Intersection(a, b).sum() / Union(a, b).sum();
  }

  void ComputeBatchIou(const Tensor& pred_tensor, const Tensor& gt_tensor,
                       Tensor* output_tensor) {
    int B = pred_tensor.shape().dim_size(0);
    int N = pred_tensor.shape().dim_size(1);
    int M = gt_tensor.shape().dim_size(1);
    auto pred = pred_tensor.tensor<float, 4>();
    auto gt = gt_tensor.tensor<float, 4>();
    int H = pred_tensor.shape().dim_size(2);
    int W = pred_tensor.shape().dim_size(3);
    auto output = output_tensor->tensor<float, 3>();
    for (int i = 0; i < B; ++i) {
      for (int n = 0; n < N; ++n) {
        for (int m = 0; m < M; ++m) {
          MatrixXfR a(H, W);
          MatrixXfR b(H, W);
          for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
              a(y, x) = pred(i, n, y, x);
              b(y, x) = gt(i, m, y, x);
            }
          }
          output(i, n, m) = ComputeIou(a, b);
        }
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("BatchIou").Device(DEVICE_CPU), BatchIouOp);
