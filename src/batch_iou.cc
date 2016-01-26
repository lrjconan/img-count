// Computes IOU score in batch.
// Input is two 4-D matrix [B, N, H, W], [B, M, H, W].
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
    // Grab the input tensor
    const Tensor& pred_tensor = context->input(0);
    const Tensor& gt_tensor = context->input(1);
    Tensor* output_tensor = NULL;

    TensorShape shape_out;
    shape_out.AddDim(pred_tensor.shape().dim_size(0));
    shape_out.AddDim(gt_tensor.shape().dim_size(0));

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
    int n = pred_tensor.shape().dim_size(0);
    int m = gt_tensor.shape().dim_size(0);
    auto pred = pred_tensor.tensor<float, 3>();
    auto gt = gt_tensor.tensor<float, 3>();
    int h = pred_tensor.shape().dim_size(1);
    int w = pred_tensor.shape().dim_size(2);
    auto output = output_tensor->matrix<float>();
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < m; ++j) {
        MatrixXfR a(h, w);
        MatrixXfR b(h, w);
        for (int y = 0; y < h; ++y) {
          for (int x = 0; x < w; ++x) {
            a(y, x) = pred(i, y, x);
            b(y, x) = gt(j, y, x);
          }
        }
        output(i, j) = ComputeIou(a, b);
      }
    } 
  }
};

REGISTER_KERNEL_BUILDER(Name("BatchIou").Device(DEVICE_CPU), BatchIouOp);
