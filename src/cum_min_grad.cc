// Computes the cumulative minimum of B vectors.
// Input shape: [B, D].
// Output shape: [B, D].

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"

using namespace tensorflow;

REGISTER_OP("CumMinGrad")
    .Input("grad_in: float")
    .Input("input: float")
    .Output("grad_out: float");

class CumMinGradOp : public OpKernel {
 public:
  explicit CumMinGradOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& grad_out_tensor = context->input(0);
    const Tensor& input_tensor = context->input(1);
    Tensor* grad_in_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_out_tensor.shape(),
                                                     &grad_in_tensor));
    ComputeCumMinGrad(grad_out_tensor, input_tensor, grad_in_tensor);
  }

 private:
  void ComputeCumMinGrad(const Tensor& grad_out_tensor,
                         const Tensor& input_tensor, Tensor* grad_in_tensor) {
    const auto& grad_out_mat = grad_out_tensor.matrix<float>();
    const auto& input_mat = input_tensor.matrix<float>();
    const auto& shape = grad_out_tensor.shape();
    auto grad_in_mat = grad_in_tensor->matrix<float>();
    for (int i = 0; i < shape.dim_size(0); ++i) {
      for (int j = 1; j < shape.dim_size(1); ++j) {
        grad_in_mat(i, j) = 0.0;
      }
    }
    for (int i = 0; i < shape.dim_size(0); ++i) {
      int cum_min_idx = 0;
      float cum_min = input_mat(i, 0);
      grad_in_mat(i, 0) = grad_out_mat(i, 0);
      for (int j = 1; j < shape.dim_size(1); ++j) {
        if (input_mat(i, j) < cum_min) {
          cum_min = input_mat(i, j);
          cum_min_idx = j;
        }
        grad_in_mat(i, cum_min_idx) += grad_out_mat(i, j);
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("CumMinGrad").Device(DEVICE_CPU), CumMinGradOp);
