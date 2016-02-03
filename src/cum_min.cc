// Computes the cumulative minimum of B vectors.
// Input shape: [B, D].
// Output shape: [B, D].

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"

using namespace tensorflow;

REGISTER_OP("CumMin").Input("input: float").Output("output: float");

class CumMinOp : public OpKernel {
 public:
  explicit CumMinOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    ComputeCumMin(input_tensor, output_tensor);
  }

 private:
  void ComputeCumMin(const Tensor& input_tensor, Tensor* output_tensor) {
    const auto& input_mat = input_tensor.matrix<float>();
    const auto& shape = input_tensor.shape();
    auto output_mat = output_tensor->matrix<float>();
    for (int i = 0; i < shape.dim_size(0); ++i) {
      float cum_min = input_mat(i, 0);
      output_mat(i, 0) = cum_min;
      for (int j = 1; j < shape.dim_size(1); ++j) {
        cum_min = MIN(cum_min, input_mat(i, j));
        output_mat(i, j) = cum_min;
      }
    }
    // LOG(INFO) << "Input shape: " << shape.dim_size(0) << ", "
    //           << shape.dim_size(1);
    // LOG(INFO) << "Output shape: " << output_tensor->shape().dim_size(0) << ", "
    //           << output_tensor->shape().dim_size(1);
    // LOG(INFO) << "Cum min output: " << output_mat;
  }
};

REGISTER_KERNEL_BUILDER(Name("CumMin").Device(DEVICE_CPU), CumMinOp);
