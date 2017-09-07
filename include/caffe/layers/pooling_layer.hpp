#ifndef CAFFE_POOLING_LAYER_HPP_
#define CAFFE_POOLING_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Pools the input image by taking the max, average, etc. within regions.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class PoolingLayer : public Layer<Dtype> {
 public:
  explicit PoolingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Pooling"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  // MAX POOL layers can output an extra top blob for the mask;
  // others can only output the pooled inputs.
  virtual inline int MaxTopBlobs() const {
    return (this->layer_param_.pooling_param().pool() ==
            PoolingParameter_PoolMethod_MAX) ? 2 : 1;
  }

  /** KAK Methods meant to be exposed to Python */
  virtual const char* PoolingType() const {
      const char* type = "";
      switch (this->layer_param_.pooling_param().pool())
      {
      case PoolingParameter_PoolMethod_MAX:        type = "MAX";  
                                                   break;

      case PoolingParameter_PoolMethod_AVE:        type = "AVE";
                                                   break;

      case PoolingParameter_PoolMethod_STOCHASTIC: type = "STOCHASTIC";
                                                   break;
      }
      return type;
  }

  /** KAK  methods to support ELL implementation, */
  virtual inline int          Channels_    () const { return channels_; }
  virtual inline vector<int>  InputShape   () const { return inputShape; }
  virtual inline int          KernelHeight () const { return kernel_h_; }
  virtual inline int          KernelWidth  () const { return kernel_w_; }
  virtual inline vector<int>  OutputShape  () const { return outputShape; }
  virtual inline int          PadHeight    () const { return pad_h_; }
  virtual inline int          PadWidth     () const { return pad_w_; }
  virtual inline int          StrideHeight () const { return stride_h_; }
  virtual inline int          StrideWidth  () const { return stride_w_; }


 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int kernel_h_, kernel_w_;
  int stride_h_, stride_w_;
  int pad_h_, pad_w_;
  int channels_;
  int height_, width_;
  int pooled_height_, pooled_width_;
  bool global_pooling_;
  Blob<Dtype> rand_idx_;
  Blob<int> max_idx_;

  // KAK Added two following variables to track input and output shape.
  vector<int>  inputShape;
  vector<int>  outputShape;
};

}  // namespace caffe

#endif  // CAFFE_POOLING_LAYER_HPP_
