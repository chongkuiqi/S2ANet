// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#pragma once
#include <torch/extension.h>
#include <torch/types.h>


// cpu和gpu版本的计算旋转iou的函数声明，函数定义均在各自的文件中
at::Tensor box_iou_rotated_cpu(
    const at::Tensor& boxes1,
    const at::Tensor& boxes2);

#ifdef WITH_CUDA
at::Tensor box_iou_rotated_cuda(
    const at::Tensor& boxes1,
    const at::Tensor& boxes2);
#endif


// python的接口，Interface for Python
// inline是C++的关键字，表示函数为内联函数，当不同的cpps include本header时，inline可以防止多个函数定义
// inline is needed to prevent multiple function definitions when this header is included by different cpps
inline at::Tensor box_iou_rotated(
    const at::Tensor& boxes1,
    const at::Tensor& boxes2) {
  // assert(boxes1.device().is_cuda() == boxes2.device().is_cuda());
  assert(boxes1.is_cuda() == boxes2.is_cuda());
  // if (boxes1.device().is_cuda()) {
  if (boxes1.is_cuda()) {
    #ifdef WITH_CUDA
        return box_iou_rotated_cuda(boxes1, boxes2);
    #else
        AT_ERROR("Not compiled with GPU support");
    #endif
  }

  return box_iou_rotated_cpu(boxes1, boxes2);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("box_iou_rotated", &box_iou_rotated, "IoU for rotated boxes");
}