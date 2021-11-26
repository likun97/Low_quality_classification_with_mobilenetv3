---
title: lowquality image classification
---

# lowquality porn nausea classification

### src/
include all tensorrt related files
edit for your files of specific model

### build/ 
* make
* ./exec          to test
* ./trt_build     to build trt from onnx
* ./trt_infer     to infer with trt model








## run
./run_qo.sh

## stop
./stop_lq_qo.sh

## Prerequires
* cmake >= 3.12
* opencv
* tensorrt >= 7.0.0.11
* cudnn >= 7.6.5
* cuda >= 10.2
* FreeImage
* ACE
* SSPlatform
* etc.
* 
### include/
include all the head files used by tensorrt & qo.

### qo/
include all qo related source files.

