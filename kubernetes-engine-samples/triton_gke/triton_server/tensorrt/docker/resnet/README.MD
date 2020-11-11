```bash

./onnx_to_tensorrt.py --explicit-batch --onnx resnet.onnx -b 16 -o resnet50.fp16.engine --fp16
```
TRITON model config with RESNET50
```
name: "ResNet50-FP16"
platform: "tensorrt_plan"
max_batch_size: 16
dynamic_batching {
  preferred_batch_size: [ 16 ]
  max_queue_delay_microseconds: 1000000
  default_queue_policy {
     timeout_action: DELAY
     default_timeout_microseconds: 1000000000
  }
}
input {
  name: "input_tensor:0"
}
output {
  name: "resnet50_v1.5/output/softmax:0"
  label_filename: "imagenet1k_labels.txt"
}
output {
  name: "ArgMax:0"
}
instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
    profile: "1"
  }
]
optimization {
  priority: PRIORITY_MAX
}
```

