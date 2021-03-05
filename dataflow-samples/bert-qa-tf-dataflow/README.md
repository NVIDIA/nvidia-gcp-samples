# bert-qa-tf-trt

Running question answering models using tf-cpu, tf-gpu, as well as trt and compare their run time and cost metrics.

There are several steps we need to take to run this repo:

1. creating a virual environment <br>
`virtualenv -p <point_to_python3.6_version> <env_name>` # Python versions are important as Dataflow is super picky and comlains if versions do not match
2. activate the created virtual environment <br>
`source <env_name>/bin/activate` # make sure you don't have any other virtual env is running - if there is one already running you need to deactivate it before starting to this one
3. install necessary packages <br>
`pip install apache-beam[gcp]` <br>
`pip install tensorflow==2.3.1` <br>
4. get the fine-tuned question answering model based on BERT Large <br>
`wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/bert_tf_savedmodel_large_qa_squad2_amp_384/versions/19.03.0/zip -O bert_tf_savedmodel_large_qa_squad2_amp_384_19.03.0.zip` # this saved model is available on NGC <br>
4.1 you might need to arrange the unzipped folder and adjust that path in the .py files
5. get the vocab.txt file <br>
`wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/bert_tf_ckpt_large_qa_squad2_amp_128/versions/19.03.1/zip -O bert_tf_ckpt_large_qa_squad2_amp_128_19.03.1.zip` <br>
5.1 you need to uncompress this file to get vocab.txt, you do not need rest of the files
6. build and push the custom image <br>
`bash build_and_push.sh` # you need to update your project-id before running this file
7. run cpu version <br>
`bash run_cpu.sh` # this will run a dataflow job using cpus, you need to update this file with your project-id - you might want to change the region as well as the machine type as well.
8. run GPU version <br>
`bash run_gpu.sh` # this will run a dataflow job using GPUs, you need to update this file with your project-id


<br><br>
TensorRT version can be found [here](https://github.com/NVIDIA/nvidia-gcp-samples/tree/master/dataflow-samples/bert-qa-trt-dataflow).
