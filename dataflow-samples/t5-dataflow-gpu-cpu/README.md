# Running T5 on Dataflow GPU and CPU


There are several steps we need to take to run this repo:

1. creating a virual environment <br>
`virtualenv -p <point_to_python3.6_version> <env_name>` # Python versions are important as Dataflow is super picky and comlains if versions do not match
2. activate the created virtual environment <br>
`source <env_name>/bin/activate` # make sure you don't have any other virtual env is running - if there is one already running you need to deactivate it before starting to this one
3. install necessary packages <br>
`pip install apache-beam[gcp]` <br>
`pip install tensorflow==2.3.1` <br>
`pip install tensorflow_text==2.3.0` <br>
4. get Google research repo to export model
`git clone git@github.com:google-research/google-research.git` <br>
4.1 add this repo to your pythonpath <br>
`export PYTHONPATH=$PYTHONPATH:<path_to_here>/google-research/t5_closed_book_qa`
5. export a t5 model <br>
`bash get_model.sh` # this call will export a T5 model, "small_ssm_nq", with a batch size of 16 and saved it to the `model_b16` directory
6. build and push the custom image <br>
`bash build_and_push.sh` # you need to update your project-id before running this file
7. run cpu version <br>
`bash run_cpu.sh` # this will run a dataflow job using cpus, you need to update this file with your project-id - you might want to change the region as well as the machine type as well.
8. run GPU version <br>
`bash run_gpu.sh` # this will run a dataflow job using GPUs, you need to update this file with your project-id

The GPU provides 10x less cost for similar run times compared to CPU. Alternatively, GPU runs 3x faster compared to CPU for a 1/3x cost.


<br><br>
These examples are derived from [here](https://cloud.google.com/blog/products/data-analytics/ml-inference-in-dataflow-pipelines) and [here](https://github.com/NVIDIA/nvidia-gcp-samples/tree/master/dataflow-samples/bert-qa-trt-dataflow).
