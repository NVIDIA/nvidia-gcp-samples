# BUILDING A COMPUTER VISION SERVICE USING NVIDIA NGC AND GOOGLE CLOUD

The following instruction is used in the NVIDIA GCP Lab [Link](https://info.nvidia.com/ngc-google-cloud-computer-vision-webinar.html).

Before going through the lab, download notebook and configuration files from [NVIDIA NGC resource](https://ngc.nvidia.com/catalog/resources/nvidia:building_cv_service_using_nvidia_ngc_and_triton_in_google_cloud).

## Prerequisite

[NVIDIA NGC Base VM Images](https://console.cloud.google.com/marketplace/product/nvidia-ngc-public/nvidia-ngc-base-test-b) is available in Google Cloud Marketplace, in the following example, we use GCP's latest A100 GPU in A2 family to launch a NGC VM. NGC VM preinstalls NVIDIA Docker. 

[TAO(TLT) Stream Analysytics container](https://ngc.nvidia.com/catalog/containers/nvidia:tlt-streamanalytics) provides the runtime dependencies for steps below. It can be launched with
```
docker run --gpus all -it  \
    --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -p8888:8888 \
    -v "$(pwd)":/tlt nvcr.io/nvidia/tlt-streamanalytics:v3.0-py3
```

## Download RGB Pretrained PeopleNet from NGC and Deploy to Triton

```
mkdir /tlt/models
ngc registry model download-version "nvidia/tlt_peoplenet:unpruned_v2.1"
```

First, we convert the pretrained models weights to a etlt format, then optimized to a TensorRT Engine.
```
detectnet_v2 export -k "tlt_encode" -m /tlt/models/tlt_peoplenet_vunpruned_v2.1/resnet34_peoplenet.tlt -o /tlt/models/tlt_peoplenet_vunpruned_v2.1/resnet34_peoplenet.etlt
tlt-converter -k "tlt_encode" -d 3,544,960 -e /tlt/models/tlt_peoplenet_vunpruned_v2.1/model.engine -o output_cov/Sigmoid,output_bbox/BiasAdd /tlt/models/tlt_peoplenet_vunpruned_v2.1/resnet34_peoplenet.etlt
```

Second, we move the pretrained engine to a GCS directory.
```
gsutil cp ~/tlt/models/tlt_peoplenet_vunpruned_v2.1/model.engine gs://dongm-tlt/tlt/triton-model/peoplenet_tlt/1/model.plan
```

Here is a sample [Triton configuration file](https://github.com/NVIDIA-AI-IOT/tlt-triton-apps/blob/main/model_repository/peoplenet_tlt/config.pbtxt) for the RGB PeopleNet model, we also drop it to the same bucket.
```
wget https://raw.githubusercontent.com/NVIDIA-AI-IOT/tlt-triton-apps/main/model_repository/peoplenet_tlt/config.pbtxt
gsutil cp config.pbtxt gs://dongm-tlt/tlt/triton-model/peoplenet_tlt/
```

The GKE A100 cluster could be created with command below:
```
export PROJECT_ID=k80-exploration
export ZONE=us-central1-a
export REGION=us-central1
export DEPLOYMENT_NAME=dongm-cv-gke

gcloud beta container clusters create ${DEPLOYMENT_NAME} \
--addons=HorizontalPodAutoscaling,HttpLoadBalancing,Istio \
--machine-type=n1-standard-8 \
--node-locations=${ZONE} \
--zone=${ZONE} \
--subnetwork=default \
--scopes cloud-platform \
--num-nodes 1 \
--project ${PROJECT_ID}

# add GPU node pools, user can modify number of node based on workloads
gcloud container node-pools create accel \
  --project ${PROJECT_ID} \
  --zone ${ZONE} \
  --cluster ${DEPLOYMENT_NAME} \
  --num-nodes 1 \
  --accelerator type=nvidia-tesla-a100,count=1 \
  --enable-autoscaling --min-nodes 1 --max-nodes 2 \
  --machine-type a2-highgpu-1g \
  --disk-size=100 \
  --scopes cloud-platform \
  --verbosity error

# so that you can run kubectl locally to the cluster
gcloud container clusters get-credentials ${DEPLOYMENT_NAME} --project ${PROJECT_ID} --zone ${ZONE}  

# deploy NVIDIA device plugin for GKE to prepare GPU nodes for driver install
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml

# make sure you can run kubectl locally to access the cluster
kubectl create clusterrolebinding cluster-admin-binding --clusterrole cluster-admin --user "$(gcloud config get-value account)"

# enable stackdriver custom metrics adaptor
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/k8s-stackdriver/master/custom-metrics-stackdriver-adapter/deploy/production/adapter.yaml
```

Last, we use [GKE Triton Marketplace Application](https://console.cloud.google.com/marketplace/details/nvidia-ngc-public/triton-inference-server) to launch a Triton deployment in GKE, pointing model repository to `gs://dongm-tlt/tlt/triton-model`. Given the compatibilty required by TensorRT, please note we will launch a A100 GKE cluster to host the application, and in the meanwhile, use Triton `v2.5.0` version of GKE application to be compatible with TAO 3.0. 

## Inference Experiment with RGB Images

Once the application has been deployed, we will leverage [TAO Triton Application](https://github.com/NVIDIA-AI-IOT/tlt-triton-apps) to send inference request. Follow [getting started guide](https://github.com/NVIDIA-AI-IOT/tlt-triton-apps#quick-start-instructions) to set up python environment, but skip start server part as server has been deployed to GKE. Find out Istio ingress IP with: `export INGRESS_HOST=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
`, for example: `34.141.169.102`. 

In `triton_dev` environment, install jupyter and ipython. 

Launch a jupyter notebook from the GCP VM ``
And set up a port forwarding to your client `gcloud compute ssh --project k80-exploration --zone us-central1-a dongm-nvidia-ngc-base-vm -- -L 8888:localhost:8888`

And run the client python script in the notebook `tao_triton_client.ipynb` or with command below:
```
python3 ${TLT_TRITON_REPO_ROOT}/tlt_triton/python/entrypoints/tlt_client.py ../inference_images/rgb \
        -m peoplenet_tlt \
        -x 1 -b 16 --mode DetectNet_v2 \
        -i https -u 34.134.191.228:80 --async \
        --output_path ../inference_images_out \
        --postprocessing_config ../inference_config/clustering_config_peoplenet_rgb.prototxt 
```

## Prepare FLIR Thermal Dataset(IR Images)

Follow notebook `ir2kitti_preprocess.ipynb` to convert data label to kitti format, preprocess the images, generate tf records for Training.

## Retrain PeopleNet with FLIR Thermal Dataset(IR Images)

We modify the TAO Training configuration (training_config/training_spec.txt in NGC resource zip file) to point to the TF records location and kick off a retraining.
```
detectnet_v2 train -e training_config/training_spec.txt -r experiments -n "final_model" -k "tlt_encode" --gpus 8
```

In the example configuration, we train for 60 epochs and the accuracy could reach 72% by the end of training.

## Deploy IR PeopleNet to Triton

First, we convert the retrained models weights to a etlt format, then optimized to a TensorRT Engine.
```
detectnet_v2 export -m /tlt/experiments/weights/final_model.tlt -o /tlt/models/ir.etlt -k "tlt_encode"
tlt-converter -k "tlt_encode" -d 3,544,960 -e /tlt/models/ir.engine -o output_cov/Sigmoid,output_bbox/BiasAdd /tlt/models/ir.etlt 
```
 
Second, we move the retrained engine to a GCS directory.
```
gsutil cp /tlt/models/ir.engine gs://dongm-tlt/tlt/triton-model/peoplenet_ir_tlt/1/model.plan
```

We also modify the config.pbtxt file to fit IR model as below,
```
name: "peoplenet_ir_tlt"
platform: "tensorrt_plan"
max_batch_size: 16
input [
  {
    name: "input_1"
    data_type: TYPE_FP32
    format: FORMAT_NCHW
    dims: [ 3, 544, 960 ]
  }
]
output [
  {
    name: "output_bbox/BiasAdd"
    data_type: TYPE_FP32
    dims: [ 4, 34, 60 ]
  },
  {
    name: "output_cov/Sigmoid"
    data_type: TYPE_FP32
    dims: [ 1, 34, 60 ]
  }
]
dynamic_batching { }
```
then copy:
```
gsutil cp config.pbtxt gs://dongm-tlt/tlt/triton-model/peoplenet_tlt/
```
 
Last, we use [GKE Triton Marketplace Application](https://console.cloud.google.com/marketplace/details/nvidia-ngc-public/triton-inference-server) to launch a Triton deployment in GKE, pointing model repository to `gs://dongm-tlt/tlt/triton-model`. Given the compatibilty required by TensorRT, please note we will launch a A100 GKE cluster to host the application, and in the meanwhile, use Triton `v2.5.0` version of GKE application to be compatible with TAO 3.0.
 
## Inference Experiment with IR Images

And run the client python script in the notebook `tao_triton_client.ipynb` or with command below
```
python3 ${TLT_TRITON_REPO_ROOT}/tlt_triton/python/entrypoints/tlt_client.py directory_to_test_images \
        -m peoplenet_tlt \
        -x 1 -b 16 --mode DetectNet_v2 \
        --class_list person \
        -i https -u 34.141.169.102:80 --async \
        --output_path directory_to_test_images_output \
        --postprocessing_config inference_config/clustering_config_peoplenet_ir.prototxt 
```
