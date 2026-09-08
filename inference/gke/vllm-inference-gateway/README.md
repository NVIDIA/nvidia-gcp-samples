# vLLM Inference Gateway on GKE

Deploy two OpenAI-compatible vLLM model servers on GKE Autopilot and expose them through one GKE Inference Gateway endpoint. The sample routes requests by the `model` value in the JSON body:

- `google/gemma-4-E2B-it` routes to a Gemma vLLM server.
- `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` routes to a NVIDIA Nemotron vLLM server.

Both model servers request one `2g.48gb` MIG partition on an NVIDIA RTX PRO 6000 GPU. The Nemotron pod uses pod affinity to run on the same physical node as the Gemma pod, so the two servers share one physical GPU through separate MIG slices.

## Prerequisites

- A Google Cloud project with billing enabled.
- Quota for one NVIDIA RTX PRO 6000 GPU in the target region.
- `gcloud`, `kubectl`, `helm`, `curl`, and `jq`.
- A Hugging Face read token with access to:
  - https://huggingface.co/google/gemma-4-E2B-it
  - https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16

## Configure environment

```bash
export PROJECT_ID=<your-project-id>
export REGION=us-central1
export CLUSTER_NAME=vllm-inference
export NETWORK=default
export SUBNET=default
export HF_TOKEN=<your-hugging-face-token>

gcloud config set project ${PROJECT_ID}
```

Enable required Google Cloud APIs:

```bash
gcloud services enable \
  compute.googleapis.com \
  container.googleapis.com \
  networkservices.googleapis.com \
  logging.googleapis.com \
  monitoring.googleapis.com
```

Create a proxy-only subnet for the regional external Application Load Balancer used by the Gateway:

```bash
gcloud compute networks subnets create ${REGION}-proxy-only-subnet \
  --purpose=REGIONAL_MANAGED_PROXY \
  --role=ACTIVE \
  --region=${REGION} \
  --network=${NETWORK} \
  --range=10.0.200.0/23
```

Create an Autopilot cluster with GKE Gateway API and managed workload monitoring:

```bash
gcloud container clusters create-auto ${CLUSTER_NAME} \
  --project=${PROJECT_ID} \
  --region=${REGION} \
  --network=${NETWORK} \
  --subnetwork=${SUBNET} \
  --release-channel=rapid \
  --gateway-api=standard \
  --auto-monitoring-scope=ALL

gcloud container clusters get-credentials ${CLUSTER_NAME} \
  --region=${REGION} \
  --project=${PROJECT_ID}
```

Install the Inference Gateway CRDs:

```bash
kubectl apply -f \
  https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/v1.3.1/manifests.yaml
```

Install the custom metrics adapter and grant it Monitoring Viewer access. This is required by the sample HPA resources that read InferencePool metrics.

```bash
kubectl apply -f \
  https://raw.githubusercontent.com/GoogleCloudPlatform/k8s-stackdriver/master/custom-metrics-stackdriver-adapter/deploy/production/adapter_new_resource_model.yaml

PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format='value(projectNumber)')

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --role=roles/monitoring.viewer \
  --member=principal://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${PROJECT_ID}.svc.id.goog/subject/ns/custom-metrics/sa/custom-metrics-stackdriver-adapter
```

Create the Hugging Face token secret:

```bash
kubectl create secret generic hf-secret \
  --from-literal=hf_api_token=${HF_TOKEN} \
  --dry-run=client -o yaml | kubectl apply -f -
```

## Deploy vLLM model servers

From this directory:

```bash
kubectl apply -f manifests/vllm-gemma.yaml
kubectl wait deploy/vllm-gemma-4-e2b --for=condition=Available --timeout=15m

kubectl apply -f manifests/vllm-nemotron.yaml
kubectl wait deploy/vllm-nemotron-nano-4b --for=condition=Available --timeout=15m
```

Check the pods:

```bash
kubectl get pods -l app=vllm-gemma-4-e2b
kubectl get pods -l app=vllm-nemotron-nano-4b
```

## Deploy Inference Gateway

Install the body-based routing extension. It reads the `model` field from OpenAI-compatible requests and sets the `X-Gateway-Model-Name` header used by the HTTPRoutes.

```bash
helm upgrade --install bbr \
  --version v1.3.1 \
  --set provider.name=gke \
  --set inferenceGateway.name=vllm-xlb \
  oci://registry.k8s.io/gateway-api-inference-extension/charts/body-based-routing
```

Install one InferencePool per model server:

```bash
helm upgrade --install vllm-gemma-4-e2b \
  --dependency-update \
  --set inferencePool.modelServers.matchLabels.app=vllm-gemma-4-e2b \
  --set provider.name=gke \
  --version v1.3.1 \
  oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool \
  -f epp-values.yaml

helm upgrade --install vllm-nemotron-nano-4b \
  --dependency-update \
  --set inferencePool.modelServers.matchLabels.app=vllm-nemotron-nano-4b \
  --set provider.name=gke \
  --version v1.3.1 \
  oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool \
  -f epp-values.yaml
```

Apply the InferenceObjectives, Gateway, and HTTPRoutes:

```bash
kubectl apply -f manifests/inference-objectives.yaml
kubectl apply -f manifests/gateway.yaml
kubectl apply -f manifests/http-routes.yaml
```

Wait for the Gateway and routes:

```bash
kubectl wait gateway/vllm-xlb --for=condition=Programmed --timeout=10m
kubectl wait httproute/vllm-gemma-4-e2b-route --for=jsonpath='{.status.parents[0].conditions[?(@.type=="Accepted")].status}=True --timeout=5m
kubectl wait httproute/vllm-nemotron-nano-4b-route --for=jsonpath='{.status.parents[0].conditions[?(@.type=="Accepted")].status}=True --timeout=5m
```

## Test routing

Get the external Gateway address:

```bash
export GW_IP=$(kubectl get gateway/vllm-xlb -o jsonpath='{.status.addresses[0].value}')
export GW_PORT=80
```

Send a request to Gemma:

```bash
cat > gemma.json <<'EOF'
{
  "model": "google/gemma-4-E2B-it",
  "messages": [
    {
      "role": "user",
      "content": "Explain in three concise bullets why GPU partitioning can help serve multiple small language models on one accelerator."
    }
  ],
  "max_tokens": 256,
  "temperature": 0.2,
  "top_p": 0.95
}
EOF

curl -sS http://${GW_IP}:${GW_PORT}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @gemma.json | jq -r '.model, .choices[0].message.content'
```

Send a request to Nemotron:

```bash
cat > nemotron.json <<'EOF'
{
  "model": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
  "messages": [
    {
      "role": "user",
      "content": "Explain in three concise bullets why GPU partitioning can help serve multiple small language models on one accelerator."
    }
  ],
  "max_tokens": 256,
  "temperature": 0.2,
  "top_p": 0.95
}
EOF

curl -sS http://${GW_IP}:${GW_PORT}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @nemotron.json | jq -r '.model, .choices[0].message.content'
```

If the Gateway was created recently, wait a few minutes and retry if you receive a temporary `404`, `400`, or `503` while the load balancer finishes reconciling.

## Observe metrics

GKE managed workload monitoring collects vLLM and Inference Gateway metrics. Useful checks:

```bash
kubectl describe hpa vllm-gemma-4-e2b
kubectl describe hpa vllm-nemotron-nano-4b
```

In Cloud Monitoring, dashboard templates are available for:

- `vLLM Prometheus Overview`
- `GKE Inference Gateway Prometheus Overview`

## Clean up

Delete the application resources:

```bash
helm uninstall vllm-gemma-4-e2b --ignore-not-found
helm uninstall vllm-nemotron-nano-4b --ignore-not-found
helm uninstall bbr --ignore-not-found

kubectl delete -f manifests/http-routes.yaml --ignore-not-found
kubectl delete -f manifests/gateway.yaml --ignore-not-found
kubectl delete -f manifests/inference-objectives.yaml --ignore-not-found
kubectl delete -f manifests/vllm-nemotron.yaml --ignore-not-found
kubectl delete -f manifests/vllm-gemma.yaml --ignore-not-found
kubectl delete secret hf-secret --ignore-not-found
```

Delete the cluster and proxy-only subnet:

```bash
gcloud container clusters delete ${CLUSTER_NAME} \
  --region=${REGION} \
  --quiet

gcloud compute networks subnets delete ${REGION}-proxy-only-subnet \
  --region=${REGION} \
  --quiet
```
