# Copyright 2020 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import time, argparse
import subprocess, sys, os, json
import dask, dask_cudf, asyncio
import socket, gcsfs
from dask.distributed import Client
from dask.distributed import wait
import xgboost as xgb
import logging
import datetime

async def start_client(scheduler_addr, train_dir, model_file, num_workers, fs, parquet=False):
  async with Client(scheduler_addr, asynchronous=True) as client:
    dask.config.set({'distributed.scheduler.work-stealing': False})
    print(dask.config.get('distributed.scheduler.work-stealing'))
    dask.config.set({'distributed.scheduler.bandwidth': 1})
    print(dask.config.get('distributed.scheduler.bandwidth'))
    await client.wait_for_workers(num_workers)
    colnames = ['label'] + ['feature-%02d' % i for i in range(1, 29)]

    if parquet is True:
        df = dask_cudf.read_parquet(train_dir, columns=colnames)
    else:
        df = dask_cudf.read_csv(train_dir, header=None, names=colnames, chunksize=None)

    X = df[df.columns.difference(['label'])]
    y = df['label']

    df = df.persist()
    X = X.persist()
    wait(df)
    wait(X)
    print("[INFO]: ------ Long waited but the data is ready now")

    start_time = time.time()
    dtrain = await xgb.dask.DaskDeviceQuantileDMatrix(client, X, y)

    del df
    del X
    del y

    output = await xgb.dask.train(client,
                        { 'verbosity': 1,
                         'learning_rate': 0.1,
                          'max_depth': 8,
                          'objective': 'reg:squarederror',
                          'subsample': 0.6,
                          'gamma': 1,
                          'verbose_eval': True,
                          'tree_method':'gpu_hist',
                          'nthread': 1
                        },
                        dtrain,
                        num_boost_round=100, evals=[(dtrain, 'train')])
    logging.info("[debug:leader]: ------ training finished")
    output['booster'].save_model('/tmp/tmp.model')
    history = output['history']
    logging.info('[debug:leader]: ------ Training evaluation history:', history)
    fs.put('/tmp/tmp.model', model_file)
    logging.info("[debug:leader]: ------model saved")
    logging.info("[debug:leader]: ------ %s seconds ---" % (time.time() - start_time))
    predictions = xgb.dask.predict(client, output, dtrain)
    print(type(predictions))
    await client.shutdown()

def launch_dask(cmd, is_shell):
  return subprocess.Popen(cmd,
                    stdout=None,
                    stderr=None,
                    shell=is_shell)

def launch_worker(cmd):
  return subprocess.check_call(cmd,
                    stdout=sys.stdout,
                    stderr=sys.stderr)

if __name__=='__main__':
  now = datetime.datetime.utcnow()
  timestamp = now.strftime("%m%d%Y%H%M%S%f")
  parser = argparse.ArgumentParser()
  logging.basicConfig(format='%(message)s')
  logging.getLogger().setLevel(logging.INFO)
  parser.add_argument(
    '--gcp-project',
    type=str,
    help='User gcp project',
    required=True)
  parser.add_argument(
    '--train-files',
    type=str,
    help='Training files local or GCS',
    required=True)
  parser.add_argument(
    '--scheduler-ip-file',
    type=str,
    help='Scratch temp file to storage scheduler ip in GCS',
    required=True)
  parser.add_argument(
    '--model-file',
    type=str,
    help="""GCS or local dir for checkpoints, exports, and summaries.
    Use an existing directory to load a trained model, or a new directory
    to retrain""",
    required=True)
  parser.add_argument(
    '--num-workers',
    type=int,
    help='num of workers for rabit')
  parser.add_argument(
    '--rmm-pool-size',
    type=str,
    help='RMM pool size',
    default='8G')
  parser.add_argument(
    '--nthreads',
    type=str,
    help='nthreads for master and worker',
    default='4')
  parser.add_argument(
    '--parquet',
    action='store_true', help='parquet files are used')

  args, _ = parser.parse_known_args()
  tf_config_str = os.environ.get('TF_CONFIG')
  tf_config_json = json.loads(tf_config_str)
  logging.info(tf_config_json)
  task_name = tf_config_json.get('task', {}).get('type')
  fs = gcsfs.GCSFileSystem(project=args.gcp_project, token='cloud')
  if task_name == 'master':
    host_name = socket.gethostname()
    host_ip = socket.gethostbyname(host_name)
    with fs.open(args.scheduler_ip_file, 'w') as f:
      f.write(host_ip)
    scheduler_addr = host_ip + ':2222'
    logging.info('[INFO]: The scheduler IP is %s', scheduler_addr)
    proc_scheduler = launch_dask(f'dask-scheduler --protocol tcp > /tmp/scheduler.log 2>&1 &', True)
    logging.info('[debug:leader]: ------ start scheduler')
    proc_worker = launch_dask(['dask-cuda-worker', '--rmm-pool-size', args.rmm_pool_size, '--nthreads', args.nthreads , scheduler_addr], False)
    logging.info('[debug:leader]: ------ start worker')
    asyncio.get_event_loop().run_until_complete(start_client(scheduler_addr,
                             args.train_files,
                             args.model_file,
                             args.num_workers,
                             fs,
                             args.parquet))
  # launch dask worker, redirect output to sys stdout/err
  elif task_name == 'worker':
    while not fs.exists(args.scheduler_ip_file):
      time.sleep(1)
    with fs.open(args.scheduler_ip_file, 'r') as f:
      scheduler_ip = f.read().rstrip("\n")
    logging.info('[debug:scheduler_ip]: ------'+scheduler_ip)
    scheduler_addr = scheduler_ip + ':2222'
    proc_worker = launch_worker(['dask-cuda-worker', '--rmm-pool-size', args.rmm_pool_size, '--nthreads' , args.nthreads, scheduler_addr])