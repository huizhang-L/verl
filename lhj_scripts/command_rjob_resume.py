# 提交训练任务到火山集群
# 支持 checkpoint 断点续训
import os
import os.path as osp
import re
import subprocess
import time
import glob
import random
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from numpy import str_
import yaml
import json
import requests
import argparse
import multiprocessing as mp
from pathlib import Path

import mmengine
from mmengine.utils import track_parallel_progress

# 全局缓存，防止重复 addHandler
_logger_cache: Dict[str, logging.Logger] = {}



def get_logger(
    name: str = "RJOBRunner",
    *,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    fmt: str = "%(asctime)s | %(levelname)5s | %(processName)s | PID %(process)d | %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> logging.Logger:
    """
    获取（并缓存）一个 logger。

    默认行为：
        • 主 / 子进程都只向终端打印，不写文件。
        • 如需同时写文件，传入 log_file="your.log" 即可。
    
    Args:
        name (str):     logger 名称（同名 logger 全局唯一，避免重复 handler）
        log_file (str): 文件路径；为 None 表示不落盘
        level (int):    logging 级别
        fmt (str):      日志格式字符串
        datefmt (str):  时间戳格式
        logger = get_logger(log_file="logs/volc_runner.log")   # 显式指定
    """
    # 若已创建同名 logger，直接复用
    if name in _logger_cache:
        return _logger_cache[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # 禁止向根 logger 逐级冒泡，避免重复输出

    formatter = logging.Formatter(fmt, datefmt=datefmt)

    # -------- Console Handler：所有进程都打印 --------
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # -------- File Handler：仅在显式指定时启用 --------
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    _logger_cache[name] = logger
    return logger


def parse_arguments():
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(
        description='配置 rjob 提交到 H 集群的参数'
    )

    
    parser.add_argument(
        '--priority',
        type=int,
        default=9,
        help='提交的任务的优先级'
    )
    parser.add_argument(
        '--max_num_workers',
        type=int,
        default=1,
        help='最大进程数量，控制了最多同时多少个任务提交到火山云集群'
    )

    parser.add_argument(
        '--retry',
        type=int,
        default=5,
        help='当任务失败时，最大重试次数'
    )  

    # 关于任务配置的参数
    parser.add_argument(
        '--task_cfg_path',  # 长短参数名
        type=str,          # 参数类型
        default='',     # 必需参数
        help='配置好的 job 文件的路径'
    )
    # 解析参数并返回
    return parser.parse_args()


class LarkReporter:

    def __init__(self, url: str):
        self.url = url

    def post(self,
             content: Union[str, List[List[Dict]]],
             title: Optional[str] = None):
        """Post a message to Lark.

        When title is None, message must be a str. otherwise msg can be in rich
        text format (see
        https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/im-v1/message/create_json#45e0953e
        for details).
        """
        if title is None:
            assert isinstance(content, str)
            msg = {'msg_type': 'text', 'content': {'text': content}}
        else:
            if isinstance(content, str):
                content = [[{'tag': 'text', 'text': content}]]
            msg = {
                'msg_type': 'post',
                'content': {
                    'post': {
                        'zh_cn': {
                            'title': title,
                            'content': content
                        }
                    }
                }
            }
        requests.post(self.url, data=json.dumps(msg))


class TrainingTask:
    """
    描述单个训练任务。等价于 dataclass 版本，但手动写 __init__ 等方法。
    """
    def __init__(self,
                 bash_path: str,
                 name: str,
                 nnodes: int,
                 n_gpus_per_node: int,
                 output_dir: str):
        self.bash_path = bash_path
        self.name = name 
        self.nnodes = nnodes
        self.n_gpus_per_node = n_gpus_per_node
        self.output_dir = output_dir

    def get_cmd(self, template: str) -> str:
        """
        生成最终的启动命令。template 应包含一个 {task_cmd} 占位符。
        例如:
          "{task_cmd} > logs/{name}.log 2>&1"
        """
        core_cmd = (
            f"bash {self.bash_path}"
        )
        return template.format(task_cmd=core_cmd)

    def get_log_path(self) -> str:
        """
        从 self.train_config 指向的 YAML 中读取 output_dir，
        并返回 output_dir/log 目录的绝对路径。
        """

        # 3. 拼接 log 子目录
        # log_dir = os.path.join(self.output_dir, 'logs')

        # # 可选：创建该目录（如果你希望在这里就保证目录存在）
        self.output_dir = self.output_dir.replace("/nvme/lvhuijie/mnt/ailab-llmfudan", "/mnt/ailab-llmfudan/lvhuijie")
        os.makedirs(self.output_dir, exist_ok=True)

        log_path = os.path.join(self.output_dir, f'{self.name}.log')

        return log_path


class RJOBRunner:
    def __init__(
        self,
        max_num_workers: int = 32,
        retry: int = 100,
        priority: int = 9,
        debug: bool = False,
    ):
        self.max_num_workers = max_num_workers
        self.retry = retry
        self.debug = debug
        self.priority = priority

    def __call__(self, tasks: List[TrainingTask]):
        """Launch multiple tasks and summarize the results.

        Args:
            tasks (list[TrainingTask]): A list of task configs, usually generated by
                Partitioner.
        """
        status = self.launch(tasks)
        status_list = list(status)  # change into list format
        # self.summarize(status_list)

    def launch(self, tasks: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
        """Launch multiple tasks."""
        if not self.debug:
            status = track_parallel_progress(
                self._launch,
                tasks,
                nproc=self.max_num_workers,
                keep_order=False,
            )
        else:
            status = [self._launch(task, random_sleep=True) for task in tasks]
        return status

    def _run_task(self, task_name, log_path, poll_interval=60):
        """Poll rjob status until both active and pending are 0.

        Break if no dict line is found.
        """
        logger = get_logger()
        logger.info(task_name)
        status = None
        time.sleep(10)
        while True:
            get_cmd = f'brainctl get rjob {task_name} -n ailab-llmfudan'
            get_result = subprocess.run(get_cmd,
                                        shell=True,
                                        text=True,
                                        capture_output=True)
            output = get_result.stdout

            # check if the command is executed successfully
            if get_result.returncode != 0:
                logger.error(f'rjob get command failed: {get_result.stderr}')
                logger.info('retrying...')
                status = 'ERROR'
                continue

            found_dict = False
            for line in output.splitlines():
                # logger.info(f'line: {line}')
                if 'verl' not in line:
                    continue
                if 'Unknown' in line:
                    status = 'Unknown'
                    found_dict = True
                    break
                if 'Starting' in line:
                    status = 'Starting'
                    found_dict = True
                    break
                if 'Pending' in line:
                    status = 'Pending'
                    found_dict = True
                    break
                if 'Running' in line:
                    status = 'Running'
                    found_dict = True
                    break
                if 'Timeout' in line:
                    status = 'Timeout'
                    found_dict = True
                    break
                if 'Restarting' in line:
                    status = 'Restarting'
                    found_dict = True
                    break
                if 'Queued' in line:
                    status = 'Queued'
                    found_dict = True
                    break
                if 'Suspended' in line:
                    status = 'Suspended'
                    found_dict = True
                    break
                if 'Submitted' in line:
                    status = 'Submitted'
                    found_dict = True
                    break 
                if 'Inqueue' in line:
                    status = 'Inqueue'
                    found_dict = True
                    break 
                if 'Succeeded' in line:
                    status = 'FINISHED'
                    break
                if 'Stopped' in line:
                    status = 'STOPPED'
                    break
                if 'Failed' in line or 'failed' in line:
                    status = 'FAILED'
                    break
                if 'Cancelled' in line:
                    status = 'CANCELLED'
                    break
                logger.warning(f'Unrecognized status in: {output}')
            if found_dict:
                time.sleep(poll_interval)
                continue
            break
        logger.info(f'[RJOB] Final status returned: {status}')
        logger.info(log_path)
        if log_path:
            # get_cmd = f'rjob logs job {task_name}'
            # get_result = subprocess.run(get_cmd,
            #                             shell=True,
            #                             text=True,
            #                             capture_output=True)
            # # logger.info(get_result)
            # output = get_result.stdout
            with open(log_path, 'w', encoding='utf-8') as f:
                # f.write(output + '\n' + status)
                f.write(status)
        return status

    def _launch(self, task: TrainingTask, random_sleep: Optional[bool] = None):
        num_gpus = task.n_gpus_per_node
        num_nodes = task.nnodes
        task_name = task.name

        # Build up VCC command
        pwd = os.getcwd()

        """Launch a single task via rjob bash script."""
        if random_sleep is None:
            random_sleep = self.max_num_workers > 8
        if random_sleep:
            sleep_time = random.randint(0, 60)
            logger = get_logger()
            logger.info(f'Sleeping for {sleep_time} seconds to launch task')
            time.sleep(sleep_time)
        # Normalize task name
        logger = get_logger()
        # logger.info(f'Task config: {cfg}')
        # Obtain task_id in safe way, if not exist, use default value
        logger.info(f'Task name: {task_name}')
        # Generate temporary parameter file
        pwd = os.getcwd()

        # try:
        # Construct rjob submit command arguments
        args = []
        # Basic parameters
        args.append(f'--name={task_name}')
        # args.append(f'--task_name={task_name}')
        if num_gpus > 0:
            args.append(f'--gpu={num_gpus}')
            args.append(f'--memory={num_gpus * 160000}')
            args.append(f'--cpu={num_gpus * 16}')
        # else:
        #     if hasattr(task, 'memory'):
        #         args.append(f'--memory={getattr(task, "memory")}')
        #     elif self.rjob_cfg.get('memory', 300000):
        #         args.append(f'--memory={self.rjob_cfg["memory"]}')
        #     if hasattr(task, 'cpu'):
        #         args.append(f'--cpu={getattr(task, "cpu")}')
        #     elif self.rjob_cfg.get('cpu', 16):
        #         args.append(f'--cpu={self.rjob_cfg["cpu"]}')
        args.append(f'--priority={self.priority}')
        args.append(f'--charged-group=llmfudan_gpu')
        args.append(f'--private-machine=group')

        args.append(f'--mount=gpfs://gpfs1/lvhuijie:/mnt/shared-storage-user/lvhuijie')
        args.append(f'--mount=gpfs://gpfs1/shared-storage-ailab-llmfudan:/mnt/shared-storage-user/shared-storage-ailab-llmfudan')
        args.append(f'--image=registry.h.pjlab.org.cn/ailab-llmfudan/lvhuijie:cuda-124-20251012234943')
        args.append(f'-P 1')
        args.append(f'--host-network=true')
        args.append(f'--auto-restart=false')
        args.append(f'-e DISTRIBUTED_JOB=true')

        args.append(f'--store-host-nvme')
        args.append(f'--custom-resources brainpp.cn/fuse=1')
        args.append(f'--termination-grace-period-seconds 300')
        args.append(f'-e GROUP=llmfudan_gpu')
        args.append(f'--task-type=normal')

        args.append(f'-e WANDB_MODE=offline')
        args.append(f'-e HYDRA_FULL_ERROR=1')

        # Get launch command through task.get_command
        # compatible with template
        tmpl = '{task_cmd}'
        get_cmd = partial(task.get_cmd,
                            template=tmpl)
        entry_cmd = get_cmd()
        entry_cmd = f'bash -exc "source /root/miniconda3/etc/profile.d/conda.sh && cd {pwd} && conda activate verl && {entry_cmd}"'
        


        # Construct complete command
        cmd = f"rjob submit {' '.join(args)} -- {entry_cmd}"
        logger = get_logger()
        logger.info(f'Running command: {cmd}')
        # Log output
        if self.debug:
            out_path = None
        else:
            out_path = task.get_log_path()
            # mmengine.mkdir_or_exist(osp.split(out_path)[0])

        retry = self.retry
        if retry == 0:
            retry = 20
        while retry > 0:
            # Only submit, no polling
            result = subprocess.run(cmd,
                                    shell=True,
                                    text=True,
                                    capture_output=True)
            logger.info(f'Command output: {result.stdout}')
            # logger.info(f'CMD: {cmd}')
            # logger.info(f'Result: {result}')
            task_name = re.findall(r'created rjob_name:\s*(.*)', result.stdout)[-1]      
            if result.stderr:
                logger.error(f'Command error: {result.stderr}')
            logger.info(f'Return code: {result.returncode}')
            if result.returncode == 0:
                break
            retry -= 1
            retry_time = random.randint(5, 60)
            logger.info(f"Remaining {retry} retry, next retry in {retry_time} seconds")
            time.sleep(retry_time)
        if result.returncode != 0:
            # Submit failed, return directly
            return task_name, result.returncode

        # Submit successful, start polling
        status = self._run_task(task_name, out_path)
        # output_paths = task.get_output_paths()
        returncode = 0 if status == 'FINISHED' else 1
        # if self._job_failed(returncode, output_paths):
        #     returncode = 1


        return task_name, returncode

    # def _job_failed(self, return_code: int, output_paths: List[str]) -> bool:
    #     return return_code != 0 or not all(osp.exists(output_path) for output_path in output_paths)



def run_tasks(args):  
    # 这里可以添加你的主要逻辑
    # 例如: process_data(args.input, args.output, args.verbose, args.count)

    task_cfg_path=args.task_cfg_path

    task_cfg = yaml.safe_load(open(task_cfg_path, "r"))
    tasks = [
        TrainingTask(
            bash_path=j["bash_path"],
            name=j.get("name", f"job{i}"),
            nnodes=j["nnodes"],
            n_gpus_per_node=j["n_gpus_per_node"],
            output_dir=j["output_dir"]
        )
        for i, j in enumerate(task_cfg["jobs"])
    ]

    rjobrunner = RJOBRunner(
        max_num_workers=args.max_num_workers,
        retry=args.retry,
        priority=args.priority,
    )

    # volcrunner(tasks)
    rjobrunner(tasks)

if __name__=="__main__":
    args = parse_arguments()
    run_tasks(args)