# 提交训练任务到火山集群
# 支持 checkpoint 断点续训
import os
import os.path as osp
import re
import subprocess
import time
import glob
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

# 环境变量
LARK_BOT_URL = os.environ['LARK_BOT_URL']
VOLC_QUEUE_NAME = os.environ['VOLC_QUEUE_NAME']

CONDA_ENV_NAME = os.environ['VERL_CONDA_ENV']
VOLC_CFG_PATH = os.environ['VERL_VOLC_CFG_PATH']

def get_logger(
    name: str = "VOLCRunner",
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
        description='配置verl提交到火山云集群的参数'
    )

    parser.add_argument(
        '--bashrc_path',  # 长短参数名
        type=str,          # 参数类型
        default='/root/.bashrc',     # 必需参数
        help='需要激活的 bashrc 路径'
    )
    
    parser.add_argument(
        '--preemptible',
        action='store_true',  # 标志参数，不需要值
        help='是否使用闲时资源'
    )
    
    parser.add_argument(
        '--priority',
        type=int,
        default=4,
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

    parser.add_argument(
        '--keep_tmp_file',
        action='store_true',  # 标志参数，不需要值
        help='是否保留临时配置文件'
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
        os.makedirs(self.output_dir, exist_ok=True)

        log_path = os.path.join(self.output_dir, f'{self.name}.log')

        return log_path


class VOLCRunner:
    def __init__(self,
                 queue_name: str,
                 bashrc_path: str,
                 conda_env_name: str,
                 volcano_config_path: str,
                 preemptible: bool,
                 priority: Optional[int],
                 max_num_workers: int,
                 retry: int,
                 lark_bot_url: str,
                 keep_tmp_file: bool,
                 debug: bool = False,
        ):        
        if lark_bot_url:
            self.lark_reporter = LarkReporter(lark_bot_url)
        else:
            self.lark_reporter = None
        
        self.max_num_workers = max_num_workers
        self.retry = retry

        self.queue_name = queue_name
        self.bashrc_path = bashrc_path
        self.conda_env_name = conda_env_name
        self.volcano_config_path = volcano_config_path
        self.preemptible = preemptible
        self.priority = priority
        self.keep_tmp_file = keep_tmp_file

        self.debug = debug

    def __call__(self, tasks: List[TrainingTask]):
        """Launch multiple tasks and summarize the results.

        Args:
            tasks (list[TrainingTask]): A list of task configs, usually generated by
                Partitioner.
        """
        status = self.launch(tasks)
        status_list = list(status)  # change into list format
        self.summarize(status_list)
    
    def launch(self, tasks: List[TrainingTask]) -> List[Tuple[str, int]]:
        """Launch multiple tasks.

        Args:
            tasks (list[TrainingTask]): A list of task configs, usually generated by
                Partitioner.

        Returns:
            list[tuple[str, int]]: A list of (task name, exit code).
        """

        if not self.debug:
            # 多进程形式以此执行每个 task
            status = track_parallel_progress(self._launch,
                                             tasks,
                                             nproc=self.max_num_workers,
                                             keep_order=False)
        else:
            status = [self._launch(task, random_sleep=False) for task in tasks]
        return status

    def _launch(self, task: TrainingTask, random_sleep: bool = True):
        num_gpus = task.n_gpus_per_node
        num_nodes = task.nnodes
        task_name = task.name

        # Build up VCC command
        pwd = os.getcwd()
        # Dump task config to file
        mmengine.mkdir_or_exist('lhj_scripts/tmp/')
        # Using uuid to avoid filename conflict
        import uuid
        uuid_str = str(uuid.uuid4())

        # 临时 volc 配置文件保存路径，在提交任务时，使用该路径
        volc_cfg_file = f'{pwd}/lhj_scripts/tmp/{uuid_str}_volc_cfg.yaml'
        volc_cfg = self._generate_tmp_volc_cfg(num_gpus, num_nodes)
        with open(volc_cfg_file, 'w', encoding="utf-8") as fp:
            yaml.dump(volc_cfg, fp, sort_keys=False)
        try:
            shell_cmd = "{task_cmd}"
            
            # 根据配置得到火山集群任务提交命令，其中shell_cmd留有了'{task_cmd}'，供具体任务命令替换
            tmpl = ('volc ml_task submit'
                    f" --conf '{volc_cfg_file}'"
                    f" --entrypoint \"{shell_cmd}\""
                    f" --task_name {task_name}"
                    f" --resource_queue_name {self.queue_name}")
            if self.preemptible:
                tmpl += ' --preemptible'
            if self.priority is not None:
                tmpl += f' --priority {self.priority}'

            get_cmd = partial(task.get_cmd,
                              template=tmpl)
            cmd = get_cmd()

            logger = get_logger()
            logger.info(f'Running Command: {cmd}')

            log_path = task.get_log_path()

            retry = self.retry

            while True:
                task_status, returncode = self._run_task(cmd,
                                                         log_path,
                                                         poll_interval=20)
                # output_paths = task.get_output_paths()
                if not (self._job_failed(task_status)) \
                        or retry <= 0:
                    break
                retry -= 1
                if random_sleep:
                    # time.sleep(random.randint(0, 10))
                    time.sleep(60)
        finally:
            # Clean up
            if not self.keep_tmp_file:
                os.remove(volc_cfg_file)
            else:
                pass
        logger.info(f'Task Status: {task_status}; ReturnCode: {returncode}')
        return task_name, task_status, returncode

    def _run_task(self, cmd, log_path, poll_interval):
        logger = get_logger()
        result = subprocess.run(cmd,
                                shell=True,
                                text=True,
                                capture_output=True)

        # logger.info(result)
        logger.info(f'Command output: {result.stdout}')
        if result.stderr:
            logger.error(f'Command ERROR: {result.stderr}')
        logger.info(f'Return code: {result.returncode}')

        pattern = r'(?<=task_id=).*(?=\n\n)'
        match = re.search(pattern, result.stdout)
        if match:
            task_id = match.group()
            logger.info(task_id)
            ask_cmd = f'volc ml_task get --id {task_id} --output json ' + \
                        '--format Status'
            log_cmd = f'volc ml_task logs --task {task_id} --instance head_0'
            while True:
                task_status = os.popen(ask_cmd).read()
                pattern = r'(?<=\[{"Status":").*(?="}\])'
                match = re.search(pattern, task_status)
                if self.debug:
                    print(task_status)
                logs = os.popen(log_cmd).read()
                with open(log_path, 'w', encoding='utf-8') as f:
                    f.write(logs)
                if match:
                    task_status = match.group()
                    if task_status in [
                            'Success', 'Failed', 'Cancelled', 'Exception',
                            'Killing', 'SuccessHolding', 'FailedHolding',
                            'Killed'
                    ]:
                        with open(log_path, 'a', encoding='utf-8') as f:
                            f.write(f"\n{task_status}")
                        break
                # If pattern not found or command failed, sleep and retry
                time.sleep(poll_interval)
        else:
            task_status = 'Exception'

        return task_status, result.returncode

    def _job_failed(self, task_status: str) -> bool:
        return task_status != 'Success'

    def _generate_tmp_volc_cfg(self, num_gpus, num_nodes):
        config_path = self.volcano_config_path
        with open(config_path) as fp:
            volc_cfg = yaml.safe_load(fp)
        if num_gpus <= 0:
            flavor = 'ml.r3i.16xlarge'
        elif num_gpus == 1:
            flavor = 'ml.pni2l.3xlarge'
        elif num_gpus == 2:
            flavor = 'ml.pni2l.7xlarge'
        elif num_gpus <= 4:
            flavor = 'ml.pni2l.14xlarge'
        elif num_gpus <= 8:
            flavor = 'ml.hpcpni2l.28xlarge'
        else:
            raise NotImplementedError

        for i in range(len(volc_cfg['TaskRoleSpecs'])):
            if volc_cfg['TaskRoleSpecs'][i]['RoleName'] == 'head':
                volc_cfg['TaskRoleSpecs'][i]['Flavor'] = flavor
                volc_cfg['TaskRoleSpecs'][i]['RoleReplicas'] = 1
            if volc_cfg['TaskRoleSpecs'][i]['RoleName'] == 'worker':
                volc_cfg['TaskRoleSpecs'][i]['Flavor'] = flavor
                volc_cfg['TaskRoleSpecs'][i]['RoleReplicas'] = num_nodes - 1
        return volc_cfg

    def summarize(self, status: List[Tuple[str, int]]) -> None:
        """Summarize the results of the tasks.

        Args:
            status (list[tuple[str, int]]): A list of (task name, exit code).
        """

        failed_logs = []
        for _task, status, code in status:
            if status != "Success":
                get_logger().error(f'{_task} failed with status {status}')
                failed_logs.append(_task)
        if self.lark_reporter:
            num_succeeded = len(status) - len(failed_logs)
            if len(failed_logs) > 0:
                content = 'Tasks Failed!!!\n'
                content += f'{num_succeeded} tasks succeeded, '
                content += f'{len(failed_logs)} tasks failed.'
                self.lark_reporter.post(title=f'Bad news: {len(failed_logs)} '
                                        'failed.',
                                        content=content)
            else:
                content = 'Tasks Success!!!\n'
                content += f'{num_succeeded} tasks succeeded.\n'
                self.lark_reporter.post(title='Great news: all tasks '
                                        'finished!',
                                        content=content)


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

    volcrunner = VOLCRunner(lark_bot_url=LARK_BOT_URL,
                            max_num_workers=args.max_num_workers,
                            retry=args.retry,
                            queue_name=VOLC_QUEUE_NAME,
                            bashrc_path=args.bashrc_path,
                            conda_env_name=CONDA_ENV_NAME,
                            volcano_config_path=VOLC_CFG_PATH,
                            preemptible=args.preemptible,
                            priority=args.priority,
                            keep_tmp_file=args.keep_tmp_file,)
    volcrunner(tasks)

if __name__=="__main__":
    args = parse_arguments()
    run_tasks(args)

    # log_path = "/fs-computility/llm_fudan/lvhuijie/my_git_repo/verl/checkpoints/test/qwen3-0.6b/test/GRPO/test/logs/qwen3-0_6b_gsm8k_GRPO_test.log"
    # task_status = "Success"
    # with open(log_path, 'a', encoding='utf-8') as f:
    #     f.write(f"\n{task_status}")