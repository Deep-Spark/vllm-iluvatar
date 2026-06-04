# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import torch
import vllm.envs as envs
from filelock import FileLock
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


class StaticQuantManager:
    def __init__(
        self,
        layer_id: int,
        shape: tuple,
        dtype: torch.dtype,
        total_layer_num: int,
        device: str = None,
        tp_size: int = None,
        tp_rank: int = None,
        file_save_path: str = None,
        save_step: int = 100,
        info_step: int = 100,
    ):
        # update parament
        if tp_size is None:
            tp_size = get_tensor_model_parallel_world_size()
        if tp_rank is None:
            tp_rank = get_tensor_model_parallel_rank()
        if file_save_path is None:
            file_save_path = envs.VLLM_ATTN_STATIC_QUANT_SCALE_FILE_PATH
        if device is None:
            device = "cuda"

        # check parament
        if file_save_path in [None, ""]:
            self.disable = True
            return

        para_dir = os.path.dirname(file_save_path)
        assert os.path.exists(
            para_dir
        ), f"StaticQuantManager workdir {para_dir} not exist!"
        self.disable = os.path.exists(file_save_path)
        if self.disable:
            return

        assert layer_id is not None
        assert total_layer_num is not None

        world_rank = torch.distributed.get_rank()
        work_dir = os.path.join(para_dir, "StaticQuantManagerWorkdir")
        self.operator = world_rank == 0 and layer_id == 0
        if not os.path.exists(work_dir):
            if self.operator:
                logger.debug(f"StaticQuantManager Creat {work_dir}!")
                os.mkdir(work_dir)
        self.file_save_path = file_save_path
        self.work_dir = work_dir

        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.world_rank = world_rank
        self.layer_id = layer_id
        self.total_layer_num = total_layer_num
        self.save_step = save_step
        self.info_step = info_step
        self.update_count = 0
        self.save_flag = False
        self.scales = torch.zeros(shape, dtype=dtype, device=device)
        logger.debug(
            f"StaticQuantManager info: world_rank:{self.world_rank} tp_rank:{self.tp_rank} layer_id:{self.layer_id} scale shape:{shape}  self.scales:{self.scales.device}"
        )

    def check_enable(self):
        return not self.disable

    def update_data(self, data):
        if self.disable:
            return

        self.scales = torch.max(data, self.scales)

        # save file
        self.update_count += 1
        if self.update_count % self.info_step == 0 and self.operator:
            logger.info(f"StaticQuantManager run update_data {self.update_count} step")

        if self.update_count % self.save_step == 0:
            # step1: save to disk
            save_file_path = os.path.join(
                self.work_dir, f"{self.layer_id}_{self.tp_rank}.pt"
            )
            lock_file_path = os.path.join(
                self.work_dir, f"{self.layer_id}_{self.tp_rank}.lock"
            )
            lock = FileLock(lock_file_path)
            cpu_data = self.scales.cpu()
            with lock:
                torch.save(cpu_data, save_file_path)

            # step2: merge and save
            if self.save_flag and self.operator:
                save_dict = {}
                for idx in range(self.total_layer_num):
                    tp_datas = []
                    for tp_rank in range(self.tp_size):
                        load_file = os.path.join(self.work_dir, f"{idx}_{tp_rank}.pt")
                        lock_file_path = os.path.join(
                            self.work_dir, f"{idx}_{tp_rank}.lock"
                        )
                        lock = FileLock(lock_file_path)
                        with lock:
                            cur_data = torch.load(load_file)
                            tp_datas.append(cur_data)

                    layer_data = torch.concat(tp_datas)
                    save_dict[f"layer_{idx}"] = layer_data

                torch.save(save_dict, self.file_save_path)
                logger.info(
                    f"StaticQuantManager save to {self.file_save_path} with {self.update_count} step"
                )
            self.save_flag = True
