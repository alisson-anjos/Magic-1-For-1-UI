import time
import torch
import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

import lightning as L
from lightning.pytorch.utilities import rank_zero_info, rank_zero_only

from deepspeed.accelerator import get_accelerator

class CacheCleanupCallback(L.Callback): # this is quite important to avoid memory reallocatioon for different length of tokens during training.
    def __init__(self):
        super().__init__()  # Call parent's init
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx > 0 and (batch_idx+1) % 5 == 0:  # after every 4 batches, it will clear the cache
            torch.cuda.empty_cache()
            get_accelerator().empty_cache()
    
    def on_train_epoch_end(self, trainer, pl_module):
        # Clear cache between epochs
        torch.cuda.empty_cache()
        get_accelerator().empty_cache()


class BasicCallback(L.Callback):
    def __init__(self):
        super().__init__()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        real_step = trainer.global_step  # + config.epoch_begin * config.epoch_steps
        # TODO
        if trainer.is_global_zero:  # logging
            t_now = time.time_ns()
            kt_s = 0
            try:
                t_cost = (t_now - trainer.my_time_ns) / 1e9
                self.log("REAL it/s", 1.0 / t_cost, prog_bar=True, on_step=True)
            except:
                pass

            for param_group in trainer.optimizers[0].param_groups:
                lr = param_group["lr"]
                break

            trainer.my_lr = lr

            trainer.my_time_ns = t_now
            self.log("lr", lr, prog_bar=True, on_step=True)
            self.log("step", int(real_step), prog_bar=False, on_step=True)
            self.log("loss", outputs["loss"], prog_bar=True, on_step=True)
        
        if hasattr(self, "model_ema"):
            self.model_ema.forward(self.model)
            print("Update EMA model",trainer.global_step)
        