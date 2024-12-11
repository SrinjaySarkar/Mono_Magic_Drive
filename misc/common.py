import pickle
import importlib
from functools import update_wrapper
from omegaconf import OmegaConf
from omegaconf import DictConfig
from PIL import Image
import numpy as np

import torch
import accelerate
from accelerate.state import AcceleratorState
from accelerate.utils import recursively_apply


def load_module(name):
    p, m = name.rsplit(".", 1)
    # print("name:",name)
    # if name == magicdrive.networks.bbox_embedder.ContinuousBBoxWithTextEmbedding:
    #     name=model.bbox_embedder.ContinuousBBoxWithTextEmbedding
    mod = importlib.import_module(p)
    model_cls = getattr(mod, m)
    return model_cls

def move_to(obj, device, filter=lambda x: True):
    if torch.is_tensor(obj):
        if filter(obj):
            return obj.to(device)
        else:
            return obj
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device, filter)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device, filter))
        return res
    elif obj is None:
        return obj
    else:
        raise TypeError(f"Invalid type {obj.__class__} for move_to.")


# take from torch.ao.quantization.fuse_modules
# Generalization of getattr
def _get_module(model, submodule_key):
    tokens = submodule_key.split('.')
    cur_mod = model
    for s in tokens:
        cur_mod = getattr(cur_mod, s)
    return cur_mod


# Generalization of setattr
def _set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)


def convert_to_fp16(tensor):
    """
    Recursively converts the elements nested list/tuple/dictionary of tensors in FP32 precision to FP16.

    Args:
        tensor (nested list/tuple/dictionary of `torch.Tensor`):
            The data to convert from FP32 to FP16.

    Returns:
        The same data structure as `tensor` with all tensors that were in FP32 precision converted to FP16.
    """

    def _convert_to_fp16(tensor):
        return tensor.half()

    def _is_fp32_tensor(tensor):
        return hasattr(tensor, "dtype") and (
            tensor.dtype == torch.float32
        )

    return recursively_apply(_convert_to_fp16, tensor,
                             test_type=_is_fp32_tensor)


class ConvertOutputsToFp16:
    """
    Decorator to apply to a function outputing tensors (like a model forward pass) that ensures the outputs in FP32
    precision will be convert back to FP16.

    Args:
        model_forward (`Callable`):
            The function which outputs we want to treat.

    Returns:
        The same function as `model_forward` but with converted outputs.
    """

    def __init__(self, model_forward):
        self.model_forward = model_forward
        update_wrapper(self, model_forward)

    def __call__(self, *args, **kwargs):
        return convert_to_fp16(self.model_forward(*args, **kwargs))

    def __getstate__(self):
        raise pickle.PicklingError(
            "Cannot pickle a prepared model with automatic mixed precision, please unwrap the model with `Accelerator.unwrap_model(model)` before pickling it."
        )


convert_outputs_to_fp16 = ConvertOutputsToFp16


def deepspeed_zero_init_disabled_context_manager():
    """
    returns either a context list that includes one that will disable zero.Init or an empty context list
    """
    deepspeed_plugin = AcceleratorState().deepspeed_plugin \
        if accelerate.state.is_initialized() else None
    if deepspeed_plugin is None:
        return []

    return [deepspeed_plugin.zero3_init_context_manager(enable=False)]




def draw_box_on_imgs(cfg, idx, val_input, ori_imgs, transparent_bg=False):
    if transparent_bg:
        in_imgs = [Image.new('RGB', img.size) for img in ori_imgs]
    else:
        in_imgs = ori_imgs
    out_imgs = show_box_on_views(
        OmegaConf.to_container(cfg.dataset.object_classes, resolve=True),
        in_imgs,
        val_input['meta_data']['gt_bboxes_3d'][idx].data,
        val_input['meta_data']['gt_labels_3d'][idx].data.numpy(),
        val_input['meta_data']['lidar2image'][idx].data.numpy(),
        val_input['meta_data']['img_aug_matrix'][idx].data.numpy(),
    )
    if transparent_bg:
        for i in range(len(out_imgs)):
            out_imgs[i].putalpha(Image.fromarray(
                (np.any(np.asarray(out_imgs[i]) > 0, axis=2) * 255).astype(np.uint8)))
    return out_imgs