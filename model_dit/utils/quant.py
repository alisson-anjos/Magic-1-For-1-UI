import os
from pathlib import Path
from safetensors.torch import load_file, save_file
from optimum.quanto import quantization_map, requantize
import json
import torch

def fp8_linear_forward(cls, inputs, original_dtype=torch.bfloat16):
    if inputs.dtype != torch.bfloat16:
        return cls.original_forward(inputs.to(original_dtype))
    else:
        return cls.original_forward(inputs)

def quanto(
    model,
    model_path: str = None,
    save: bool = False
):
    if model_path is not None and os.path.exists(model_path) and os.path.exists(os.path.join(model_path, "quantization_map.json")):
        state_dict = load_file(os.path.join(model_path, "model.safetensors"))
        with open(os.path.join(model_path, "quantization_map.json"), 'r') as f:
            quant_map = json.load(f)
        requantize(model, state_dict, quant_map, device=torch.device('cuda'))
    else:
        from optimum.quanto import freeze, qint8, quantize
        quantize(model, qint8)
        freeze(model)
        if save:
            weight_path = os.path.join(model_path, "model.safetensors")
            json_path = os.path.join(model_path, "quantization_map.json")

            # mkdir
            if not os.path.exists(model_path):
                os.makedirs(model_path, exist_ok=True)

            print("saving model in", weight_path)
            print("saving quantization map in", json_path)

            save_file(model.state_dict(), weight_path)
            with open(json_path, 'w') as f:
                json.dump(quantization_map(model), f)

    for key, layer in model.named_modules():
        if isinstance(layer, torch.nn.Linear):
            original_forward = layer.forward
            setattr(layer, "original_forward", original_forward)
            setattr(layer, "forward", lambda inputs, m=layer: fp8_linear_forward(m, inputs))
    
    return model