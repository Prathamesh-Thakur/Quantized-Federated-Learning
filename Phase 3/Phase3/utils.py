import torch

def quantize_weights(state_dict):
    quantized_dict = {}
    scaling_factors = {}

    for key, tensor in state_dict.items():
        max_val = torch.max(torch.abs(tensor))

        if max_val == 0:
            quantized_dict[key] = tensor.to(torch.int8)
            scaling_factors[key] = 1.0
            pass

        scale = max_val / 127.0

        quantized_tensor = torch.round(tensor / scale)
        quantized_tensor = torch.clamp(quantized_tensor, min = -127, max = 128).to(torch.int8)

        quantized_dict[key] = quantized_tensor
        scaling_factors[key] = scale.item()
    
    return quantized_dict, scaling_factors

def dequantize_weights(quantized_dict, scaling_factors):
    dequanitzed_dict = {}

    for key, quantized_tensor in quantized_dict.items():
        scale = scaling_factors[key]

        dequanitzed_tensor = (quantized_tensor).to(torch.float32) * scale

        dequanitzed_dict[key] = dequanitzed_tensor
    
    return dequanitzed_dict

