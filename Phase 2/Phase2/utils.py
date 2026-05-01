# Import required library
import torch

def quantize_weights(state_dict):
    """Function to quantize weight arrays"""
    quantized_dict = {}
    scaling_factors = {}

    # Iterate over the layer weights
    for key, tensor in state_dict.items():
        # Fetch the absolute maximum value of the specific layer's weight tensor
        max_val = torch.max(torch.abs(tensor))

        # If zero, keep weights unchanged
        if max_val == 0:
            quantized_dict[key] = tensor.to(torch.int8)
            scaling_factors[key] = 1.0
            pass

        # Calculate scaling factor by dividing the maximum value with the maximum possible integer value
        scale = max_val / 127.0

        # Quantize the tensor
        quantized_tensor = torch.round(tensor / scale)
        
        # To avoid outliers, clamp the output and reduce the precision to integer
        quantized_tensor = torch.clamp(quantized_tensor, min = -127, max = 128).to(torch.int8)

        quantized_dict[key] = quantized_tensor
        scaling_factors[key] = scale.item()
    
    return quantized_dict, scaling_factors


def dequantize_weights(quantized_dict, scaling_factors):
    """Function to dequanitze the weights"""
    dequanitzed_dict = {}

    # Iterate over the weight tensors for each layer
    for key, quantized_tensor in quantized_dict.items():
        # Fetch the scaling factor for the current layer
        scale = scaling_factors[key]

        # Dequantize by increasing precision to float and multiply by scaling factor to get original weight value
        dequanitzed_tensor = (quantized_tensor).to(torch.float32) * scale

        dequanitzed_dict[key] = dequanitzed_tensor
    
    return dequanitzed_dict

