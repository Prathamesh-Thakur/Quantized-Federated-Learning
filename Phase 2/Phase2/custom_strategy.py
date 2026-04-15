import torch
from flwr.app import ArrayRecord
from flwr.serverapp.strategy import FedAvg

from .utils import dequantize_weights

class CustomFedAvg(FedAvg):
    def aggregate_train(self, server_round, replies):
        for reply in replies:
            arrays = reply.content["arrays"]
            torch_array = arrays.to_torch_state_dict()
            
            scaling_factors = reply.content["scaling factors"]

            scaling_factors = dict(scaling_factors)

            dequantized_tensors = dequantize_weights(torch_array, scaling_factors)

            returned_array = ArrayRecord(dequantized_tensors)

            reply.content["arrays"] = returned_array

        return super().aggregate_train(server_round, replies)