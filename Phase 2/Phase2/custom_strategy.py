# Library imports
import torch
from flwr.app import ArrayRecord
from flwr.serverapp.strategy import FedAvg

# Import the function for dequantizing weights
from .utils import dequantize_weights

# Define the class for the custom strategy
class CustomFedAvg(FedAvg):
    """The aggregate_train function fetches the arrays from the clients, so we modify it so that it dequantizes the fetched weights
    first and then passes them to FedAvg"""
    def aggregate_train(self, server_round, replies):
        # Iterate over all client replies
        for reply in replies:
            # Fetch the weightss arrays
            arrays = reply.content["arrays"]
            
            # Convert to torch state dictionary
            torch_array = arrays.to_torch_state_dict()
            
            # Get the factors which were used for quantizing each weight array 
            scaling_factors = reply.content["scaling factors"]

            # Convert the ConfigRecord object back to Python dictionary
            scaling_factors = dict(scaling_factors)

            # Dequantize the weight tensors
            dequantized_tensors = dequantize_weights(torch_array, scaling_factors)

            # Convert the weight tensors to an ArrayRecord for the Server strategy
            returned_array = ArrayRecord(dequantized_tensors)

            # Set the updated array back in the client reply
            reply.content["arrays"] = returned_array

        # Return it to the main function definition
        return super().aggregate_train(server_round, replies)