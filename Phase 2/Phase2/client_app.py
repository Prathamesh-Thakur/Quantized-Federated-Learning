# Import libraries
import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict, ConfigRecord
from flwr.clientapp import ClientApp

# Import model, data loader, training and testing functions
from Phase2.task import Net, load_data
from Phase2.task import test as test_fn
from Phase2.task import train as train_fn

# Import quantization function
import copy
from .utils import quantize_weights

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    trainloader, _ = load_data(partition_id, num_partitions, batch_size)

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )
    updated_weights = copy.deepcopy(model.state_dict())
    
    # Transfer the model weights to the CPU
    updated_weights_cpu = {k: v.cpu() for k, v in updated_weights.items()}

    # Quantize the updated weights
    quantized_weights, scaling_factors = quantize_weights(updated_weights_cpu)

    # Calculate total memory usage in bytes
    total_bytes = sum([tensor.nelement() * tensor.element_size() for tensor in quantized_weights.values()])

    # Convert to MB
    total_bytes_in_mb = total_bytes / (1024 * 1024)

    model_record = ArrayRecord(quantized_weights)
    
    # Construct metrics dictionary
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
        "payload_size_mb": total_bytes_in_mb
    }
    metric_record = MetricRecord(metrics)

    # Construct and return reply Message
    content = RecordDict({"arrays": model_record, "metrics": metric_record, "scaling factors": ConfigRecord(scaling_factors)})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    _, valloader = load_data(partition_id, num_partitions, batch_size)

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
