import numpy as np
import matplotlib.pyplot as plt

phase_1_metrics = {}
phase_2_metrics = {}
phase_3_metrics = {}

rounds = [i for i in range(1, 16)]
server_rounds = [i for i in range(16)]

phase_1_metrics["client_train"] = {
    "payload_size": [4.2656e+01 for _ in range(15)],
    "loss_values": [3.8548e-01, 2.6526e-01, 2.1076e-01, 1.7346e-01, 1.3832e-01, 1.1392e-01, 9.1281e-02, 7.2284e-02, 5.8925e-02, 
                    5.0093e-02, 3.5491e-02, 2.7472e-02, 2.4808e-02, 1.7094e-02, 1.9317e-02]
}

phase_1_metrics["client_test"] = { 
    "accuracies": [8.9304e-01, 9.1520e-01, 9.2775e-01, 9.3321e-01, 9.3206e-01, 9.3490e-01, 9.3713e-01, 9.3667e-01, 9.4006e-01, 
                   9.4052e-01, 9.4221e-01, 9.3844e-01, 9.4152e-01, 9.4206e-01, 9.4306e-01],
    "loss_values": [3.0482e-01, 2.5024e-01, 2.1345e-01, 2.0494e-01, 2.1765e-01, 2.1009e-01, 2.1301e-01, 2.3149e-01, 2.2607e-01, 
                    2.2527e-01, 2.3015e-01, 2.4960e-01, 2.4404e-01, 2.6962e-01, 2.5622e-01]
}

phase_1_metrics["server_test"] = {
    "accuracies": [2.5600e-01, 7.3600e-01, 7.8300e-01, 7.6700e-01, 7.8600e-01, 7.6800e-01, 7.6200e-01, 7.6000e-01, 7.7000e-01, 
                   7.7000e-01, 7.6300e-01, 7.5500e-01, 7.5900e-01, 7.5700e-01, 7.5000e-01, 7.4600e-01],
    "loss_values": [1.5195e+00, 7.1308e-01, 6.3492e-01, 6.5919e-01, 6.8156e-01, 7.7525e-01, 8.1369e-01, 8.9645e-01, 9.3110e-01, 
                    1.0194e+00, 1.0385e+00, 1.1929e+00, 1.1791e+00, 1.2449e+00, 1.2990e+00, 1.3382e+00]
}

phase_2_metrics["client_train"] = {
    "payload_size": [1.0664e+01 for _ in range(15)],
    "loss_values": [3.9131e-01, 2.7389e-01, 2.3012e-01, 2.0063e-01, 1.8040e-01, 1.5758e-01, 1.4062e-01, 1.2350e-01, 1.0926e-01, 
                    9.3006e-02, 8.1968e-02, 7.2545e-02, 6.2643e-02, 5.3584e-02, 4.9976e-02]
}

phase_2_metrics["client_test"] = {
    "accuracies": [8.9497e-01, 9.1374e-01, 9.2144e-01, 9.2782e-01, 9.2852e-01, 9.3013e-01, 9.3544e-01, 9.3783e-01, 9.3890e-01, 
                   9.3829e-01, 9.3544e-01, 9.3644e-01, 9.3706e-01, 9.3829e-01, 9.4075e-01],
    "loss_values": [3.0278e-01, 2.5597e-01, 2.3733e-01, 2.2200e-01, 2.2383e-01, 2.1492e-01, 2.0790e-01, 2.0787e-01, 2.0864e-01, 
                    2.1465e-01, 2.3030e-01, 2.3352e-01, 2.3788e-01, 2.4270e-01, 2.3915e-01]
}

phase_2_metrics["server_test"] = {
    "accuracies": [2.1200e-01, 7.4000e-01, 7.8700e-01, 7.8300e-01, 7.6800e-01, 7.9200e-01, 7.8900e-01, 7.9100e-01, 7.8800e-01, 
                   7.8100e-01, 7.8300e-01, 7.8400e-01, 7.8000e-01, 7.8700e-01, 7.8400e-01, 7.8500e-01],
    "loss_values": [1.4988e+00, 7.0503e-01, 6.3865e-01, 6.5979e-01, 6.8899e-01, 6.7904e-01, 6.8383e-01, 7.2419e-01, 7.7866e-01, 
                    8.2483e-01, 8.9818e-01, 9.3087e-01, 9.6321e-01, 9.4325e-01, 1.0621e+00, 1.1003e+00]
}

phase_3_metrics["client_train"] = {
    "payload_size": [1.0664e+01 for _ in range(15)],
    "loss_values": [3.9111e-01, 3.8230e-01, 3.6767e-01, 3.5605e-01, 3.4258e-01, 3.3139e-01, 3.2284e-01, 3.1290e-01, 3.0733e-01, 
                    3.0041e-01, 2.9635e-01, 2.9056e-01, 2.8925e-01, 2.8454e-01, 2.8567e-01]
}

phase_3_metrics["client_test"] = {
    "accuracies": [2.8447e-01, 3.3056e-01, 4.0620e-01, 5.0823e-01, 6.1888e-01, 7.0922e-01, 7.7062e-01, 8.0817e-01, 8.3372e-01, 
                   8.5280e-01, 8.6280e-01, 8.6911e-01, 8.7458e-01, 8.7096e-01, 8.6273e-01],
    "loss_values": [1.4842e+00, 1.2966e+00, 1.3970e+00, 1.1863e+00, 1.0731e+00, 9.6498e-01, 8.7282e-01, 7.9723e-01, 7.3504e-01, 
                    6.8562e-01, 6.5290e-01, 6.3396e-01, 6.3089e-01, 6.5007e-01, 6.9411e-01]
}

phase_3_metrics["server_test"] = {
    "accuracies": [2.7200e-01, 2.9000e-01, 3.4700e-01, 4.3000e-01, 5.2300e-01, 6.0300e-01, 6.4600e-01, 6.8100e-01, 7.0400e-01, 
                   7.0700e-01, 7.0700e-01, 7.0200e-01, 6.9900e-01, 6.8800e-01, 6.7200e-01, 6.3000e-01],
    "loss_values": [1.4438e+00, 1.3968e+00, 1.3330e+00, 1.2567e+00, 1.1729e+00, 1.0887e+00, 1.0104e+00, 9.4286e-01, 8.8694e-01, 
                    8.4627e-01, 8.2024e-01, 8.0861e-01, 8.1108e-01, 8.2694e-01, 8.6554e-01, 9.3905e-01]
}

plt.figure(figsize=(8, 5))
plt.plot(rounds, np.cumsum(phase_1_metrics["client_train"]["payload_size"]), marker='o')
plt.plot(rounds, np.cumsum(phase_2_metrics["client_train"]["payload_size"]), marker='o')
        
plt.title("Cumulative Payload Size Over Rounds")
plt.xlabel("Round")
plt.ylabel("Cumulative Payload Size (MB)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(["Without Quantization", "With Quantization"])
plt.tight_layout()
plt.savefig("cumulative_payload.png")

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
fig.suptitle("Loss Metrics Across Phases: Training vs. Validation", fontsize=16)

axes[0].plot(rounds, phase_1_metrics["client_train"]["loss_values"], marker='o', label='Client Train Loss')
axes[0].plot(rounds, phase_1_metrics["client_test"]["loss_values"], marker='s', label='Client Eval Loss')
axes[0].plot(server_rounds, phase_1_metrics["server_test"]["loss_values"], marker='^', label='Server Test Loss')

axes[1].plot(rounds, phase_2_metrics["client_train"]["loss_values"], marker='o', label='Client Train Loss')
axes[1].plot(rounds, phase_2_metrics["client_test"]["loss_values"], marker='s', label='Client Eval Loss')
axes[1].plot(server_rounds, phase_2_metrics["server_test"]["loss_values"], marker='^', label='Server Test Loss')

axes[2].plot(rounds, phase_3_metrics["client_train"]["loss_values"], marker='o', label='Client Train Loss')
axes[2].plot(rounds, phase_3_metrics["client_test"]["loss_values"], marker='s', label='Client Eval Loss')
axes[2].plot(server_rounds, phase_3_metrics["server_test"]["loss_values"], marker='^', label='Server Test Loss')

axes[0].set_title("Phase 1: Unquantized FedAvg")
axes[0].set_xlabel("Round")
axes[0].set_ylabel("Cross-Entropy Loss")
axes[0].grid(True, linestyle='--', alpha=0.6)
axes[0].legend()

axes[1].set_title("Phase 2: Quantized FedAvg")
axes[1].set_xlabel("Round")
axes[1].grid(True, linestyle='--', alpha=0.6)
axes[1].legend()

axes[2].set_title("Phase 3: Quantized FedAdam")
axes[2].set_xlabel("Round")
axes[2].grid(True, linestyle='--', alpha=0.6)
axes[2].legend()

plt.tight_layout()
plt.savefig("loss_values.png")

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
fig.suptitle("Accuracy Metrics Across Phases: Client vs. Server", fontsize=16)

axes[0].plot(rounds, phase_1_metrics["client_test"]["accuracies"], marker='s', label='Client Eval Accuracy', color='orange')
axes[0].plot(server_rounds, phase_1_metrics["server_test"]["accuracies"], marker='^', label='Server Test Accuracy', color='green')

axes[1].plot(rounds, phase_2_metrics["client_test"]["accuracies"], marker='s', label='Client Eval Accuracy', color='orange')
axes[1].plot(server_rounds, phase_2_metrics["server_test"]["accuracies"], marker='^', label='Server Test Accuracy', color='green')

axes[2].plot(rounds, phase_3_metrics["client_test"]["accuracies"], marker='s', label='Client Eval Accuracy', color='orange')
axes[2].plot(server_rounds, phase_3_metrics["server_test"]["accuracies"], marker='^', label='Server Test Accuracy', color='green')

axes[0].set_title("Phase 1: Unquantized FedAvg")
axes[0].set_xlabel("Round")
axes[0].set_ylabel("Accuracy")
axes[0].grid(True, linestyle='--', alpha=0.6)
axes[0].legend()

axes[1].set_title("Phase 2: Quantized FedAvg")
axes[1].set_xlabel("Round")
axes[1].grid(True, linestyle='--', alpha=0.6)
axes[1].legend()

axes[2].set_title("Phase 3: Quantized FedAdam")
axes[2].set_xlabel("Round")
axes[2].grid(True, linestyle='--', alpha=0.6)
axes[2].legend()

plt.tight_layout()
plt.savefig("accuracy_values.png")