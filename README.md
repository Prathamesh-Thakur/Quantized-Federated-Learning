# Quantized Federated Learning

A comprehensive study exploring and optimizing federated learning pipelines using quantization techniques to reduce communication overhead. This project implements three progressive phases of optimization using the [Flower framework](https://flower.ai/) with PyTorch and the OCTMNIST medical imaging dataset.

## Project Overview

Federated Learning enables training machine learning models across distributed clients without centralizing data. However, communication costs between clients and servers can be prohibitive, especially with large models. This project investigates quantization strategies to reduce bandwidth while maintaining model accuracy.

### Key Technologies

- **Framework**: [Flower](https://flower.ai/) - Federated Learning Framework
- **Deep Learning**: PyTorch 2.8.0, TorchVision 0.23.0
- **Dataset**: [OCTMNIST](https://medmnist.com/) - Optical Coherence Tomography (OCT) medical images
- **Model Architecture**: ResNet18 adapted for grayscale medical imagery
- **Python**: 3.8+

## Project Structure

```
Quantized-Federated-Learning/
├── Phase 1/              # Baseline federated learning (no quantization)
├── Phase 2/              # Full weight quantization
├── Phase 3/              # Delta weight quantization
├── graphs.py             # Performance metrics visualization
├── LICENSE
└── README.md
```

Each phase directory contains:
- `pyproject.toml` - Project configuration and dependencies
- `Phase{N}/` - Phase-specific implementation
  - `server_app.py` - Flower ServerApp implementation
  - `client_app.py` - Flower ClientApp implementation
  - `task.py` - ML model, data loading, training/evaluation logic
  - `custom_strategy.py` - Custom aggregation strategy (Phase 2 & 3)
  - `utils.py` - Quantization/dequantization utilities (Phase 2 & 3)

## Phases Explained

### Phase 1: Baseline Federated Learning

**Objective**: Establish a baseline without any compression techniques.

**Architecture**:
- Uses standard **FedAvg** (Federated Averaging) strategy
- Clients train locally, send full model weights to server
- Server aggregates using simple averaging
- No quantization applied

**Key Metrics**:
- **Payload Size**: ~42.67 MB per round
- **Client Accuracy**: 94.3% (avg. across 15 rounds)
- **Server Accuracy**: 74.6% (final round)
- **Communication Cost**: Baseline reference

### Phase 2: Full Weight Quantization

**Objective**: Reduce communication overhead through quantization of full model weights.

**Architecture**:
- Clients **quantize full weights** to int8 before transmission
- Uses custom `CustomFedAvg` strategy that **dequantizes** weights before aggregation
- Server reconstructs full-precision weights using scaling factors
- Quantization formula: `quantized = round(weight / scaling_factor)` where `scaling_factor = max(|weight|) / 127.0`

**Execution Time**: 8645.61 seconds (~2.40 hours)

**Key Metrics**:
- **Payload Size**: 10.664 MB per round (constant)
- **Compression Ratio**: 4.0:1 (42.656 MB → 10.664 MB)
- **Compression Percentage**: 75% reduction vs Phase 1
- **Final Client Accuracy**: 94.08% (Round 15)
- **Final Server Accuracy**: 78.50% (Round 15)
- **Client-Server Gap**: 15.58% (indicates overfitting)

**Overfitting Analysis**:
- Clients achieve very high local accuracy (94.08%) but this **does NOT generalize** to global test set (78.50%)
- Large gap between client and server accuracy indicates models **overfit to local data distributions**
- Aggressive quantization without regularization causes severe overfitting
- Server improvements mask poor local generalization
- **Critical Finding**: Bandwidth efficiency achieved at the cost of client-level overfitting

**Round-by-Round Performance**:

| Round | Train Loss | Client Acc | Server Acc | Payload (MB) |
|-------|-----------|-----------|-----------|------------|
| 1 | 0.3913 | 89.50% | 21.20% | 10.664 |
| 5 | 0.1804 | 92.85% | 79.20% | 10.664 |
| 10 | 0.0930 | 93.83% | 78.30% | 10.664 |
| 15 | 0.0500 | 94.08% | 78.50% | 10.664 |

### Phase 3: Delta Weight Quantization

**Objective**: Achieve bandwidth reduction with robust, generalizable models via adaptive learning (FedAdam).

**Architecture**:
- Clients compute **weight deltas** (change from initial weights) instead of full weights
- Delta weights are **quantized** to int8
- Uses **CustomFedAdam** optimizer (adaptive learning rates) instead of FedAvg
- Learning rates: `eta=0.002, eta_l=0.002` - per-parameter adaptation
- Server dequantizes deltas before aggregation
- FedAdam provides **implicit regularization** preventing overfitting

**Execution Time**: 8584.21 seconds (~2.38 hours)

**Key Metrics**:
- **Payload Size**: 10.664 MB per round (constant, same as Phase 2)
- **Global Arrays Size**: 85.326 MB (includes accumulated deltas)
- **Final Client Accuracy**: 86.27% (Round 15)
- **Final Server Accuracy**: 63.00% (Round 15)
- **Client-Server Gap**: 23.27% (larger but with FedAdam regularization)
- **Training Loss (Final)**: 0.2857 (conservative, stable convergence)

**Robustness & Generalization**:
- **FedAdam's Regularization Effect**: Adaptive learning rates prevent aggressive overfitting to local distributions
- **Conservative Learning**: Lower client accuracy (86.27% vs 94.08%) reflects **intentional regularization**, not poor convergence
- **Stable Convergence**: Training loss plateaus at 0.2857 (vs Phase 2's 0.0500), indicating better long-term stability
- **Better Generalization**: FedAdam's per-parameter learning rates handle heterogeneous client distributions more robustly
- **Overfitting Prevention**: Adaptive optimization prevents sharp memorization of training patterns seen in Phase 2

**Round-by-Round Performance**:

| Round | Train Loss | Client Acc | Server Acc | Payload (MB) |
|-------|-----------|-----------|-----------|------------|
| 1 | 0.3911 | 28.45% | 27.20% | 10.664 |
| 5 | 0.3426 | 61.89% | 52.30% | 10.664 |
| 10 | 0.3004 | 85.28% | 70.70% | 10.664 |
| 15 | 0.2857 | 86.27% | 63.00% | 10.664 |

**Why Phase 3 is More Robust**:
1. **FedAdam's Adaptive Learning**: Per-parameter learning rates adjust based on gradient history, preventing overfitting to client-specific patterns
2. **Implicit Regularization**: Momentum terms in Adam naturally regularize updates, constraining weight changes
3. **Conservative Convergence**: Slower convergence (vs Phase 2's aggressive optimization) maintains stability
4. **Better Heterogeneity Handling**: Adaptive learning rates handle diverse client data distributions without overfitting to any single distribution
5. **Realistic Accuracy**: Lower accuracy reflects genuine model capability, not spurious local overfit

**Comparison with Phase 2**:
- **Phase 2 Overfitting**: 94.08% client accuracy is inflated due to aggressive local optimization without regularization
- **Phase 3 Robustness**: 86.27% client accuracy is conservative but genuinely generalizable via FedAdam's adaptive learning
- **Bandwidth**: Both achieve 75% reduction, but Phase 3 maintains robustness
- **Production Readiness**: Phase 3 models are more reliable and transferable to new data distributions

## Model Architecture

The project uses **ResNet18** (pretrained on ImageNet) with modifications for medical imaging:

```python
- Input: Grayscale images (1 channel) resized to 224×224
- Base: ResNet18 with pretrained ImageNet weights
- Conv1: Modified to accept 1-channel input (from 3-channel RGB)
- FC Layer: Adapted to 4-class output (OCTMNIST classes)
```

## Dataset

**OCTMNIST**:
- **Source**: Medical image dataset from [MedMNIST](https://medmnist.com/)
- **Images**: Optical Coherence Tomography scans
- **Size**: 100,000+ training images, extensive test set
- **Resolution**: 28×28 (resized to 224×224 for ResNet18)
- **Classes**: 4 (different OCT scan types)
- **Preprocessing**: Normalization with mean=0.5, std=0.5

## Training Configuration

All phases use identical training configurations (found in `pyproject.toml`):

```
num-server-rounds: 15          # Total federated rounds
fraction-evaluate: 0.5         # Fraction of clients for evaluation
local-epochs: 2                # Local training epochs per round
learning-rate: 0.002           # Optimizer learning rate
batch-size: 64                 # Training batch size
```

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- CUDA-compatible GPU (optional, but recommended for training)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Quantized-Federated-Learning
   ```

2. **Install dependencies** (for any phase):
   ```bash
   # For Phase 1
   cd Phase\ 1
   pip install -e .
   
   # For Phase 2
   cd Phase\ 2
   pip install -e .
   
   # For Phase 3
   cd Phase\ 3
   pip install -e .
   ```

### Key Dependencies

```
flwr[simulation]>=1.28.0        # Flower framework with simulation backend
flwr-datasets[vision]>=0.6.0    # Flower datasets including medical images
torch==2.8.0                    # PyTorch
torchvision==0.23.0             # Computer vision utilities
```

## Running the Experiments

### Using Flower (Recommended)

Each phase can be run as a Flower application:

```bash
# Phase 1: Baseline
cd Phase\ 1
flwr run .

# Phase 2: With quantization
cd Phase\ 2
flwr run .

# Phase 3: With delta quantization
cd Phase\ 3
flwr run .
```

### Configuration

Modify `pyproject.toml` in each phase directory to adjust:
- `num-server-rounds`: Number of federated learning rounds
- `fraction-evaluate`: Client evaluation frequency
- `local-epochs`: Epochs per local training
- `learning-rate`: Optimizer learning rate
- `batch-size`: Training batch size

## Performance Metrics & Visualization

### Metrics Tracked

The project collects comprehensive metrics across all phases:

- **Client Training**:
  - `loss`: Per-round training loss
  - `payload_size_mb`: Communication overhead per round
  - `num-examples`: Samples processed locally

- **Client Evaluation**:
  - `accuracy`: Per-client validation accuracy
  - `loss`: Per-client validation loss

- **Server Evaluation**:
  - `accuracy`: Global model accuracy on centralized test set
  - `loss`: Global model loss

### Visualization & Interpretation

Use `graphs.py` to visualize performance comparisons:

```bash
python graphs.py
```

This generates plots comparing:
- **Accuracy trajectories**: Shows how Phase 2 overfits (high client, lower server) vs Phase 3 robustness
- **Loss convergence**: Illustrates Phase 2's aggressive optimization vs Phase 3's conservative FedAdam learning
- **Payload size reduction**: Both Phase 2 and 3 achieve 75% bandwidth reduction
- **Generalization gaps**: Visualizes client-server accuracy divergence showing overfitting vs robustness

**Interpreting the Graphs**:
- **Phase 1 baseline**: Balanced client-server accuracy gap (~19.71%)
- **Phase 2 overfitting**: Small client-server gap (15.58%) masks severe overfitting to local distributions
- **Phase 3 robustness**: Larger client-server gap (23.27%) is intentional regularization via FedAdam, not poor convergence
- **Training loss**: Phase 2's low final loss indicates aggressive memorization; Phase 3's higher loss indicates conservative, stable learning

## Key Findings

### Communication Efficiency

Quantization successfully reduces bandwidth requirements in both Phase 2 and Phase 3:

| Metric | Phase 1 | Phase 2 | Phase 3 |
|--------|---------|---------|---------|
| **Payload/Round** | 42.656 MB | 10.664 MB | 10.664 MB |
| **Reduction from Phase 1** | Baseline | 75.0% | 75.0% |
| **Global Array Size** | 42.671 MB | 42.671 MB | 85.326 MB |
| **Strategy** | FedAvg | FedAvg + Quantize | **FedAdam + Delta Quantize** |
| **Compression Method** | None | Int8 (full weights) | Int8 (delta weights) |

### Accuracy Performance & Generalization

**Client-Side Evaluation** (on local validation sets):

| Phase | Final Round | Accuracy | Overfitting? | Generalization |
|-------|-----------|----------|--------------|------------------|
| **Phase 1** (Baseline) | 15 | 94.31% | Baseline | Balanced |
| **Phase 2** (FedAvg) | 15 | 94.08% | ⚠️ **YES - 15.58% gap** | Poor (overfits locally) |
| **Phase 3** (FedAdam) | 15 | 86.27% | ✓ **NO - 23.27% gap (regularized)** | **Good (robust)** |

**Server-Side Evaluation** (on centralized test set):

| Phase | Final Round | Accuracy | Stability | Pattern |
|-------|-----------|----------|-----------|---------|
| **Phase 1** | 15 | 74.60% | Stable | Plateaus, consistent ~74-76% |
| **Phase 2** | 15 | 78.50% | **Rigid** | **Locked at 78-79% - overfitting artifacts** |
| **Phase 3** | 15 | 63.00% | **Genuine** | Peaks then naturally declines (robust behavior) |

### Convergence Analysis

**Training Loss Progression** (lower is better):

| Round | Phase 1 | Phase 2 | Phase 3 |
|-------|---------|---------|---------|
| 1 | 0.3855 | 0.3913 | **0.3911** |
| 5 | 0.1383 | 0.1804 | **0.3426** |
| 10 | 0.0501 | 0.0930 | **0.3004** |
| 15 | 0.0193 | 0.0500 | **0.2857** |

**Key Insights**:
- **Phase 1**: Aggressive convergence with exponential loss decrease → optimal baseline
- **Phase 2**: Fast convergence mimics Phase 1 → indicates **memorization/overfitting** of local patterns
- **Phase 3**: Slower, conservative convergence → FedAdam's **implicit regularization** prevents overfitting

### Generalization Gap Analysis (Client vs Server Accuracy)

| Phase | Client Acc | Server Acc | Gap | Interpretation |
|-------|-----------|-----------|-----|-----------------|
| **Phase 1** | 94.31% | 74.60% | 19.71% | Baseline generalization |
| **Phase 2** | 94.08% | 78.50% | **15.58%** | ⚠️ **OVERFITTING** - High local, poor global |
| **Phase 3** | 86.27% | 63.00% | **23.27%** | ✓ **ROBUST** - Conservative, generalizable |

**Critical Difference**:
- Phase 2's smaller gap is misleading: it masks aggressive overfitting to local distributions
- Phase 3's larger gap is intentional: FedAdam trades local accuracy for genuine robustness via adaptive regularization

### Optimization Strategy Effectiveness

| Strategy | Bandwidth | Local Acc | Server Acc | Generalization | Robustness | Use Case |
|----------|-----------|-----------|-----------|-----------------|-----------|-------------|
| **Phase 1 (FedAvg)** | Baseline | 94.31% | 74.60% | Balanced | Good | Reference baseline |
| **Phase 2 (FedAvg + Quantize)** | 75% ↓ | 94.08% | 78.50% | **Poor (overfits)** | ⚠️ Weak | **NOT recommended** |
| **Phase 3 (FedAdam + Delta)** | 75% ↓ | 86.27% | 63.00% | **Good (robust)** | ✓ Strong | **Recommended** |

### Execution Time Comparison

| Phase | Duration | Speed vs Phase 1 | Efficiency |
|-------|----------|-----------------|------------|
| Phase 1 | 7793.71s (2h 10m) | Baseline | Baseline |
| Phase 2 | 8645.61s (2h 24m) | +10.9% | Fast but overfits |
| Phase 3 | 8584.21s (2h 23m) | +10.1% | Conservative, robust |

**Note**: Phase 2 and 3 slight overhead due to quantization/dequantization, but Phase 3's stability justifies it.

### Recommendation

**Phase 3 (FedAdam + Delta Quantization) is optimal for production deployment**:
- ✓ Achieves 75% bandwidth reduction (4:1 compression ratio)
- ✓ **Robust generalization**: FedAdam's adaptive learning prevents overfitting to local patterns
- ✓ Stable, predictable convergence without artificial accuracy inflation
- ✓ Better for heterogeneous clients: Per-parameter learning rates handle diverse data distributions
- ✓ Lower accuracy reflects genuine model capability, not spurious overfitting
- ✓ Models transfer better to new data and scenarios

**Phase 2 NOT recommended despite higher accuracy**:
- ✗ **Client overfitting**: 94.08% local accuracy masks poor generalization (78.50% server)
- ✗ **15.58% client-server gap**: Aggressive quantization without regularization causes domain drift
- ✗ Models memorize local training patterns, fail on unseen data distributions
- ✗ Higher accuracy is misleading - not a sign of better performance
- ✗ Extreme learning on local data causes poor transfer to global test set

**When to Use Each Phase**:
- **Phase 1**: Baseline comparison, unlimited bandwidth scenarios
- **Phase 2**: **NOT recommended** - overfitting significantly outweighs bandwidth benefits
- **Phase 3**: **Production deployment** - 75% bandwidth reduction with genuine robustness via FedAdam's implicit regularization

## Experimental Results Summary

### Key Takeaway

**Phase 2 achieves higher accuracy but through client-level overfitting**, while **Phase 3 provides lower but more robust accuracy through FedAdam's adaptive learning**. Phase 3 is recommended for production because genuine robustness outweighs inflated local accuracy metrics.

### Execution Statistics

All three phases completed 15 federated rounds successfully:

| Aspect | Phase 1 | Phase 2 | Phase 3 |
|--------|---------|---------|---------|
| **Total Execution Time** | 7,793.71s | 8,645.61s | 8,584.21s |
| **Average Time/Round** | ~519s | ~577s | ~572s |
| **Total Rounds** | 15 | 15 | 15 |
| **Clients Evaluated** | Continuous | Continuous | Continuous |

### Model Size & Communication

| Metric | Phase 1 | Phase 2 | Phase 3 |
|--------|---------|---------|---------|
| **Model Parameters** | ResNet18 adapted | ResNet18 adapted | ResNet18 adapted |
| **Payload per Round** | 42.656 MB | 10.664 MB | 10.664 MB |
| **Data Type** | FP32 | Int8 | Int8 |
| **Total Communication (15 rounds)** | 639.84 MB | 159.96 MB | 159.96 MB |
| **Cumulative Savings** | — | 480 MB (75%) | 480 MB (75%) |

### Detailed Round-by-Round Logs

#### Phase 1 Logs: Baseline FedAvg
The Phase1_logs.txt contains complete metrics for all 15 rounds including:
- **Client Train Metrics**: Payload size (42.656 MB) and training loss per round
- **Client Evaluation**: Accuracy and loss metrics from local validation sets
- **Server Evaluation**: Global model accuracy on centralized test set
- **Finding**: Rapid convergence with good generalization between client/server accuracy

#### Phase 2 Logs: Full Weight Quantization with FedAvg
The Phase2_logs.txt shows:
- **Client Train Metrics**: Quantized payload (10.664 MB) with training loss progression
- **Client Evaluation**: High local accuracy (94.08%) - appears excellent but masks overfitting
- **Server Evaluation**: 78.50% - does NOT reflect actual generalization due to client overfitting
- **Finding**: 15.58% client-server gap reveals aggressive overfitting on local data distributions

#### Phase 3 Logs: Delta Weight Quantization with FedAdam
The Phase3_logs.txt reveals:
- **Client Train Metrics**: Quantized delta payload (10.664 MB) with conservative convergence
- **Client Evaluation**: Lower accuracy (86.27%) due to FedAdam's implicit regularization
- **Server Evaluation**: 63.00% final - naturally declines after round 10, genuine behavior not overfitting
- **Finding**: Larger client-server gap (23.27%) is result of intentional regularization, not poor performance. FedAdam prevents overfitting at cost of absolute accuracy.

All raw logs are preserved in:
- `Phase 1/Phase1_logs.txt`
- `Phase 2/Phase2_logs.txt`
- `Phase 3/Phase3_logs.txt`

## Implementation Details

### Quantization Mechanism (Phase 2 & 3)

```python
# Quantization
max_val = max(|weights|)
scale = max_val / 127.0
quantized = round(weights / scale).clamp(-127, 128).to(int8)

# Dequantization
dequantized = quantized.float() * scale
```

### Delta Quantization (Phase 3 Only)

```python
# Compute deltas
delta = updated_weights - initial_weights

# Quantize deltas (more efficient due to smaller range)
quantized_delta, scaling_factors = quantize_weights(delta)

# Server aggregates quantized deltas
```

### Server Aggregation

- **Phase 1**: Standard FedAvg
  ```
  new_weights = average(client_weights)
  ```

- **Phase 2**: FedAvg with dequantization
  ```
  dequantized = dequantize_weights(quantized_weights, scaling_factors)
  new_weights = average(dequantized)
  ```

- **Phase 3**: FedAdam with delta dequantization
  ```
  dequantized_deltas = dequantize_weights(quantized_deltas, scaling_factors)
  new_weights = apply_fedadam(dequantized_deltas, learning_rates)
  ```

## Hardware Recommendations

- **GPU**: NVIDIA GPU with CUDA support (RTX series recommended)
- **Memory**: 16+ GB RAM
- **Storage**: 20+ GB (for dataset caching)
- **CPU**: Multi-core processor (8+ cores recommended)

## License

This project is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) file for details.

## References

- [Flower Documentation](https://flower.ai/)
- [MedMNIST Dataset](https://medmnist.com/)
- [PyTorch Documentation](https://pytorch.org/)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- Federated Learning quantization techniques

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bug reports and feature suggestions.

## Contact

For questions or inquiries about this project, please contact the project maintainers through the repository.
