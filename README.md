# Multi-Task Sentence Transformer

This project implements a multi-task learning framework using a sentence transformer based on BERT. The model is designed to perform both sentence classification and named entity recognition (NER) tasks simultaneously.

## Table of Contents
- [Task 1: Sentence Transformer Implementation](#task-1-sentence-transformer-implementation)
- [Task 2: Multi-Task Learning Implementation](#task-2-multi-task-learning-implementation)
- [Task 3: Training Considerations](#task-3-training-considerations)
- [Task 4: Layer-wise Learning Rates](#task-4-layer-wise-learning-rates)
- [Installation & Usage](#installation--usage)

## Task 1: Sentence Transformer Implementation

### Architecture
- **Base Model**: BERT-base-uncased
- **Output Dimension**: 768
- **Key Components**:
  - Mean pooling layer for fixed-length representations
  - Normalization layer for regularization

### Key Design Choices
1. Mean token pooling instead of CLS token pooling as it generalizes better for sentence embeddings as per original Sentence-BERT paper.

### Sample Embeddings
```
Sentence: This is a test sentence.
Embedding shape: torch.Size([768])
First 5 values: [ 0.06606375 -0.217688   -0.15390255 -0.34627596 -0.01862787]

Sentence: Another sentence with different length.
Embedding shape: torch.Size([768])
First 5 values: [ 0.26112017 -0.42951652  0.02608635  0.11721424 -0.25028437]

Sentence: Machine learning is fascinating.
Embedding shape: torch.Size([768])
First 5 values: [ 0.15958539  0.07248507 -0.14404075  0.04608793  0.42710415]
```

## Task 2: Multi-Task Learning Implementation

### Task A: Sentence Classification
- **Input**: 768-dimensional sentence embedding from pooling layer.
- **Output**: Logit for each class (no need for softmax again as it is being applied internally in loss function).
- **Architecture**: 
  - Dropout layer (p=0.1) for regularization.
  - Linear layer mapping to the number of classes.

### Task B: Named Entity Recognition
- **Input**: Token-level embeddings from last hidden state.
- **Output**: Logits for each token across NER labels
- **Architecture**:
  - Dropout layer for regularization.
  - Linear layer applied to each token embedding.

### Multi-Task Architecture
```
Input Text
    ↓
[BERT Backbone]
    ↓
Sentence Embedding (768d)
    ↓
├── Classification Head
│   └── Classes: [Topic, Sentiment, etc.]
└─- NER Head
    └── Tags: [B-PER, I-PER, B-ORG, etc.]
```

## Task 3: Training Considerations

### Freezing Strategy Analysis

1. **Full Network Freezing**
   - **Use Case**: Pure feature extraction
   - **Advantages**: 
     - Consistent embeddings
     - Fast inference
   - **Disadvantages**: No domain adaptation

2. **Backbone Freezing**
   - **Use Case**: New task adaptation
   - **Advantages**:
     - Prevents catastrophic forgetting
     - Faster training
   - **Disadvantages**: Limited feature adaptation

3. **Task-Head Freezing**
   - **Use Case**: Single task fine-tuning
   - **Advantages**:
     - Maintains performance on frozen tasks
     - Efficient task-specific updates
   - **Disadvantages**: 
     - May lead to the model overfitting to the trainable task. 
     - Changes in the backbone (if unfrozen) could affect the frozen head's performance.

### Transfer Learning Strategy
- **Pre-trained Model**: BERT-base-uncased
  - Comprehensive language understanding
  - Reasonable model size (110M parameters)
  - Extensive pre-training corpus

- **Layer Treatment**:
  - Lower layers (1-4): Frozen (universal features)
  - Middle layers (5-8): Gradual unfreezing
  - Upper layers (9-12): Always trainable

## Task 4: Layer-wise Learning Rates

### Implementation Details
- base_lr=5e-5, lr_decay=0.95
- Base Learning Rate: Set to 5e-5, a common starting point for fine-tuning BERT models.
- Learning Rate Decay: Set to 0.95, so each lower layer has a slightly smaller LR than the one above.
- Upper Layers: Higher LRs to adapt to task-specific features.
- Lower Layers: Smaller LRs to preserve pre-trained weights.
- Task Heads: Assigned the base LR (5e-5) to rapidly learn new patterns.

### Benefits
- Allows different layers to learn at appropriate rates.
- Reduces the risk of catastrophic forgetting by stabilizing lower layers.
- Leads to better generalization and faster convergence.

### Impact in Multitask
- Balancing Tasks: Controls how much the shared backbone adapts, preventing dominance by one task.
- Shared Representations: Ensures shared layers maintain useful features for all tasks.
- Task-specific Adaptation: Allows task heads to learn rapidly without negatively affecting others.


## Installation & Usage

### Prerequisites
- Python 3.8+
- PyTorch 2.0+ (with or without CUDA support)
- CUDA toolkit (optional, for GPU support)
- Docker

### Installation Options

#### 1. CPU-only Installation
```bash
pip install -r requirements.txt
```

#### 2. GPU-enabled Installation
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Then install other requirements
pip install -r requirements.txt
```

#### 3. Docker Installation
```bash
docker build -t sentence-transformer .

# CPU-only
docker run sentence-transformer
# GPU-enabled
docker run --gpus all sentence-transformer
```

### Running Tests
```bash
# Run embedding tests
python -m tests.test_embeddings

```

### Common Issues
1. **CUDA not available**: 
   - Error: `AssertionError: Torch not compiled with CUDA enabled`
   - Solution: Either reinstall PyTorch with CUDA support or use CPU mode

2. **GPU Out of Memory**:
   - Solution: Reduce batch size or use gradient accumulation
   ```python
   # In config/config.py
   BATCH_SIZE = 16  # Reduce if needed
   GRADIENT_ACCUMULATION_STEPS = 4  # Increase for larger effective batch size
   ```

