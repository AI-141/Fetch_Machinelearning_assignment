class Config:
    MODEL_NAME = 'bert-base-uncased'
    HIDDEN_SIZE = 768  # Hidden size for BERT-base
    NUM_CLASSES_TASK_A = 6  # Number of classes in the classification task
    NUM_LABELS_TASK_B = 7  # Number of labels in the NER task
    BATCH_SIZE = 2
    NUM_EPOCHS = 3
    LEARNING_RATE_BACKBONE = 5e-5
    LEARNING_RATE_HEADS = 1e-4
    LEARNING_RATE_DECAY = 0.95  # For layer-wise learning rates
    CLASSIFICATION_CLASSES = [0, 1, 2, 3, 4, 5]
    NER_LABELS = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']

    