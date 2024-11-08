# train.py

import torch
from models.sentence_transformer import SentenceTransformer
from models.task_heads import MultiTaskModel, SentenceClassificationHead, NamedEntityRecognitionHead
from trainers.trainer import MultiTaskTrainer
from config.config import Config
from utils.data_utils import get_data_loaders

def main():
    # Initialize model
    backbone = SentenceTransformer(model_name=Config.MODEL_NAME)
    
    # Initialize task heads
    task_heads = {
        'classification': SentenceClassificationHead(
            input_dim=Config.HIDDEN_SIZE,
            num_classes=Config.NUM_CLASSES_TASK_A
        ),
        'ner': NamedEntityRecognitionHead(
            input_dim=Config.HIDDEN_SIZE,
            num_labels=Config.NUM_LABELS_TASK_B
        )
    }
    
    # Create multi-task model
    model = MultiTaskModel(backbone, task_heads)
    
    # Get data loaders
    classification_loader, ner_loader = get_data_loaders(batch_size=Config.BATCH_SIZE)
    
    # Initialize trainer
    trainer = MultiTaskTrainer(
        model=model,
        num_epochs=Config.NUM_EPOCHS
    )
    
    # Train the model
    trainer.train(classification_loader, ner_loader)
    
if __name__ == "__main__":
    main()
