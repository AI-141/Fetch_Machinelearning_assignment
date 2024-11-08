import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from transformers import get_linear_schedule_with_warmup
from config.config import Config

class LayerwiseLearningRateOptimizer:
    def __init__(self, model):
        self.param_groups = self._group_parameters(model)
    
    def _group_parameters(self, model):
        params = []
        
        # Handle backbone parameters
        if hasattr(model.backbone.backbone, 'encoder'):
            for i, layer in enumerate(model.backbone.backbone.encoder.layer):
                params.append({
                    'params': layer.parameters(),
                    'lr': Config.LEARNING_RATE_BACKBONE * (Config.LEARNING_RATE_DECAY ** (11 - i))
                })
        else:
            # Fallback for other backbones
            params.append({
                'params': model.backbone.parameters(),
                'lr': Config.LEARNING_RATE_BACKBONE
            })

        # Handle task-specific heads
        for head in model.task_heads.values():
            params.append({
                'params': head.parameters(),
                'lr': Config.LEARNING_RATE_HEADS
            })
        
        return params
    
    def get_optimizer(self):
        return AdamW(self.param_groups)
    
class MultiTaskTrainer:
    def __init__(self, model, num_epochs=3):
        self.model = model
        self.num_epochs = num_epochs
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Setup optimizer with layer-wise learning rates
        optimizer_wrapper = LayerwiseLearningRateOptimizer(model)
        self.optimizer = optimizer_wrapper.get_optimizer()
        
        # Scheduler (You can add a scheduler if needed)
        self.scheduler = None 
        
        # Loss functions for different tasks
        self.loss_fns = {
            'classification': CrossEntropyLoss(),
            'ner': CrossEntropyLoss(ignore_index=-100)
        }
        
    def train(self, classification_loader, ner_loader):
        self.model.to(self.device)
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            self.model.train()
            total_loss = 0
            num_batches = 0

            # Train on classification task
            for batch in classification_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                task = batch['task'][0]  # Get the task name from the first item

                self.optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask, task)
                loss = self.loss_fns[task](outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            # Train on NER task
            for batch in ner_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                task = batch['task'][0]  # Get the task name from the first item

                self.optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask, task)
                # Reshape outputs and labels for loss computation
                outputs = outputs.view(-1, outputs.shape[-1])
                labels = labels.view(-1)
                loss = self.loss_fns[task](outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            print(f"Average training loss: {avg_loss:.4f}")
