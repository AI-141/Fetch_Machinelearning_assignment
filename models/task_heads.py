import torch.nn as nn

class SentenceClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(input_dim, num_classes)
    
    def forward(self, sentence_embeddings):
        x = self.dropout(sentence_embeddings)
        logits = self.classifier(x)
        return logits

class NamedEntityRecognitionHead(nn.Module):
    def __init__(self, input_dim, num_labels):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(input_dim, num_labels)
    
    def forward(self, token_embeddings):
        x = self.dropout(token_embeddings)
        logits = self.classifier(x)
        return logits

class MultiTaskModel(nn.Module):
    def __init__(self, backbone, task_heads):
        super().__init__()
        self.backbone = backbone  # This is the SentenceTransformer with pooling
        self.task_heads = nn.ModuleDict(task_heads)
    
    def forward(self, input_ids, attention_mask, task):
        if task == 'classification':
            # Get sentence embeddings from the backbone
            sentence_embeddings = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            # Pass through the classification head
            logits = self.task_heads[task](sentence_embeddings)
        elif task == 'ner':
            # Get token embeddings (last_hidden_state) from the backbone's backbone
            outputs = self.backbone.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            token_embeddings = outputs.last_hidden_state 
            # Pass through the NER head
            logits = self.task_heads[task](token_embeddings)
        else:
            raise ValueError(f"Unsupported task: {task}")
        return logits
