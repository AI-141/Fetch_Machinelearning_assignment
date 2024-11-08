import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import List

class ClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer_name='bert-base-uncased', max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0) 
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long),
            'task': 'classification'
        }

class NERDataset(Dataset):
    def __init__(self, texts, labels, tokenizer_name='bert-base-uncased', max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.label_map = {label: i for i, label in enumerate(self.get_ner_labels())}
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        ner_labels = self.labels[idx]
        
        encoding = self.tokenizer(
            text.split(),
            is_split_into_words=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0) 
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Align labels with tokens
        word_ids = encoding.word_ids()
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Special token
            else:
                label_ids.append(self.label_map.get(ner_labels[word_idx], -100))
        
        labels = torch.tensor(label_ids, dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'task': 'ner'
        }
    
    @staticmethod
    def get_ner_labels():
        return ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']

def create_classification_data():
    texts = [
        "The stock market crashed today due to unforeseen circumstances.",
        "The new movie received rave reviews from critics.",
        "The weather is expected to be sunny throughout the week.",
        "Advancements in AI are transforming the technology sector.",
        "Political tensions are rising in the eastern region.",
        "The local sports team won the championship after a tough season.",
    ]
    
    # Classification labels (0: Finance, 1: Entertainment, 2: Weather, 3: Technology, 4: Politics, 5: Sports)
    classification_labels = [0, 1, 2, 3, 4, 5]
    
    return texts, classification_labels

def create_ner_data():
    texts = [
        "Apple Inc. is planning to open a new store in New York City.",
        "Microsoft CEO Satya Nadella announced new AI features today.",
        "The Golden Gate Bridge in San Francisco is an iconic landmark.",
        "Tesla's Elon Musk spoke about SpaceX's latest rocket launch.",
        "Google's headquarters in Mountain View employs thousands.",
    ]
    
    # NER labels for each word in the sentences
    ner_labels = [
        ['B-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'I-LOC', 'I-LOC', 'I-LOC', 'I-LOC', 'I-LOC', 'O'],
        ['B-ORG', 'B-PER', 'I-PER', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
        ['B-LOC', 'I-LOC', 'I-LOC', 'I-LOC', 'I-LOC', 'O', 'B-LOC', 'I-LOC', 'I-LOC', 'I-LOC', 'O', 'O'],
        ['B-ORG', 'B-PER', 'I-PER', 'O', 'O', 'B-ORG', 'O', 'O', 'O', 'O', 'O'],
        ['B-ORG', 'O', 'O', 'O', 'B-LOC', 'I-LOC', 'O', 'O', 'O']
    ]
    
    return texts, ner_labels

def get_data_loaders(batch_size=2):
    # Create classification data
    classification_texts, classification_labels = create_classification_data()
    classification_dataset = ClassificationDataset(
        texts=classification_texts,
        labels=classification_labels
    )
    
    # Create NER data
    ner_texts, ner_labels = create_ner_data()
    ner_dataset = NERDataset(
        texts=ner_texts,
        labels=ner_labels
    )
    
    # Create data loaders
    classification_loader = DataLoader(
        classification_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    ner_loader = DataLoader(
        ner_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    return classification_loader, ner_loader
