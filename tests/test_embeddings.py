import torch
from models.sentence_transformer import SentenceTransformer

def test_embeddings():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SentenceTransformer().to(device)
    
    sentences = [
        "This is a test sentence.",
        "Another sentence with different length.",
        "Machine learning is fascinating."
    ]
    
    tokenizer = model.tokenizer
    encoding = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Get sentence embeddings
    with torch.no_grad():
        embeddings = model(input_ids, attention_mask)
    
    # Output results
    for i, sentence in enumerate(sentences):
        print(f"Sentence: {sentence}")
        print(f"Embedding shape: {embeddings[i].shape}")
        print(f"First 5 values: {embeddings[i][:5].cpu().numpy()}\n")

if __name__ == "__main__":
    test_embeddings()
    