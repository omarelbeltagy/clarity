"""
BERT-based summary generation for QA pairs.
Generates dense vector representations using BERT embeddings.
"""
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from loguru import logger


MODEL_NAME = "bert-base-uncased"


def mean_pooling(model_output, attention_mask):
    """Average token embeddings weighted by attention mask."""
    token_embeddings = model_output[0]
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)


def generate_bert_embeddings(texts, model, tokenizer, batch_size=8, max_length=256):
    """Generate mean-pooled BERT embeddings for a list of texts."""
    embeddings = []
    model.eval()
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating BERT summaries"):
        batch = texts[i:i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = model(**encoded)
            pooled = mean_pooling(outputs, encoded["attention_mask"])
        embeddings.append(pooled.cpu().numpy())
    return np.vstack(embeddings)


def generate_bert_summary(data, **kwargs):
    """
    Given a list of dicts (dataset entries),
    generate BERT summaries and add a new key `summary_bert` to each entry.
    
    Args:
        data: List of dataset items
        **kwargs: Ignored (for compatibility with app.py)
    
    Returns:
        List of items with added 'summary_bert' field
    """
    if not data:
        logger.warning("Empty dataset received, skipping summarization.")
        return data

    logger.info(f"Loading BERT model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    # Extract context from interview_question + interview_answer
    texts = []
    for item in data:
        question = item.get("interview_question", "")
        answer = item.get("interview_answer", "")
        context = question + "\n" + answer
        texts.append(context)
    
    emb = generate_bert_embeddings(texts, model, tokenizer)

    for i, item in enumerate(data):
        # store as list to remain JSON serializable
        item["summary_bert"] = emb[i].tolist()

    logger.info("BERT summaries added to dataset")
    return data