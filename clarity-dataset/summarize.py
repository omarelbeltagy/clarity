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


def generate_bert_summary(data, text_field="context_clean"):
    """
    Given a list of dicts (dataset entries),
    generate BERT summaries for the specified field,
    and add a new key `summary_bert` to each entry.
    """
    if not data:
        logger.warning("Empty dataset received, skipping summarization.")
        return data

    logger.info(f"Loading BERT model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    texts = [item.get(text_field, "") for item in data]
    emb = generate_bert_embeddings(texts, model, tokenizer)

    for i, item in enumerate(data):
        # store as list to remain JSON serializable
        item["summary_bert"] = emb[i].tolist()

    return data
"""
BERT-based summary generation for QA pairs.
Generates dense vector representations using BERT embeddings.
"""
from typing import List, Dict, Any
import torch
from transformers import BertTokenizer, BertModel
from loguru import logger


class BertSummarizer:
    """Generate BERT-based summaries (embeddings) for text pairs."""
    
    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 512):
        """
        Initialize BERT model and tokenizer.
        
        Args:
            model_name: HuggingFace model identifier
            max_length: Maximum sequence length for tokenization
        """
        logger.info(f"Loading BERT model: {model_name}")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"BERT model loaded on device: {self.device}")
    
    def _encode_text(self, text: str) -> torch.Tensor:
        """
        Encode a single text into BERT embeddings.
        
        Args:
            text: Input text to encode
            
        Returns:
            CLS token embedding as tensor
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token embedding as sentence representation
            cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        return cls_embedding.cpu()
    
    def generate_summary(self, question: str, answer: str) -> List[float]:
        """
        Generate BERT-based summary for a QA pair.
        
        Args:
            question: Interview question text
            answer: Interview answer text
            
        Returns:
            List of floats representing the combined embedding
        """
        # Combine question and answer with separator
        combined_text = f"{question} [SEP] {answer}"
        
        # Generate embedding
        embedding = self._encode_text(combined_text)
        
        # Convert to list of floats
        return embedding.squeeze().tolist()
    
    def generate_batch(self, items: List[Dict[str, Any]], 
                      question_key: str = "interview_question",
                      answer_key: str = "interview_answer") -> List[List[float]]:
        """
        Generate summaries for a batch of QA pairs.
        
        Args:
            items: List of dictionaries containing QA pairs
            question_key: Key for question field in items
            answer_key: Key for answer field in items
            
        Returns:
            List of embeddings (each as list of floats)
        """
        summaries = []
        total = len(items)
        
        for idx, item in enumerate(items):
            if (idx + 1) % 100 == 0:
                logger.info(f"Processing {idx + 1}/{total} items...")
            
            question = item.get(question_key, "")
            answer = item.get(answer_key, "")
            
            summary = self.generate_summary(question, answer)
            summaries.append(summary)
        
        logger.info(f"Generated {len(summaries)} BERT summaries")
        return summaries


def generate_bert_summary(data: List[Dict[str, Any]], 
                       question_key: str = "interview_question",
                       answer_key: str = "interview_answer",
                       model_name: str = "bert-base-uncased") -> List[Dict[str, Any]]:
    """
    Add BERT-based summaries to dataset items.
    
    Args:
        data: List of dataset items
        question_key: Key for question field
        answer_key: Key for answer field
        model_name: BERT model to use
        
    Returns:
        List of items with added 'summary_bert' field
    """
    logger.info(f"Generating BERT summaries for {len(data)} items...")
    
    summarizer = BertSummarizer(model_name=model_name)
    summaries = summarizer.generate_batch(data, question_key, answer_key)
    
    # Add summaries to data
    for item, summary in zip(data, summaries):
        item["summary_bert"] = summary
    
    logger.info("BERT summaries added to dataset")
    return data