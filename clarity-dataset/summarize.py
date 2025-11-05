"""
BERT-based summary generation for QA pairs.
Generates dense vector representations using BERT embeddings.
"""
import numpy as np
import re
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM

from utils.logger import logger

# Basic configuration
BERT_NAME = "bert-base-uncased"
BART_NAME = "facebook/bart-large-cnn"


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


# Use BERT to select (MMR)
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')


def split_sentences(text: str):
    return [s.strip() for s in _SENT_SPLIT.split(text or "") if s.strip()]


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def select_topk_mmr(text, bert_model, bert_token, k=6, lam=0.65):
    sents = split_sentences(text)
    if not sents:
        return []

    emb = generate_bert_embeddings(sents, bert_model, bert_token)
    centroids = emb.mean(axis=0)

    chosen, picked = [], set()
    while len(chosen) < min(k, len(sents)):
        scores = []
        for i, e in enumerate(emb):
            if i in picked: continue
            rel = cos(e, centroids)
            div = max((cos(e, emb[j]) for j in picked), default=0.0)
            scores.append((lam * rel - (1 - lam) * div, i))
        _, idx = max(scores)
        picked.add(idx)
        chosen.append(sents[idx])
    return chosen


# Use BART to generate summaries
def _chunk_by_tokens(text, tokenizer, max_tokens=900):
    if not text:
        return []

    ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []

    for i in range(0, len(ids), max_tokens):
        piece = ids[i: i + max_tokens]
        chunks.append(tokenizer.decode(piece, skip_special_tokens=True))

    return chunks or [""]


def bart_summarize_text(text: str, tokenizer, model, device="cpu", max_input_tokens=900,
                        min_summary_tokens=40, max_summary_tokens=300, num_beams=4):
    # single text
    if not text or not text.strip():
        return ""

    chunks = _chunk_by_tokens(text, tokenizer, max_tokens=max_input_tokens)

    partial = []
    for ch in chunks:
        inputs = tokenizer(ch, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            ids = model.generate(
                **inputs,
                num_beams=num_beams,
                min_length=min_summary_tokens,
                max_length=max_summary_tokens,
                length_penalty=2.0,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
        partial.append(tokenizer.decode(ids[0], skip_special_tokens=True))

    if len(partial) == 1:
        return partial[0]

    # combine partial summary parts
    combined = " ".join(partial)
    inputs = tokenizer(combined, return_tensors="pt", truncation=True, max_length=1024).to(device)

    with torch.no_grad():
        ids = model.generate(
            **inputs,
            num_beams=num_beams,
            min_length=min_summary_tokens,
            max_length=max_summary_tokens,
            length_penalty=2.0,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

    return tokenizer.decode(ids[0], skip_special_tokens=True)


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

    logger.info(f"Loading BERT model: {BERT_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BERT_NAME)
    model = AutoModel.from_pretrained(BERT_NAME)

    # Extract contexts from context_clean, append to array
    contexts = []
    for item in data:
        context = item.get("context_clean", "")
        contexts.append(context)

    emb = generate_bert_embeddings(contexts, model, tokenizer)

    for i, item in enumerate(data):
        # store as list to remain JSON serializable
        item["summary_bert"] = emb[i].tolist()

    logger.info("BERT summaries added to dataset")
    return data


def generate_bart_summary(data, source_field="context_clean", target_field="summary_bart"):
    if not data:
        logger.warning("Empty dataset received, skipping summarization.")
        return data

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading BART model: {BART_NAME} on {device} ...")

    bert_tok = AutoTokenizer.from_pretrained(BERT_NAME)
    bert_model = AutoModel.from_pretrained(BERT_NAME)
    bart_tok = AutoTokenizer.from_pretrained(BART_NAME)
    bart_model = AutoModelForSeq2SeqLM.from_pretrained(BART_NAME).to(device).eval()

    for item in tqdm(data, desc="Generating summaries ..."):
        source = item.get(source_field, "") or ""

        key_sentences = select_topk_mmr(source, bert_model, bert_tok, k=6, lam=0.65)
        selected_text = " ".join(key_sentences)

        summary = bart_summarize_text(selected_text, bart_tok, bart_model, device=device)
        item[target_field] = summary

        emb = generate_bert_embeddings([summary], bert_model, bert_tok)
        item["summary_bert"] = emb[0].tolist()

    logger.info("BART summaries generated successfully.")
    return data


# Tests
if __name__ == "__main__":
    data = [
        {
            "context_clean": "President Biden said the new economic plan will create more jobs. However, some critics argue that the tax increase may hurt small businesses. The government insists that overall growth will be strong."},
        {
            "context_clean": "The company reported strong quarterly earnings, driven by growth in its cloud services. Analysts expect revenue to continue rising next quarter."}
    ]

    data = generate_bart_summary(data)

    for d in data:
        print("\n---SUMMARY---")
        print(d["summary_bart"])
        print("Vector dim:", len(d["summary_bert"]))
