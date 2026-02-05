"""
Complete filler word removal system using spaCy with custom extensions
Handles both single-word fillers and multi-word phrases
"""

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Token
import re
from typing import List, Set, Optional
from datasets import load_dataset
import os
import json


class SpacyCleaner:
    """
    A comprehensive filler word removal system that:
    1. Uses spaCy for intelligent tokenization and stopword handling
    2. Adds custom filler words and phrases
    3. Allows selective stopword preservation
    """
    
    # Conversational fillers (sounds)
    FILLER_SOUNDS = {
        "um", "umm", "ummm", "uh", "uhh", "uhhh", "ah", "ahh",
        "eh", "ehh", "hm", "hmm", "hmmm", "mm", "mmm", "oh", "ohh"
    }
    
    # Single-word discourse markers and hedges
    FILLER_WORDS = {
        "basically", "literally", "actually", "honestly", "frankly",
        "seriously", "clearly", "obviously", "definitely", "certainly",
        "apparently", "presumably", "supposedly", "arguably",
        "well,", "okay", "ok", "right", "yeah", "yep",
        "alright", "anyway", "anyways", "whatever", "please",
        "listen", "look", "folks", "essentially", "technically",
        "practically", "virtually", "just", "maybe", "perhaps",
        "probably", "possibly", "sorta", "kinda", "stuff",
        "anyhoo", "anyhow","somehow", "someway", "really", "truly", "very", "much",
        "mr", "mr.", "mrs", "mrs.", "president", "chancellor"
    }
    
    # Multi-word filler phrases
    FILLER_PHRASES = [
        "you know", "i mean", "i think", "i guess", "i suppose",
        "you see", "you know what", "you know what i mean",
        "kind of", "sort of", "a bit", "a little", "a little bit",
        "at the end of the day", "by the end of the day", "to be honest", "to be fair",
        "to be frank", "to be clear", "to tell you the truth", "to be honest"
        "the thing is", "the fact is", "the point is",
        "by the way", "in other words", "so to speak", "so to say"
        "as it were", "if you will", "more or less",
        "you know what i'm saying", "what i'm trying to say is",
        "that being said", "having said that", "all things considered",
        "for what it's worth", "at this point in time", "mr. president", "thank you"
    ]
    
    def __init__(
        self,
        model: str = "en_core_web_sm",
        preserve_stopwords: Optional[Set[str]] = None,
        custom_fillers: Optional[Set[str]] = None,
        custom_phrases: Optional[List[str]] = None,
        clear_name: bool = True
    ):
        """
        Initialize the filler word remover
        
        Args:
            model: spaCy model to use
            preserve_stopwords: Set of stopwords to NOT remove (e.g., {"not", "no"})
            custom_fillers: Additional single-word fillers to remove
            custom_phrases: Additional multi-word phrases to remove
        """
        self.nlp = spacy.load(model)
        
        self.fillers = self.FILLER_SOUNDS | self.FILLER_WORDS
        if custom_fillers:
            self.fillers |= custom_fillers
        
        self.filler_phrases = self.FILLER_PHRASES.copy()
        if custom_phrases:
            self.filler_phrases.extend(custom_phrases)
        
        self.phrase_patterns = self._compile_phrase_patterns()
        
        # Stopwords to preserve (commonly: not, no, never for sentiment)
        self.preserve_stopwords = preserve_stopwords or set()

        self._compile_title_patterns()
        
        for filler in self.fillers:
            self.nlp.vocab[filler].is_stop = True
        
        print(f"Initialized with {len(self.fillers)} filler words and {len(self.filler_phrases)} filler phrases")
    
    def _compile_phrase_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for multi-word phrases"""
        patterns = []
        
        for phrase in self.filler_phrases:
            # Make pattern case-insensitive and flexible with whitespace
            pattern = r'\b' + re.escape(phrase).replace(r'\ ', r'\s+') + r'\b'
            patterns.append(re.compile(pattern, re.IGNORECASE))
        
        return patterns

    def _compile_title_patterns(self):
        """Pre-compile patterns for common titles and honorifics"""
        # Direct addresses
        self.rx_direct_address = re.compile(
            r"(?i)\b(?:sir|ma'am|madam)\b\s*[.,;:!?—–-]?\s*"
        )
        
        # Honorific titles (Mr., Mrs., Ms., President)
        self.rx_honoured_title = re.compile(
            r"(?i)\b(?:mr|mister|ms|mrs|madam)\.?\s+president\b\s*[.,;:!?—–-]?\s*"
        )

    def _compile_name_patterns(self, president_name: str) -> List[re.Pattern]:
        """
        Compile regex patterns for a specific president name.
        
        Args:
            president_name: Full name like "Joe Biden" or "Donald Trump"
            
        Returns:
            List of compiled regex patterns to match name variations
        """
        if not president_name or not president_name.strip():
            return []
        
        patterns = []
        
        # Split name into parts
        name_parts = president_name.strip().split()
        if not name_parts:
            return []
        
        # Get first name, last name, and full name
        first_name = name_parts[0]
        last_name = name_parts[-1]
        full_name = " ".join(name_parts)
        
        # Pattern 1: "President [Full Name]"
        full_escaped = re.escape(full_name).replace(r'\ ', r'\s+')
        patterns.append(re.compile(
            r"(?i)\bpresident\s+" + full_escaped + r"(?:\s*'?s)?\s*[.,;:!?—–-]?\s*"
        ))
        
        # Pattern 2: "President [Last Name]"
        patterns.append(re.compile(
            r"(?i)\bpresident\s+" + re.escape(last_name) + r"(?:\s*'?s)?\s*[.,;:!?—–-]?\s*"
        ))
        
        # Pattern 3: "[Full Name]" standalone
        patterns.append(re.compile(
            r"(?i)\b" + full_escaped + r"(?:\s*'?s)?\s*[.,;:!?—–-]?\s*"
        ))
        
        # Pattern 4: "Mr./Mrs. [Last Name]"
        patterns.append(re.compile(
            r"(?i)\b(?:mr|mister|ms|mrs)\.?\s+" + re.escape(last_name) + r"(?:\s*'?s)?\s*[.,;:!?—–-]?\s*"
        ))
        
        # Pattern 5: Just "[Last Name]" (more aggressive, use carefully)
        # Only at word boundaries and followed by punctuation or end
        patterns.append(re.compile(
            r"(?i)\b" + re.escape(last_name) + r"(?:\s*'?s)?\s*(?=[.,;:!?—–-]|\s|$)"
        ))
        
        return patterns

    def _remove_president_names(self, text: str, president_name: Optional[str]) -> str:
        """
        Remove president name mentions from text.
        
        Args:
            text: Input text
            president_name: Name like "Joe Biden", "Donald Trump", etc.
            
        Returns:
            Text with president names removed
        """
        if not president_name or not text:
            return text
        
        cleaned = text
        
        # Remove direct addresses (sir, ma'am)
        cleaned = self.rx_direct_address.sub(' ', cleaned)
        
        # Remove honorific titles (Mr. President, etc.)
        cleaned = self.rx_honoured_title.sub(' ', cleaned)
        
        # Remove specific name patterns
        name_patterns = self._compile_name_patterns(president_name)
        for pattern in name_patterns:
            cleaned = pattern.sub(' ', cleaned)
        
        return cleaned

    def _clean_punctuation(self, text: str) -> str:
        """
        Normalize punctuation after token removal:
        - Remove double spaces
        - Remove spaces before punctuation (, . ; : ! ?)
        - Collapse repeated punctuation
        """
        if not text:
            return ""

        text = re.sub(r"\s+([.,;:!?])", r"\1", text)
        text = re.sub(r"([.,;:!?]){2,}", r"\1", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r'\[[^\]]*\]', ' ', text)
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
        text = re.sub(r'([a-z])([A-Z])', r'\1. \2', text)
        text = re.sub(r'^[\s\.\,\;\:\!\?\-—–"\'\(\)\[\]\{]+', '', text)
        return text


    
    def _remove_phrase_fillers(self, text: str) -> str:
        """Remove multi-word filler phrases using regex"""
        cleaned = text
        for pattern in self.phrase_patterns:
            cleaned = pattern.sub(' ', cleaned)
        return cleaned

    def _remove_token_with_attached_punct(self, token):
        """Yield token indices to remove (token + punctuation neighbors)."""
        to_remove = {token.i}

        # punctuation immediately before
        if token.i > 0:
            prev = token.doc[token.i - 1]
            if prev.is_punct:
                to_remove.add(prev.i)

        # punctuation immediately after
        if token.i < len(token.doc) - 1:
            nxt = token.doc[token.i + 1]
            if nxt.is_punct:
                to_remove.add(nxt.i)

        return to_remove

    def _rebuild_text(self, doc, remove_indices):
        """
        Rebuild text from tokens after removals.
        Uses spaCy's whitespace info to preserve original formatting.
        """
        parts = []

        for i, token in enumerate(doc):
            if i in remove_indices:
                continue
            
            # For contractions like "n't", "'s", "'d" - attach to previous token
            if token.text in "'":
                if parts:
                    # Remove any trailing space from previous part
                    if parts[-1].endswith(' '):
                        parts[-1] = parts[-1].rstrip()
                    parts.append(token.text)
                else:
                    parts.append(token.text)
            else:
                parts.append(token.text_with_ws)

        text = ''.join(parts)
        return text.rstrip()  # Remove trailing whitespace


    
    def clean_text(
        self,
        text: str,
        remove_stopwords: bool = False,
        president_name: Optional[str] = None
    ) -> str:
        """
        Clean text by removing filler words and optionally stopwords
        
        Args:
            text: Input text to clean
            remove_stopwords: Whether to remove standard stopwords too
            keep_punctuation: Whether to keep punctuation marks
            president_name: Name of president to remove (e.g., "Joe Biden")  # ADD THIS LINE
            
        Returns:
            Cleaned text
        """
        remove_indices = set()
        if not text or not text.strip():
            return ""

        if president_name:
            text = self._remove_president_names(text, president_name)
        
        text = self._remove_phrase_fillers(text)
        
        doc = self.nlp(text)
        
        for token in doc:
            if token.is_space:
                continue

            # Remove filler word
            if token.lower_ in self.fillers:
                remove_indices |= self._remove_token_with_attached_punct(token)
                continue

            # Remove stopword (unless preserved)
            if remove_stopwords and token.is_stop and token.lower_ not in self.preserve_stopwords:
                remove_indices |= self._remove_token_with_attached_punct(token)
                continue

        text = self._rebuild_text(doc, remove_indices)

        text = self._clean_punctuation(text)

        return text
    
    def process_dataset(
        self,
        dataset_name: str = "ailsntua/QEvasion",
        question_col: str = "interview_question",
        answer_col: str = "interview_answer",
        president_col: str = "president",
        remove_stopwords: bool = False,
        batch_size: int = 100,
        clear_names: bool = True
    ) -> tuple:
        """
        Process entire dataset and return cleaned versions as lists
        
        Args:
            dataset_name: HuggingFace dataset name
            question_col: Column name for questions
            answer_col: Column name for answers
            president_col: Column name for president names
            remove_stopwords: Whether to remove stopwords
            batch_size: Batch size for processing
            clear_names: Whether to remove president names
        
        Returns:
            Tuple of (train_list, test_list) with cleaned records as Python lists
        """
        print(f"Loading dataset: {dataset_name}")
        ds_train = load_dataset(dataset_name, split="train")
        ds_test = load_dataset(dataset_name, split="test")
        
        def clean_records(dataset):
            """Clean a dataset and return as list of dicts"""
            cleaned_records = []
            
            for record in dataset:
                president = record.get(president_col, None)
                
                question_clean = self.clean_text(
                    record[question_col],
                    remove_stopwords=remove_stopwords,
                    president_name=president if clear_names else None
                )
                
                answer_clean = self.clean_text(
                    record[answer_col],
                    remove_stopwords=remove_stopwords,
                    president_name=president if clear_names else None
                )
                
                # Create cleaned record preserving all original fields
                cleaned_record = dict(record)
                cleaned_record[f"{question_col}_clean"] = question_clean
                cleaned_record[f"{answer_col}_clean"] = answer_clean
                
                cleaned_records.append(cleaned_record)
            
            return cleaned_records
        
        print("Cleaning training data...")
        train_cleaned = clean_records(ds_train)
        
        print("Cleaning test data...")
        test_cleaned = clean_records(ds_test)
        
        print("✓ Dataset cleaning complete!")
        return train_cleaned, test_cleaned


# Convenience function
def create_cleaner(
    preserve_negation: bool = True,
    custom_fillers: Optional[Set[str]] = None,
    clear_name: bool = True
) -> SpacyCleaner:
    """
    Create a cleaner with sensible defaults
    
    Args:
        preserve_negation: Keep words like "not", "no", "never" (important for sentiment)
        custom_fillers: Additional fillers specific to your domain
    """
    preserve = {"not", "no", "never", "neither", "nor", "none", "in", "on", "at", "the", "a", "but"} if preserve_negation else None
    return SpacyCleaner(
        preserve_stopwords=preserve,
        custom_fillers=custom_fillers,
        clear_name=clear_name
    )


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    cleaner = create_cleaner(preserve_negation=True)
    train_clean, test_clean = cleaner.process_dataset(
        dataset_name="ailsntua/QEvasion",
        remove_stopwords=False,
        clear_names=True  # Set to False to keep president names
    )

    # Save as JSON files
    os.makedirs("./data/cleaned", exist_ok=True)

    with open("./data/cleaned/train.json", "w", encoding="utf-8") as f:
        json.dump(train_clean, f, ensure_ascii=False, indent=2)

    with open("./data/cleaned/test.json", "w", encoding="utf-8") as f:
        json.dump(test_clean, f, ensure_ascii=False, indent=2)

    print(f"✓ Saved train.json ({len(train_clean)} records)")
    print(f"✓ Saved test.json ({len(test_clean)} records)")

