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
import pandas as pd


class FillerWordRemover:
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
        "like", "well", "so", "okay", "ok", "right", "yeah", "yep",
        "alright", "anyway", "anyways", "whatever", "please",
        "listen", "look", "folks", "essentially", "technically",
        "practically", "virtually", "just", "maybe", "perhaps",
        "probably", "possibly", "sorta", "kinda", "stuff",
        "anyhoo", "anyhow","somehow", "someway", "really", "truly",
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
        "for what it's worth", "at this point in time"
    ]
    
    def __init__(
        self,
        model: str = "en_core_web_sm",
        preserve_stopwords: Optional[Set[str]] = None,
        custom_fillers: Optional[Set[str]] = None,
        custom_phrases: Optional[List[str]] = None
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
        
        # Build complete filler set
        self.fillers = self.FILLER_SOUNDS | self.FILLER_WORDS
        if custom_fillers:
            self.fillers |= custom_fillers
        
        # Build complete phrase list
        self.filler_phrases = self.FILLER_PHRASES.copy()
        if custom_phrases:
            self.filler_phrases.extend(custom_phrases)
        
        # Compile phrase patterns (sorted by length for longest match first)
        self.phrase_patterns = self._compile_phrase_patterns()
        
        # Stopwords to preserve (commonly: not, no, never for sentiment)
        self.preserve_stopwords = preserve_stopwords or set()
        
        # Add fillers to spaCy's stopword list
        for filler in self.fillers:
            self.nlp.vocab[filler].is_stop = True
        
        print(f"Initialized with {len(self.fillers)} filler words and {len(self.filler_phrases)} filler phrases")
    
    def _compile_phrase_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for multi-word phrases"""
        patterns = []
        # Sort by length (descending) to match longer phrases first
        sorted_phrases = sorted(self.filler_phrases, key=len, reverse=True)
        
        for phrase in sorted_phrases:
            # Make pattern case-insensitive and flexible with whitespace
            pattern = r'\b' + re.escape(phrase).replace(r'\ ', r'\s+') + r'\b'
            patterns.append(re.compile(pattern, re.IGNORECASE))
        
        return patterns

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
        - Avoid spaces before punctuation
        - Preserve spaces where needed
        """
        tokens = []
        for i, token in enumerate(doc):
            if i in remove_indices:
                continue

            if token.is_punct:
                # Attach punctuation directly to previous token (no leading space)
                tokens.append(token.text)
            elif tokens and re.match(r'^[.,;:!?\'")\]]', token.text):
                # Safety: attach punctuation directly if regex matches
                tokens.append(token.text)
            else:
                # Normal word
                if tokens:
                    tokens.append(' ' + token.text)
                else:
                    tokens.append(token.text)
        return ''.join(tokens)


    
    def clean_text(
        self,
        text: str,
        remove_stopwords: bool = False,
        keep_punctuation: bool = True
    ) -> str:
        """
        Clean text by removing filler words and optionally stopwords
        
        Args:
            text: Input text to clean
            remove_stopwords: Whether to remove standard stopwords too
            keep_punctuation: Whether to keep punctuation marks
            
        Returns:
            Cleaned text
        """
        remove_indices = set()
        if not text or not text.strip():
            return ""
        
        # Step 1: Remove multi-word phrases first
        text = self._remove_phrase_fillers(text)
        
        # Step 2: Process with spaCy
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

        # Step 4: Rebuild clean text using spaCy token whitespace to avoid inserting artificial spaces
        text = self._rebuild_text(doc, remove_indices)

        # Step 5: Final punctuation cleanup
        text = self._clean_punctuation(text)

        return text
    
    def process_dataset(
        self,
        dataset_name: str = "ailsntua/QEvasion",
        question_col: str = "interview_question",
        answer_col: str = "interview_answer",
        remove_stopwords: bool = False,
        batch_size: int = 100
    ) -> tuple:
        """
        Process entire dataset and return cleaned versions
        
        Returns:
            Tuple of (train_dataset, test_dataset) with cleaned columns added
        """
        print(f"Loading dataset: {dataset_name}")
        ds_train = load_dataset(dataset_name, split="train")
        ds_test = load_dataset(dataset_name, split="test")
        
        def clean_batch(batch):
            questions_clean = []
            answers_clean = []
            
            for q, a in zip(batch[question_col], batch[answer_col]):
                questions_clean.append(self.clean_text(q, remove_stopwords=remove_stopwords))
                answers_clean.append(self.clean_text(a, remove_stopwords=remove_stopwords))
            
            return {
                f"{question_col}_clean": questions_clean,
                f"{answer_col}_clean": answers_clean
            }
        
        print("Cleaning training data...")
        ds_train_clean = ds_train.map(clean_batch, batched=True, batch_size=batch_size)
        
        print("Cleaning test data...")
        ds_test_clean = ds_test.map(clean_batch, batched=True, batch_size=batch_size)
        
        print("✓ Dataset cleaning complete!")
        return ds_train_clean, ds_test_clean


# Convenience function
def create_cleaner(
    preserve_negation: bool = True,
    custom_fillers: Optional[Set[str]] = None
) -> FillerWordRemover:
    """
    Create a cleaner with sensible defaults
    
    Args:
        preserve_negation: Keep words like "not", "no", "never" (important for sentiment)
        custom_fillers: Additional fillers specific to your domain
    """
    preserve = {"not", "no", "never", "neither", "nor", "none", "in", "on", "at", "the", "a", "but"} if preserve_negation else None
    return FillerWordRemover(
        preserve_stopwords=preserve,
        custom_fillers=custom_fillers
    )


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    # Example 1: Basic usage
    print("=" * 70)
    print("EXAMPLE 1: Basic Filler Removal")
    print("=" * 70)
    
    cleaner = create_cleaner()
    
    test_cases = [
        "I don't talk about whether or not I'd use military force. That's not appropriate to be talking about. But I can tell you this: They will not be doing nuclear weapons. That I can tell you. Okay? They're not going to be doing nuclear weapons. You can bank on it.Okay. Please.[.]",
        "Um, well, you know, I think we should, like, proceed with this.",
        "At the end of the day, basically, we need to focus on results.",
        "So, uh, to be honest, I mean, this is actually quite important.",
        "You know what I'm saying? Like, it's kind of complicated, right?",
    ]
    
    for text in test_cases:
        cleaned = cleaner.clean_text(text)
        print(f"\nOriginal:  {text}")
        print(f"Cleaned:   {cleaned}")
        print(f"Reduction: {len(text)} → {len(cleaned)} chars ({100*(1-len(cleaned)/len(text)):.1f}%)")
    
    # Example 2: With stopword removal
    print("\n" + "=" * 70)
    print("EXAMPLE 2: With Stopword Removal (preserving negation)")
    print("=" * 70)
    
    text = "Well, I do not think this is the right approach for us."
    print(f"\nOriginal: {text}")
    print(f"Fillers only: {cleaner.clean_text(text, remove_stopwords=False)}")
    print(f"With stopwords: {cleaner.clean_text(text, remove_stopwords=True)}")
    
    # Example 3: Custom fillers for political interviews
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Custom Fillers for Political Domain")
    print("=" * 70)
    
    political_fillers = {
        "listen", "look", "folks", "believe me",
        "let me be clear", "make no mistake"
    }
    
    # Note: multi-word phrases go in custom_phrases parameter
    political_cleaner = FillerWordRemover(
        custom_fillers={"listen", "look", "folks"},
        custom_phrases=["believe me", "let me be clear", "make no mistake"]
    )
    
    political_text = "Look, folks, believe me, we need to move forward on this issue."
    print(f"\nOriginal: {political_text}")
    print(f"Cleaned:  {political_cleaner.clean_text(political_text)}")
    
    # Example 4: Process small sample
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Dataset Processing (Sample)")
    print("=" * 70)
    
    # Simulate dataset
    sample_data = {
        "interview_question": [
            "Um, Mr. President, what are your thoughts on this?",
            "So, like, what's your position on the economy?"
        ],
        "interview_answer": [
            "Well, you know, I think we should focus on growth.",
            "At the end of the day, basically, jobs are key."
        ]
    }
    
    for i, (q, a) in enumerate(zip(sample_data["interview_question"], 
                                    sample_data["interview_answer"])):
        print(f"\nPair {i+1}:")
        print(f"Q (original): {q}")
        print(f"Q (cleaned):  {cleaner.clean_text(q)}")
        print(f"A (original): {a}")
        print(f"A (cleaned):  {cleaner.clean_text(a)}")
    
    # Example 5: Statistics
    print("\n" + "=" * 70)
    print("STATISTICS")
    print("=" * 70)
    print(f"Total filler words loaded: {len(cleaner.fillers)}")
    print(f"Total filler phrases loaded: {len(cleaner.filler_phrases)}")
    print(f"Stopwords preserved: {cleaner.preserve_stopwords}")
    
    print("\n" + "=" * 70)
    print("To process your full dataset, uncomment this:")
    print("=" * 70)
    print("""
# Process full dataset
cleaner = create_cleaner(preserve_negation=True)
train_clean, test_clean = cleaner.process_dataset(
    dataset_name="ailsntua/QEvasion",
    remove_stopwords=False  # Set True to also remove stopwords
)

# Save to disk or push to hub
train_clean.save_to_disk("./data/qevasion_train_clean")
test_clean.save_to_disk("./data/qevasion_test_clean")
""")