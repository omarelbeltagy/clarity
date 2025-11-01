from dataclasses import dataclass


@dataclass
class ClassificationRequest:
    question: str
    context: str
