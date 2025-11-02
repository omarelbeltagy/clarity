"""Data transfer objects used by the API and CLI.

Simple container classes (dataclasses) that represent request payloads
and lightweight DTOs for internal calls.
"""

from dataclasses import dataclass, asdict


@dataclass
class ClassificationRequest:
    """Request payload for classification endpoints / model API.

    Parameters
    ----------
    question : str
        The question or query text to classify.
    context : str
        The accompanying context text to consider during classification.

    Examples
    --------
    >>> req = ClassificationRequest(question="When is the event?", context="The event is on 2025-01-01.")
    """
    question: str
    context: str

    def dict(self):
        return asdict(self)
