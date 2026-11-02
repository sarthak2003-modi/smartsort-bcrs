"""SmartSort: AI vision + sensor fusion for Singapore BCRS return verification."""

__version__ = "0.1.0"

from smartsort.pipeline import Decision, evaluate_return

__all__ = ["Decision", "evaluate_return", "__version__"]
