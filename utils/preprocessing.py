import html
from core.logger import get_logger
import re


logger = get_logger(__name__)
class Preprocessing:
    """
    Service for cleaning input text before generating embeddings.
    """

    def __init__(self, text: str):
        self.text = text

    def cleaning(self) -> str:
        """
        Clean input text by removing HTML, normalizing whitespace,
        and replacing URLs.

        Returns:
            Cleaned text.
        """
        try:
            logger.info("Starting text preprocessing.")

            if not self.text or not self.text.strip():
                raise ValueError("Input text cannot be empty.")

            text = self.text

            # Normalize whitespace
            text = re.sub(r"\s+", " ", text)

            # Remove HTML tags
            text = re.sub(r"<.*?>", "", text)

            # Decode HTML entities
            text = html.unescape(text)

            # Replace URLs
            text = re.sub(r"http\S+|www\S+|https\S+", "url", text)

            text = text.strip()

            logger.info(
                "Text preprocessing completed successfully. Length: %d",
                len(text),
            )

            return text

        except Exception:
            logger.exception("Text preprocessing failed.")
            raise

    def __call__(self) -> str:
        """
        Allows the class instance to be called like a function.
        """
        return self.cleaning()

if __name__ == "__main__":
    preprocess = Preprocessing("For the last quarter of 2010 , Componenta 's net sales doubled to EUR131m from EUR76m for the same period a year earlier , while it moved to a zero pre-tax profit from a pre-tax loss of EUR7m")
    clean_text = preprocess()
    print(clean_text)