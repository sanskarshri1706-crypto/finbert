import torch
from transformers import AutoTokenizer, AutoModel

from utils.preprocessing import Preprocessing
from core.logger import get_logger

logger = get_logger(__name__)


class GenerateEmbedding:
    """
    Generates embeddings using a transformer model.
    """

    def __init__(self, embedding_model_name: str = "ProsusAI/finbert"):
        try:
            logger.info(
                "Loading embedding model: %s",
                embedding_model_name
            )

            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

            logger.info(
                "Using device: %s",
                self.device
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                embedding_model_name
            )

            logger.info(
                "Tokenizer loaded successfully"
            )

            self.model = AutoModel.from_pretrained(
                embedding_model_name
            )

            self.model.to(self.device)
            self.model.eval()

            logger.info(
                "Embedding model loaded successfully"
            )

        except Exception:
            logger.exception(
                "Failed to load embedding model"
            )
            raise


    def get_embedding(self, text: str) -> torch.Tensor:
        """
        Convert text into transformer embedding.

        Args:
            text: Input sentence.

        Returns:
            CLS embedding tensor.
        """

        try:
            if not text or not text.strip():
                raise ValueError(
                    "Input text cannot be empty"
                )

            logger.info(
                "Starting text preprocessing"
            )

            preprocessing_text = Preprocessing(text)
            clean_text = preprocessing_text()

            logger.info(
                "Text cleaned successfully"
            )


            encodings = self.tokenizer(
                clean_text,
                padding=True,
                truncation=True,
                max_length=160,
                return_tensors="pt"
            )

            logger.info(
                "Tokenization completed"
            )


            input_ids = encodings["input_ids"].to(
                self.device
            )

            attention_mask = encodings["attention_mask"].to(
                self.device
            )


            with torch.no_grad():

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )


            # CLS token embedding
            cls_embeddings = outputs.last_hidden_state[:, 0, :]


            logger.info(
                "Embedding generated successfully. Shape: %s",
                cls_embeddings.shape
            )


            return cls_embeddings.cpu()


        except Exception:
            logger.exception(
                "Embedding generation failed"
            )
            raise


if __name__ == "__main__":
    embed = GenerateEmbedding()
    embeddings = embed.get_embedding("For the last quarter of 2010 , Componenta 's net sales doubled to EUR131m from EUR76m for the same period a year earlier , while it moved to a zero pre-tax profit from a pre-tax loss of EUR7m")
    print(embeddings)
    print(embeddings.shape)

