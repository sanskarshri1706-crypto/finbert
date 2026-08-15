import torch
import torch.nn as nn

from services.embedding_services import GenerateEmbedding
from core.logger import get_logger


logger = get_logger(__name__)


class MCPClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Linear(768, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(16, 3)
        )

    def forward(self, x):
        return self.layer(x)


class GetInference:

    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = MCPClassifier().to(self.device)

        try:
            state_dict = torch.load(
                model_path,
                map_location=self.device,
                weights_only=True
            )

            self.model.load_state_dict(state_dict)
            self.model.eval()

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise

        # Embedding model
        self.embed = GenerateEmbedding()


    def predict(self, text: str):

        try:
            # Generate embedding
            # Expected output: torch.Size([1,768])
            embedding = self.embed.get_embedding(text)

            embedding = embedding.to(self.device)

            logger.info(f"Embedding shape: {embedding.shape}")

            with torch.no_grad():

                # Model output shape: [1,3]
                output = self.model(embedding)

                logger.info(f"Model output: {output}")

                prediction = torch.argmax(
                    output,
                    dim=1
                ).item()
                print(prediction)


            labels = {
                0: "positive",
                1: "negative",
                2: "neutral"
            }

            sentiment = labels[prediction]

            logger.info(
                f"Prediction: {sentiment}"
            )

            return sentiment


        except Exception as e:
            logger.error(
                f"Inference error: {e}"
            )
            return None



if __name__ == "__main__":

    MODEL_PATH = "model/model_finbert.pth"

    classifier = GetInference(
        MODEL_PATH
    )


    test_text = """
     iran did strike on oil containers in qatar ships
    """


    result = classifier.predict(
        test_text
    )


    print("----------------------------")
    print("Text:")
    print(test_text)
    print("----------------------------")
    print("Sentiment:")
    print(result)
    print("----------------------------")




        

