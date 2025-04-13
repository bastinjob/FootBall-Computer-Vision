from inference import get_model
import dotenv
import os


def get_model(DETECTION_MODEL_ID: str):

    dotenv.load_dotenv()
    ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
    DETECTION_MODEL_ID = DETECTION_MODEL_ID
    DETECTION_MODEL = get_model(model_id=DETECTION_MODEL_ID, api_key=ROBOFLOW_API_KEY)

    return DETECTION_MODEL
