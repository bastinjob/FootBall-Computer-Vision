import os
from roboflow import Roboflow
from ultralytics import YOLO
import dotenv


#get the api key from the enviroment variable
dotenv.load_dotenv()
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
HOME = os.path.expanduser("~")

#connect to you robolfow project 
def download_dataset(roboflow_api_key: str, workspace: str, project_name: str, version_number: int):
    
    rf = Roboflow(api_key=roboflow_api_key)
    project = rf.workspace(workspace).project(project_name)
    version = project.version(version_number)
    dataset = version.download("yolov8")
    return dataset


#Finetune the model on your dataset
def train_yolo(model_name: str, data_yaml_path: str, epochs=50, imgsz=1280, batch =6, plots=True):

    model = YOLO(model_name)
    model.train(
        data = data_yaml_path,
        epochs = epochs,
        imgsz = imgsz,
        batch = batch,
        plots = plots
    )


#validate fine tuned model
def validate_model(model_path: str, data_yaml_path: str):
    model = YOLO(model_path)
    model.val(data=data_yaml_path)

#deploy the finetunded model to robolfow universe
def deploy_model_to_roboflow(project, version, model_path: str):
    version.deploy(model_type="yolov8", model_path=model_path)

def main():
     # Roboflow dataset details
    workspace = "bastinjob1998-gmail-com"
    project_name = "football-players-detection-3zvbc-no5ik"
    version_number = 3

    # Download dataset
    dataset = download_dataset(ROBOFLOW_API_KEY, workspace, project_name, version_number)
    

    # Train model
    data_yaml_path = os.path.join(dataset.location, "data.yaml")
    train_yolo("yolov8s.pt", data_yaml_path)

    # Define trained model path (adjust `train` if needed)
    trained_model_path = os.path.join(HOME, "runs", "detect", "train", "weights", "best.pt")

    # Validate trained model
    validate_model(trained_model_path, data_yaml_path)

    # Deploy to Roboflow
    deploy_model_to_roboflow(project_name, version_number, os.path.join(HOME, "runs", "detect", "train2"))


if __name__ == "__main__":
    main() 