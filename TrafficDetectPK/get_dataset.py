from roboflow import Roboflow
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("RF_API_KEY")

try:
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("moiz-chauhan-u4zyj").project("traffic-erawl")
    dataset = project.version(6).download("yolov8")

    print("Dataset downloaded successfully!")
    print(f"Dataset location: {dataset.location}")

except Exception as e:
    print("Error occured: ", e)