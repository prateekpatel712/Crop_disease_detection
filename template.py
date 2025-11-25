import os 
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)

project_name = "Crop_disease_detection"

list_of_files = [
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/model_monitoring.py",
    f"src/{project_name}/pipelines/__init__.py",
    f"src/{project_name}/pipelines/training_pipeline.py",
    f"src/{project_name}/pipelines/prediction_pipeline.py",
    f"src/{project_name}/exceptions.py",
    f"src/{project_name}/logging.py",
    f"src/{project_name}/utils.py",
    "app.py",
    "Dockerfile",
    "requirements.txt",
    "README.md",
]

TEMPLATES ={
    "README.md": "#CROP DISEASE DETECTOR\n\nThis project uses deep learning for detection of plant leaf diseases.",
    
    "app.py": """from fastapi import FastAPI\n

    app = FastAPI()\n

    @app.get("/")\n
    def home():
        return {"message":"Crop Disease Detector API is running"}
    
    """,

    f"src/{project_name}/utils.py": """import yaml\n

    def load_config(path=\"config/config.yaml\"):
        with open(path, \"r\") as f:
            return yaml.safe_load(f)
    """,

    f"src/{project_name}/logging.py": """import logging\n
    import sys\n

    LOG_FORMAT = \"[[%(asctime)s] %(levelname)s in %(filename)s:%(lineno)d - %(message)s]\"\n
    logging.basicConfig(
    level=logging.INFO, 
    format=LOG_FORMAT
    handlers=[logging.StreamHandler(sys.stdout)])
    
    """,

    f"src/{project_name}/exceptions.py": """import sys\n
    class CustomException(Exception):
        def __init__(self, error_message: Exception, error_detail: sys):
            super().__init__(error_message)

            _,_,exc_tb = error_detail.exc_info()

            self.error_message = (
                f\"Error occurred in file: {exc_tb.tb_frame.f_code.co_filename}, "
                f"line: {exc_tb.tb_lineno}, "
                f"message: {error_message}\"
            )
        def __str__(self):
            return self.error_message
        """,
} 

for file in list_of_files:
    file_path = Path(file)
    file_dir, file_name = os.path.split(file_path)
    
    if file_dir != "":
        os.makedirs(file_dir, exist_ok=True)
        logging.info(f"Creating directory: {file_dir} for file: {file_name}")

    if (not file_path.exists() or file_path.stat().st_size==0):
            with open(file_path, 'w') as f:
                Content = TEMPLATES.get(file, "")
                f.write(Content)
            logging.info(f"Creating file Templates: {file_path}")
            
    else:
            logging.info(f"File already exists: {file_path}")
    