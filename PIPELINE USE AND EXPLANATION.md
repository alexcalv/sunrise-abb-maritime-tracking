# HOW TO USE THE PROPOSED PIPELINE

## REQUIREMENTS:
- Python 3.7.9 and pip 20.1.1
- Memurai Developer Edition  
  [Download Memurai](https://www.memurai.com/get-memurai?version=windows-redis)

---

## (Recommended) STEP 0 – VIRTUAL ENVIRONMENT

### Creation:
```bash
python -m venv SUNRISE
```

### How to activate the virtual environment:

- **Git Bash:**
  ```bash
  source SUNRISE/Scripts/activate
  ```
  or
  ```bash
  . SUNRISE/Scripts/activate
  ```

- **Command Prompt:**
  ```bash
  SUNRISE\Scripts\activate.bat
  ```

- **PowerShell:**
  ```bash
  . .\SUNRISE\Scripts\Activate.ps1
  ```

---

## STEP 1 – RUNNING THE CODE

### Terminal 1:
```bash
pip install -r requirements.txt
celery -A broker.celery_app worker --loglevel=info --concurrency=1 --pool=solo
```

### Terminal 2:
```bash
python run_pipeline.py --mode test --input ../data/sample.jpg
```

---

## PIPELINE EXPLANATION

- **run_pipeline.py** – Main file that starts the pipeline.
- **config/pipeline.yaml** – Contains base settings.
- **broker/celery_app.py** – Celery app and inference task for detect_batch. New tasks (e.g., tracking, ReID) should be added here.
- **ingestion/ingestor.py** – Reads datasets and dispatches queue tasks. (for now it differentiates MVTD and SMD as image and video outputs)
- **output/aggregator.py** – Merges JSON results into a summary file.
- **data/** – Place for datasets.
- **outputs/** – Contains results, annotated frames, and summary.

The system is distributed - a message broker (Redis, running locally via Memurai) is inbetween the data ingestion step and the actual inference workers, meaning that multiple worker processes can pull and process batches of frames in parallel rather than one at a time. 

Detection results are saved as JSON files per frame and merged into a single summary.json for review, alongside annotated images with bounding boxes drawn. 

The specific model is not trained yet! It should be swappable by changing the configuration file.
