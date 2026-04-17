# Drone Disease Detection: Field Patch Classifier

This repository contains the patch classification module for my Project on Drone-based Crop Disease Detection. It extracts patches from drone imagery and classifies them into `field` and `non_field` categories, written into a csv file.

## Project Structure
* **`Data Prep/`**: Scripts for extracting and generating patches from stiched drone imagery.
  * `patch generator.py`: Generates patches from images.
  * `selectNonFields.py`: As this is for training the AI, this script creates a window in which we can manually classify the field patches, there is option for all images, that are placed in a 4 by 3 grid, to be either field or non-field, or you could check and select image one by one to be a field patch or not.
* **`Classification/`**: Contains model training, inference scripts, and saved weights.
  * `src/`: Source code for training the models.
  * `field_model_pytorch.pth`: Trained PyTorch model.
  * `pytorch_inference_results.csv` & `pytorch_confusion_matrix.png`: Evaluation metrics and results.

## Requirements
Install the required dependencies using:
```bash
pip install -r Classification/requirements.txt