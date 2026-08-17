# AI Industrial Surface Defect Detection

A computer vision project for detecting common surface defects in hot-rolled steel using deep learning.

I built this project on the **NEU-DET dataset**, which contains six types of steel surface defects:

* Crazing
* Inclusion
* Patches
* Pitted Surface
* Rolled-in Scale
* Scratches

The main goal was not just to train a CNN, but to compare different approaches, understand their strengths and weaknesses, and build something that can actually be used through a simple web interface.

## What I built

I first trained a custom CNN as a baseline. It showed strong training performance but poor validation performance, which made the overfitting problem clear.

I then used transfer learning with two pretrained models:

* ResNet50
* EfficientNetB0

ResNet50 performed the best and was selected as the final model.

| Model          |   Accuracy |   F1 Score |
| -------------- | ---------: | ---------: |
| Custom CNN     |     ~16.7% |        ~5% |
| EfficientNetB0 |     97.78% |     97.75% |
| **ResNet50**   | **99.72%** | **99.72%** |

The final ResNet50 achieved **99.72% accuracy on the 360-image validation set**.

## Explainability

I also used **Grad-CAM** to understand where the model was looking when making a prediction.

For example, for a scratch image, the Grad-CAM heatmap highlighted the region containing the surface defect instead of focusing on an unrelated part of the image.

This makes the prediction easier to interpret and gives some insight into whether the model is using meaningful visual features.

## Streamlit App

The final model is wrapped in a Streamlit application.

The user can:

1. Upload a steel surface image.
2. Get the predicted defect.
3. See the prediction confidence.
4. View probabilities for all six defect classes.
5. See a Grad-CAM visualization explaining the prediction.

## Project Structure

```text
AI_Industrial_Defect_Detection/
│
├── 01_EDA.ipynb
├── 02_Preprocessing.ipynb
├── 03_Model_Training.ipynb
├── 04_Explainability_GradCAM.ipynb
├── app.py
├── resnet50.keras
├── efficientnetb0.keras
├── class_mapping.json
├── requirements.txt
└── README.md
```

The original NEU-DET dataset is not included in this repository.

## Tech Used

Python, TensorFlow/Keras, ResNet50, EfficientNetB0, CNN, Grad-CAM, NumPy, Matplotlib, Scikit-learn and Streamlit.

## Running the App

Install the dependencies:

```bash
pip install -r requirements.txt
```

Then run:

```bash
streamlit run app.py
```

The app will open in your browser.

## What I learned from this project

One of the main takeaways from this project was that a more complex model is not automatically a better model.

The custom CNN was a useful baseline, but with a relatively small dataset it struggled to generalize. Transfer learning gave much better results, and comparing ResNet50 with EfficientNetB0 helped me choose the final model based on actual validation performance rather than assumptions.

This project also gave me practical experience with image preprocessing, transfer learning, model evaluation, explainability and deployment.
