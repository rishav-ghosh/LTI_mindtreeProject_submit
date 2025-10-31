🧠 Brain Scan AI — Alzheimer’s Disease classification

This project focuses on detecting and classifying Alzheimer’s disease from MRI brain scans using deep learning pre-trained models automated MLOps tools.
It integrates a pre-trained Convolutional Neural Network (CNN) model with a Flask web application to provide real-time predictions and automated deployment pipelines

The system classifies MRI images into four stages of Alzheimer’s disease:

🟢 Non-Demented

🟡 Very Mild Dementia

🟠 Mild Dementia

🔴 Moderate Dementia

The model uses transfer learning on pre-trained CNN architectures to achieve high accuracy.
For deployment, the project implements Git, DVC, Jenkins, and Docker for continuous integration and versioned model management.

🧰 Tech Stack
Component	Technology Used
Programming Language	Python
Deep Learning Framework	TensorFlow / Keras
Web Framework	Flask
MLOps Tools	Git, DVC, Jenkins, Docker
Storage	Google Drive (Model Storage via gdown)
Frontend	HTML, CSS, JavaScript
