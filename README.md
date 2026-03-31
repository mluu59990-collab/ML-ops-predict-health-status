
## Demo


Watch the demo video here: [Demo Video](/Users/Documents/ML_OPS/DAY11/Demo_img/ml.mp4)
This project includes a Flask-based web application for fitness prediction using a trained machine learning pipeline.

### Demo Overview
Users can:
- enter health-related features such as age, height, weight, heart rate, and sleep hours
- submit the form through a clean web interface
- receive a predicted fitness label from the trained model

### Interface Preview
| Home Page | Input Form | Prediction Result |
|----------|------------|------------------|
| ![Home](/Users/Documents/ML_OPS/DAY11/Demo_img/img1.jpg) | ![Form](/Users/Documents/ML_OPS/DAY11/Demo_img/img2.jpg) | ![Result](/Users/Documents/ML_OPS/DAY11/Demo_img/img3.jpg) |

### Run the Demo Locally
```bash
git clone https://github.com/mluu59990-collab/ML-ops-predict-health-status.git
cd ML-ops-predict-health-status
pip install -r requirements.txt
python app.py