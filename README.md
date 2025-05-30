# Lead Premium Subscription Probability API

A simple Flask web application that accepts feature data points for a lead and returns the probability of that lead having a premium subscription.

## Features

- **Single Endpoint**: `/predict` accepts a JSON payload of feature values.
- **Pre-trained Model**: Deserialized with `joblib` from a saved pipeline or XGBoost model.
- **JSON Responses**: Returns the probability of the positive class (premium subscription).
- **Health Check**: `/` endpoint to verify the service is running.

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/lead-probability-api.git
   cd lead-probability-api
Create & activate a virtual environment

bash
Always show details

Copy
python -m venv venv
# On Windows (PowerShell)
.\\venv\\Scripts\\Activate.ps1
# On macOS/Linux
source venv/bin/activate
Install dependencies

bash
Always show details

Copy
pip install -r requirements.txt
Place your trained model

Ensure your serialized model file (e.g. xgb_f1_score_optimized_pipeline.joblib) lives in the project root.

Usage
Run the Flask app

bash
Always show details

Copy
python app.py
By default, it listens on http://0.0.0.0:5000.

Health check

bash
Always show details

Copy
curl http://localhost:5000/
Response:

json
Always show details

Copy
{ "status": "ok" }
Predict endpoint

URL: POST /predict

Payload:

json
Always show details

Copy
{
  "features": [f1, f2, ..., fn]
}
Response:

json
Always show details

Copy
{
  "probability": 0.73
}
Example:

bash
Always show details

Copy
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"features":[0.5,1.2,0.3,...]}'
Deployment
This app can be deployed to services like Render, Heroku, or AWS Elastic Beanstalk.

Render.com

Push to GitHub.

Create a new Web Service on Render.

Set Build Command: pip install -r requirements.txt

Set Start Command: gunicorn app:app

Ensure runtime.txt pins your Python version.

Requirements
Python 3.8+

Flask

XGBoost

Joblib

NumPy

Other dependencies are listed in requirements.txt.

License
MIT License

"""

Write README.md and display its content
with open('README.md', 'w') as f:
f.write(readme_content)

print(readme_content)

Always show details

Copy
STDOUT/STDERR
# Lead Premium Subscription Probability API

A simple Flask web application that accepts feature data points for a lead and returns the probability of that lead having a premium subscription.

## Features

- **Single Endpoint**: `/predict` accepts a JSON payload of feature values.
- **Pre-trained Model**: Deserialized with `joblib` from a saved pipeline or XGBoost model.
- **JSON Responses**: Returns the probability of the positive class (premium subscription).
- **Health Check**: `/` endpoint to verify the service is running.

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/lead-probability-api.git
   cd lead-probability-api
   ```

2. **Create & activate a virtual environment**
   ```bash
   python -m venv venv
   # On Windows (PowerShell)
   .\venv\Scripts\Activate.ps1
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Place your trained model**
   - Ensure your serialized model file (e.g. `xgb_f1_score_optimized_pipeline.joblib`) lives in the project root.

## Usage

1. **Run the Flask app**
   ```bash
   python app.py
   ```
   By default, it listens on `http://0.0.0.0:5000`.

2. **Health check**
   ```bash
   curl http://localhost:5000/
   ```
   Response:
   ```json
   { "status": "ok" }
   ```

3. **Predict endpoint**
   - **URL**: `POST /predict`
   - **Payload**:
     ```json
     {
       "features": [f1, f2, ..., fn]
     }
     ```
   - **Response**:
     ```json
     {
       "probability": 0.73
     }
     ```

   Example:
   ```bash
   curl -X POST http://localhost:5000/predict         -H "Content-Type: application/json"         -d '{"features":[0.5,1.2,0.3,...]}'
   ```

## Deployment

This app can be deployed to services like **Render**, **Heroku**, or **AWS Elastic Beanstalk**.

- **Render.com**
  1. Push to GitHub.
  2. Create a new Web Service on Render.
  3. Set **Build Command**: `pip install -r requirements.txt`
  4. Set **Start Command**: `gunicorn app:app`
  5. Ensure `runtime.txt` pins your Python version.

## Requirements

- Python 3.8+
- Flask
- XGBoost
- Joblib
- NumPy

Other dependencies are listed in `requirements.txt`.

## License

MIT License
