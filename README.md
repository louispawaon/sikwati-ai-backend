# **Sikwati-AI API**
This is a Flask-based API for classifying cacao pod images using an EfficientNetB0 model. The API provides an endpoint to classify an uploaded image and returns the predicted class along with confidence scores.

## **Getting Started**
### **Prerequisites**
- Python 3.x
- Flask
- Flask-CORS
- TensorFlow
- scikit-learn
- scikit-image

### **Installation**
1. Clone the repository
```bash
git clone https://github.com/louispawaon/sikwati-ai-backend.git
cd sikwati-ai-backend
```
2. Create a virtual environment (*optional but recommended*)
```bash
python -m venv venv
source venv/bin/activate
```
3. Install Python dependencies
```bash
pip install -r requirements.txt
```
4. Run the application
```bash
gunicorn app:app #For Production
flask run --debug #For Debugging Mode
```
5. (*Optional*) Expose the local server using Ngrok
```bash
ngrok http 5000
```
Ngrok will provide a public URL (e.g., https://your-ngrok-subdomain.ngrok.io) that you can use to access your API from anywhere.

## **Usage**

### **Home Endpoint**
- Endpoint: `/`
- Description: Welcome to Sikwati AI API
- Method: `GET`
- Response:
```json
{
  "message": "Welcome to the Cacao Pod Classification API"
}
```

### **Classify Image Endpoint**
- Endpoint: `/api/classify_image`
- Description: Classify an uploaded cacao pod image
- Method: `POST`
- Request Parameters:
    - `image` (*type*: file, *required*: true) - The cacao pod image to be classified
- Responses:
  - 200 OK
    ```json
    {
        "class": "predicted_class",
        "confidence": 0.85
    }
    ```
  - 400 Bad Request
    ```json
    {
        "error": "No file part"
    }
    ```
    ```json
    {
        "error": "No selected file"
    }
    ```
    ```json
    {
        "error": "The image may not contain a cacao pod"
    }
    ```

## **Model Details**
- The classification model is based on the EfficientNetB0 CNN Architecture

## **Author**
[Louis Miguel Pawaon](https://twitter.com/miggy_pawaon)