# Blood Group Detection from Fingerprint

This project uses deep learning to predict blood groups from fingerprint images. It implements a Convolutional Neural Network (CNN) model and provides a web interface for easy interaction.

## Features

- Blood group prediction from fingerprint images
- Supports 8 blood groups: A+, A-, B+, B-, AB+, AB-, O+, O-
- Secure login system
- User-friendly web interface
- Model performance metrics and visualization

## Project Structure

```
blood_group_fingerprint/
├── app.py                 # Flask web application
├── train_model.py         # Model training script
├── requirements.txt       # Project dependencies
├── README.md             # Project documentation
│
├── /dataset/             # Training dataset directory
│   ├── A+/              # Fingerprint images for A+
│   ├── A-/              # and so on...
│   └── README.md        # Dataset guidelines
│
├── /saved_model/         # Directory for trained model
│   └── fingerprint_model.h5
│
├── /static/              # Static files
│   ├── /uploads/        # Uploaded images
│   └── confusion_matrix.png
│
└── /templates/           # HTML templates
    ├── login.html
    ├── upload.html
    ├── result.html
    └── performance.html
```

## Requirements

- Python 3.7 or higher
- TensorFlow 2.x
- Flask
- OpenCV
- NumPy
- Scikit-learn
- Other dependencies in requirements.txt

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd blood_group_fingerprint
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare the dataset:
   - Place fingerprint images in their respective blood group folders under the `dataset` directory
   - Follow the guidelines in `dataset/README.md`

2. Train the model (optional):
```bash
python train_model.py
```

3. Run the web application:
```bash
python app.py
```

4. Access the application:
   - Open a web browser and go to `http://localhost:5000`
   - Login credentials:
     - Username: admin
     - Password: admin

## Model Architecture

The CNN model consists of:
- 3 Convolutional layers with ReLU activation
- MaxPooling layers for dimensionality reduction
- Dropout for regularization
- Dense layers for classification

## Performance Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix visualization

## Security Notes

- Change the default admin credentials in a production environment
- The secret key in app.py should be changed
- Implement additional security measures for production use

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.