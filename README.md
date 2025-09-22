# ❤️ Heart Attack Prediction App with Streamlit

## 📝 Description
This is a web application built with Streamlit that predicts the likelihood of a heart attack based on user-provided health parameters. It uses a pre-trained machine learning model to make predictions.

## ✨ Features
- 🖥️ User-friendly interface for inputting health data
- ❤️ Prediction of heart disease risk
- 🩺 Supports inputs like age, sex, chest pain type, resting blood pressure, cholesterol, etc.
- ✅ Displays results with success or error messages

## ⚙️ Installation
1. Ensure you have Python installed (version 3.7 or higher).
2. Install the required dependencies:
   ```
   pip install streamlit numpy
   ```
   Note: pickle is part of Python's standard library, so no need to install it separately.

## 🚀 Usage
1. Run the main prediction app:
   ```
   streamlit run app2.py
   ```
2. Open the provided URL in your browser.
3. Fill in the form with your health data and click "Predict" to get the result.

There is also a demo app (app.py) that showcases various Streamlit components:
   ```
   streamlit run app.py
   ```

## 📦 Dependencies
- streamlit
- numpy
- pickle (built-in)

## 🧠 Model
The app uses a pre-trained model saved in `model.sav`. Ensure this file is in the same directory as the scripts.

## 🤝 Contributing
Feel free to contribute by improving the model, UI, or adding features.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
