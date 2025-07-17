# Stokify
README.md – Stockify

📈 Stockify: Stock Price Prediction Web App

Stockify is a web application that enables users to predict the future prices of publicly listed stocks using deep learning models. Built with Django for the backend and Bootstrap for a responsive frontend, Stockify integrates a PyTorch-based LSTM model to forecast multi-day stock trends with precision and clarity.

🔗 Demo (Optional if hosted)
You can view a live demo (if hosted) or clone the repository and run it locally (instructions below).

📌 Features

* Predict future stock prices using LSTM neural networks
* Fetch historical stock data dynamically using Yahoo Finance
* Interactive Bootstrap-based UI for better accessibility
* Displays predictions in tables and visual charts (Matplotlib)
* Authentication system with Sign Up, Login, Logout functionality
* Fallback to demo data when API fails (rate limits, no internet)
* Mobile-friendly, responsive dashboard with search functionality

🛠️ Tech Stack

Frontend:

* HTML5, CSS3, Bootstrap 5
* Responsive UI with dynamic forms and charts

Backend:

* Python 3.12+
* Django 5.x
* PyTorch (LSTM model implementation)
* yfinance (for real-time stock data)
* Matplotlib & Pandas (for visualization and data processing)

📂 Folder Structure

stock\_predictor/
├── prediction/                # Django app (views, models, forms, templates)
│   ├── static/                # Static files (CSS, JS, demo data)
│   ├── templates/             # HTML templates
│   ├── utils/                 # Prediction logic (LSTM, data prep)
│   ├── views.py               # Authentication & prediction views
├── media/                     # For dynamically generated chart images
├── manage.py                  # Django entry point
├── requirements.txt           # Project dependencies

🚀 Getting Started

Step 1: Clone the Repository

git clone [https://github.com/LavishV/Stokify.git](https://github.com/LavishV/Stokify.git)
cd Stokify

Step 2: Set up Virtual Environment

python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

Step 3: Install Requirements

pip install -r requirements.txt

Step 4: Run the Server

python manage.py runserver

Step 5: Visit

Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) in your browser to access Stockify.

📷 Screenshots

(Screenshots can be added manually in your GitHub repo once you upload them in the /static/ or docs folder)

* Landing page
* Prediction chart
* Stock info & historical data
* Sign in / Sign up pages

🧠 LSTM Model Summary

The application uses a Long Short-Term Memory (LSTM) neural network implemented in PyTorch to learn from historical stock data. The model predicts multiple future steps (e.g., next 3 or 7 days) using sliding window techniques and normalized input sequences.

🛡️ Fallback Mode

If yfinance fails to fetch data due to rate limits or no internet, the app loads pre-generated demo CSVs and prediction chart from /static/prediction/demo\_data/ for offline testing.

🔐 Authentication

Users can register, login, and securely access the dashboard using Django's built-in authentication system. Navigation buttons dynamically change based on login state.

🙌 Author

Lavish Verma
CSE Department, Acropolis Institute of Technology and Research
Email: [lavishverma018@gmail.com](mailto:lavishverma018@gmail.com)
AICTE Student ID: STU6658d26acd53e1717097066

🔗 GitHub Profile: [https://github.com/LavishV](https://github.com/LavishV)
🔗 Project Repository: [https://github.com/LavishV/Stokify](https://github.com/LavishV/Stokify)

📌 Future Enhancements

* Add email verification on signup
* Deploy using Render/Heroku or Dockerize the project
* Add model evaluation metrics like RMSE & MAE on screen
* Option to export prediction results to CSV or PDF


