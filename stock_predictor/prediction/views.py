from django.shortcuts import render
from .forms import StockForm
from .utils.predict import run_prediction
import pandas as pd
import os
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib import messages

def predict_stock(request):
    if request.method == 'POST':
        form = StockForm(request.POST)
        if form.is_valid():
            ticker = form.cleaned_data['ticker']
            start_date = form.cleaned_data['start_date']
            end_date = form.cleaned_data['end_date']

            try:
                # Try to get live prediction
                df_future, df_results, image_path = run_prediction(ticker, start_date, end_date)
            except Exception as e:
                # Fallback to demo files
                df_results = pd.read_csv('prediction/static/prediction/demo_data/df_results.csv')
                df_future = pd.read_csv('prediction/static/prediction/demo_data/df_future.csv')
                image_path = 'prediction/demo_data/prediction_chart.png'
                ticker = 'AAPL'  # update for display
                print(f"⚠️ Falling back to demo data due to: {e}")
            
            # Prepare the results for display
            df_future = df_future.sort_values(by="Date", ascending=False)
            table_future = df_future.to_html(classes="table table-bordered", index=False)
            df_results = df_results.sort_values(by="Date", ascending=True)
            table_results = df_results.to_html(classes="table table-striped", index=False)

            return render(request, 'prediction/results.html', {
                'form': form,
                'image_path': image_path,
                'table_results': table_results,
                'table_future': table_future,
                'ticker': ticker
            })
    else:
        form = StockForm()

    return render(request, 'prediction/results.html', {'form': form})


# Authentication views
def signup_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        email = request.POST['email']
        password1 = request.POST['password1']
        password2 = request.POST['password2']

        if password1 != password2:
            messages.error(request, "Passwords do not match.")
            return render(request, 'prediction/signup.html')

        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists.")
            return render(request, 'prediction/signup.html')

        user = User.objects.create_user(username=username, email=email, password=password1)
        login(request, user)
        return redirect('predict_stock')  # ✅ CORRECT redirect

    return render(request, 'prediction/signup.html')



def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']

        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('predict_stock')
        else:
            messages.error(request, "Invalid username or password.")
            return render(request, 'prediction/login.html')

    return render(request, 'prediction/login.html')


def logout_view(request):
    logout(request)
    return redirect('predict_stock')

