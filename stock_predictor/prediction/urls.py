from django.urls import path
from .views import predict_stock
from . import views

urlpatterns = [
    path('', predict_stock, name='predict_stock'),
    path('signup/', views.signup_view, name='signup'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
]
