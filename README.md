# Librerías básicas
import numpy as np
import pandas as pd
import io
import joblib
import requests
from datetime import datetime
import os

# Visualización
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Análisis y estadísticas
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy import stats
from scipy.stats import boxcox

# Machine Learning y preprocesamiento
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Forecasting con Skforecast
import skforecast
from skforecast.datasets import fetch_dataset
from skforecast.recursive import ForecasterRecursive
from skforecast.direct import ForecasterDirect
from skforecast.model_selection import TimeSeriesFold, grid_search_forecaster, backtesting_forecaster
from skforecast.preprocessing import RollingFeatures
from skforecast.utils import save_forecaster, load_forecaster

# Interpretabilidad de modelos
import shap
import joblib
