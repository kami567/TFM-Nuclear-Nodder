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




fecha_actual = datetime.now().strftime("%Y-%m-%d")
st.write("Fecha actual:", fecha_actual)


# URL del archivo por defecto en GitHub
url_tabla = "https://raw.githubusercontent.com/kami567/TFM-Nuclear-Nodder/main/Table_8.1_Nuclear_Energy_Overview.xlsx"
# URL del scaler en GitHub (corrigiendo la ruta)
url_scaler = "https://raw.githubusercontent.com/kami567/TFM-Nuclear-Nodder/main/scalers.pkl"
# Definir las URLs de los modelos en GitHub
#SARIMAX_Net_Summer_Capacity_MW (1).joblib
url_sarimax_mw = "https://raw.githubusercontent.com/kami567/TFM-Nuclear-Nodder/main/SARIMAX_Net_Summer_Capacity_MW.joblib"
url_sarimax_capacity = "https://raw.githubusercontent.com/kami567/TFM-Nuclear-Nodder/main/SARIMAX_Capacity_Factor_Percent.joblib"
url_ridge = "https://raw.githubusercontent.com/kami567/TFM-Nuclear-Nodder/main/modelo_ridge_NetGeneration.pkl"

# Directorios para guardar archivos localmente antes del commit en GitHub
ruta_sin_procesar = "Datos/Datos Sin Procesar"
ruta_procesados = "Datos/Datos Procesados"

# Preguntar al usuario si quiere cargar desde GitHub o subir manualmente
use_github = st.radio(
    "¿Cómo quieres cargar el archivo?",
    ("Usar archivo por defecto de GitHub", "Subir un archivo propio")
)

if use_github == "Usar archivo por defecto de GitHub":
    # Cargar el archivo desde la URL
    excel_data = pd.ExcelFile(url_tabla, engine="openpyxl")
    st.write("Archivo cargado desde GitHub")
else:
    # Permitir la subida del archivo
    uploaded_file = st.file_uploader("Sube un archivo Excel", type=["xlsx"])
    
    if uploaded_file is not None:
        excel_data = pd.ExcelFile(uploaded_file)
        st.write("Archivo cargado desde el usuario")
    else:
        st.warning("Por favor, sube un archivo para continuar.")
        st.stop()

# Mostrar las hojas disponibles y permitir al usuario elegir
sheet_name = st.selectbox("Selecciona la hoja de datos mensuales", excel_data.sheet_names)

# Cargar la hoja seleccionada
df = pd.read_excel(excel_data, sheet_name=sheet_name)

# Limpiar y procesar los datos
monthly_data_clean = df.dropna(how='all').reset_index(drop=True)

# Definir los encabezados de las columnas
column_headers = [
    "Date",
    "Total_Operable_Units",
    "Net_Summer_Capacity_MW",
    "Net_Generation_MWh",
    "Share_of_Electricity_Percent",
    "Capacity_Factor_Percent"
]

# Extraer datos válidos
monthly_data_extracted = monthly_data_clean.iloc[7:].reset_index(drop=True)
monthly_data_extracted.columns = column_headers

# Convertir la columna de fecha a formato datetime
monthly_data_extracted['Date'] = pd.to_datetime(monthly_data_extracted['Date'], errors='coerce')
monthly_data_extracted['Date'] = monthly_data_extracted['Date'].dt.to_period('M')

# Convertir las columnas numéricas
numeric_columns = column_headers[1:]
for col in numeric_columns:
    monthly_data_extracted[col] = pd.to_numeric(monthly_data_extracted[col], errors='coerce')

monthly_data_extracted.set_index('Date', inplace=True)

# Filtrar datos desde diciembre de 1994
monthly_data_extracted = monthly_data_extracted.loc['1994-12-01':]

# Calcular valores por unidad operativa
#Variable por unidad operativa
monthly_data_per_unit = monthly_data_extracted.drop(columns=['Total_Operable_Units']).div(monthly_data_extracted['Total_Operable_Units'], axis=0)
monthly_data_per_unit['Net_Summer_Capacity_MW'] = monthly_data_extracted['Net_Summer_Capacity_MW']
monthly_data_per_unit['Capacity_Factor_Percent'] = monthly_data_extracted['Capacity_Factor_Percent']
# Mostrar resultados
st.write(f"Mostrando datos de la hoja: {sheet_name} ya dividiendo por unidad operativa (Monthly Data Per Unit)")
st.write(monthly_data_per_unit.head(5))


# Botón para descargar el archivo original
with open(ruta_archivo_original, "rb") as f:
    st.download_button(
        label="Descargar archivo original",
        data=f,
        file_name=nombre_archivo_original,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# Botón de descarga en Streamlit del archivo procesado
output = io.BytesIO()
with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
    monthly_data_per_unit.to_excel(writer, sheet_name='Monthly Data Per Unit')
    writer.close()

st.download_button(
    label="Descargar archivo procesado",
    data=output.getvalue(),
    file_name=nombre_archivo_procesado,
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)



# Descargar el archivo desde GitHub
response = requests.get(url_scaler)
if response.status_code == 200:
    scaler = joblib.load(io.BytesIO(response.content))
    print("Scaler cargado correctamente desde GitHub.")
else:
    print("Error al descargar el scaler:", response.status_code)



# Función para cargar modelos desde GitHub
def cargar_modelo(url):
    response = requests.get(url)
    if response.status_code == 200:
        return joblib.load(io.BytesIO(response.content))
    else:
        print(f"Error al descargar {url}: {response.status_code}")
        return None

# Cargar los modelos
SARIMAX_Net_Summer_Capacity_MW = cargar_modelo(url_sarimax_mw)
SARIMAX_Capacity_Factor_Percent = cargar_modelo(url_sarimax_capacity)
modelo_ridge_NetGeneration = cargar_modelo(url_ridge)
scalers = cargar_modelo(url_scaler)  # Cargamos los scalers

# Verificar que monthly_data_per_unit exista antes de crear X e y
if 'monthly_data_per_unit' in locals():

    # Definir X (variables predictoras) e Y (variable objetivo)
    X = monthly_data_per_unit.drop(columns=['Net_Generation_MWh', 'Share_of_Electricity_Percent'])
    y = monthly_data_per_unit['Net_Generation_MWh']
    
    # Obtener el scaler correcto
    scaler_X = scalers.get("Net_Summer_Capacity_MW", None)

    if scaler_X is None:
        st.error("Error: No se encontró el scaler para 'Net_Summer_Capacity_MW'")
    else:
        # Verificar columnas esperadas vs actuales
        st.write("Las variables exógenas son:", X.columns.tolist())

        # Asegurar que las columnas coinciden
        for col in scaler_X.feature_names_in_:
            if col not in X.columns:
                X[col] = 0  # O usa np.nan para marcar los valores faltantes

        # Reordenar las columnas
        X = X[scaler_X.feature_names_in_]

        # Aplicar la transformación
        X_scaled = scaler_X.transform(X)
        
else:
    st.error("Error: `monthly_data_per_unit` no está definido. Asegúrate de procesar correctamente los datos.")






# Selección de años a predecir
años_SARIMAX = st.number_input("¿Cuántos años quieres predecir?", min_value=1, max_value=20, value=10, step=1, key = "Años SARIMAX")

# Interfaz en Streamlit
st.title("Predicción con SARIMAX de variables exógenas")

# Checkbox para decidir si predecir con SARIMAX
usar_sarimax = st.checkbox("¿Quieres predecir variables exógenas con SARIMAX?")

if usar_sarimax:


    # Convertir años a pasos (steps) de predicción
    steps = años_SARIMAX * 12  # 12 meses por año

    #SARIMAX_Net_Summer_Capacity_MW
    #SARIMAX_Net_Summer_Capacity_MW
    # Realizar predicciones con SARIMAX
    pred_sarimax_Net_Summer = SARIMAX_Net_Summer_Capacity_MW.forecast(steps=steps)
    pred_sarimax_Capacity = SARIMAX_Capacity_Factor_Percent.forecast(steps=steps)


    # Inversa la transformación
    Net_summer_capacity_predict = scalers['Net_Summer_Capacity_MW'].inverse_transform(pred_sarimax_Net_Summer.values.reshape(-1, 1))
    Capacity_factor_predict = scalers['Capacity_Factor_Percent'].inverse_transform(pred_sarimax_Capacity.values.reshape(-1, 1))
    # Generar fechas futuras
    future_dates = pd.date_range(start=pd.Timestamp.today().replace(day=1), periods=len(Net_summer_capacity_predict), freq="MS").to_period('M')

    # Convertir a DataFrame con índice de fechas
    df_predictions = pd.DataFrame({
        "Fecha": future_dates,
        "Predicción Net Summer Capacity MW": Net_summer_capacity_predict.flatten(),  # Asegurar 1D
        "Predicción Capacity Factor Percent": Capacity_factor_predict.flatten()  # Asegurar 1D
    })

    #Definimos el historico de datos
    historico_x = monthly_data_per_unit[["Net_Summer_Capacity_MW", "Capacity_Factor_Percent"]].copy()

    # 🔹 Asegurar que el índice del histórico es DatetimeIndex
    if not isinstance(historico_x.index, pd.DatetimeIndex):
        historico_x.index = historico_x.index.to_timestamp()

    # Definir la columna Fecha como índice
    df_predictions.set_index("Fecha", inplace=True)
    df_predictions.index =df_predictions.index.to_timestamp()

    # 🔍 Depuración antes de mostrar
    st.write("Primeras filas del DataFrame de predicciones:")
    st.dataframe(df_predictions.head())



    # 🔹 Crear subgráficos con dos filas
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=[
        "Predicción Net Summer Capacity MW",
        "Predicción Capacity Factor Percent"
    ])

    # 🔹 Primer gráfico (Net Summer Capacity MW)
    fig.add_trace(go.Scatter(
        x=df_predictions.index, 
        y=df_predictions["Predicción Net Summer Capacity MW"], 
        mode='lines', 
        name="Predicción", 
        line=dict(color="gold")
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=historico_x.index, 
        y=historico_x["Net_Summer_Capacity_MW"], 
        mode='lines', 
        name="Histórico", 
        line=dict(color="green")
    ), row=1, col=1)

    # 🔹 Segundo gráfico (Capacity Factor Percent)
    fig.add_trace(go.Scatter(
        x=df_predictions.index, 
        y=df_predictions["Predicción Capacity Factor Percent"], 
        mode='lines', 
        name="Predicción", 
        line=dict(color="gold")
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=historico_x.index, 
        y=historico_x["Capacity_Factor_Percent"], 
        mode='lines', 
        name="Histórico", 
        line=dict(color="darkgreen")
    ), row=2, col=1)

    # 🔹 Configurar el diseño del gráfico
    fig.update_layout(
        height=800, width=1000,  # Tamaño personalizado
        title_text="Predicciones de Energía Nuclear",
        showlegend=True,  # Mostrar leyenda en ambos gráficos
        legend=dict(
            x=0,  # Posicionar a la izquierda
            y=0.5,  # Posicionar arriba
            bgcolor="rgba(0,0,0,0)"  # Fondo transparente
        ),
        template="plotly_dark"  # Modo oscuro
    )
    
    # 🔹 Mostrar el gráfico en Streamlit
    st.plotly_chart(fig)

   
    



    # Botón para descargar resultados como CSV
    csv = df_predictions.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Descargar predicciones como CSV",
        data=csv,
        file_name="predicciones_sarimax.csv",
        mime="text/csv"
    )
else:
    st.warning("No se realiza predicción con SARIMAX.")
# Interfaz en Streamlit
st.title("Predicción con Ridge")
# Selección de años a predecir
años_Ridge = st.number_input("¿Cuántos años quieres predecir?", min_value=1, max_value=20, value=10, step=1, key = "Años Ridge")

# Checkbox para decidir si predecir con Ridge
usar_ridge = st.checkbox("¿Quieres predecir Net Generation MWh con Ridge?")

if usar_ridge:
    # Definir Y_escalado correctamente
    Y_escalado = y  # Asegurar que y ha sido escalado previamente
    
    # Convertir años a pasos (steps) de predicción
    steps = años_Ridge * 12  # 12 meses por año

    # Mostrar configuración del modelo
    st.write("### 🔍 Configuración del modelo Ridge")
    st.write(f"Window size original del modelo: {modelo_ridge_NetGeneration.window_size}")

    # Ajustar window_size si hay menos datos de los esperados
    window_size = modelo_ridge_NetGeneration.window_size
    if len(Y_escalado) < window_size:
        window_size = len(Y_escalado)  # Ajustamos al máximo posible
        st.warning(f"⚠️ Window size ajustado a {window_size} porque no hay suficientes datos.")

    # Obtener la última ventana de datos
    last_window = Y_escalado[-modelo_ridge_NetGeneration.window_size:]
    last_window.index = last_window.index.to_timestamp()
    last_window = last_window.asfreq('MS')

    #Usamos el modelo para predecir los años seleccionados
    pred_ridge_NetGeneration = modelo_ridge_NetGeneration.predict(steps=steps, last_window=last_window)

    #Cambiamos de serie a DF
    df_pred = pred_ridge_NetGeneration.rename_axis("Fecha").rename("Predicción Net Generation MWh").to_frame()

    # Mostrar predicciones en tabla
    st.write("###Predicciones Net Generation MWh")
    st.dataframe(df_pred.style.format("{:.2f}"))
    #st.dataframe(df_predictions)

    # Graficar la serie original y la predicción suavizadas
    st.write("###Comparación entre Datos Originales y Predicción")

    # Asegurar que Y_escalado y pred_ridge_NetGeneration sean Series numéricas
    Y_suavizado = Y_escalado.rolling(window=4, min_periods=1).mean()
    pred_suavizado = pred_ridge_NetGeneration.rolling(window=4, min_periods=1).mean()


    # ✅ Eliminar valores NaN
    Y_suavizado = Y_suavizado.dropna()
    pred_suavizado = pred_suavizado.dropna()

    # ✅ Asegurar que el índice es un DatetimeIndex
    if not isinstance(Y_suavizado.index, pd.DatetimeIndex):
        Y_suavizado.index = Y_suavizado.index.to_timestamp()

    if not isinstance(pred_suavizado.index, pd.DatetimeIndex):
        pred_suavizado.index = pred_suavizado.index.to_timestamp()

    # ✅ Asegurar que los datos sean numéricos
    Y_suavizado = Y_suavizado.astype(float)
    pred_suavizado = pred_suavizado.astype(float)


   

    # 🔹 Crear la figura en Plotly
    fig = go.Figure()

    # 🔹 Agregar la serie original
    fig.add_trace(go.Scatter(
        x=Y_suavizado.index, 
        y=Y_suavizado, 
        mode='lines', 
        name="Datos Originales", 
        line=dict(color="limegreen")
    ))

    # 🔹 Agregar la predicción
    fig.add_trace(go.Scatter(
        x=pred_suavizado.index, 
        y=pred_suavizado, 
        mode='lines', 
        name="Predicción Ridge", 
        line=dict(color="gold")
    ))

    # 🔹 Configurar el diseño del gráfico
    fig.update_layout(
        title="Comparación entre Datos Originales y Predicción Ridge (Suavizado)",
        xaxis_title="Fecha",
        yaxis_title="MWh",
        legend=dict(
            x=0,  # Posición en la izquierda
            y=1.1,  # Posición arriba del gráfico
            font=dict(color="white"),  # Letras blancas en la leyenda
            bgcolor="rgba(0,0,0,0)"  # Fondo transparente para la leyenda
        ),
        template="plotly_dark",  # Fondo negro con letras blancas
        width=800, height=500  # Ajuste de tamaño para mejor visualización
        )

    # 🔹 Mostrar la gráfica en Streamlit
    st.plotly_chart(fig)


    # Botón para descargar predicciones en CSV
    csv = df_predictions.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Descargar predicciones como CSV",
        data=csv,
        file_name="predicciones_ridge.csv",
        mime="text/csv"
    )

    # Preguntar si el usuario quiere guardar los datos en GitHub
    guardar_github = st.checkbox("¿Quieres guardar las predicciones en GitHub?")
    if guardar_github:
        import os

        # Definir ruta de guardado
        ruta_base = "Datos/Predicciones Ridge"
        ruta_subcarpeta = os.path.join(ruta_base, pd.Timestamp.today().strftime("%Y-%m-%d"))

        # Asegurar que la subcarpeta existe
        os.makedirs(ruta_subcarpeta, exist_ok=True)

        # Guardar el archivo en la carpeta
        ruta_archivo = os.path.join(ruta_subcarpeta, "predicciones_ridge.csv")
        df_predictions.to_csv(ruta_archivo, index=False)

        # Subir a GitHub con Git
        os.system(f"git add {ruta_archivo}")
        os.system(f'git commit -m "Añadidas predicciones Ridge del {pd.Timestamp.today().strftime("%Y-%m-%d")}"')
        os.system("git push origin main")

        st.success(f"📤 Predicciones guardadas en GitHub: {ruta_archivo}")

else:
    st.warning("No se realizó predicción con Ridge.")
