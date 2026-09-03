"""
Dashboard Interactivo de Análisis Exploratorio de Datos (EDA)
Aplicación genérica para explorar cualquier archivo CSV
Basado en el Capítulo 1: "Exploratory Data Analysis" de Practical Statistics for Data Scientists
"""

import io
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.stats import trim_mean
from ucimlrepo import fetch_ucirepo

warnings.filterwarnings('ignore')

# Configurar el tema y estilo
st.set_page_config(page_title="Dashboard EDA", layout="wide", initial_sidebar_state="expanded")
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def detect_variable_type(series):
    if pd.api.types.is_numeric_dtype(series):
        return 'Cuantitativa'
    
    is_categorical = (
        pd.api.types.is_object_dtype(series) or 
        pd.api.types.is_string_dtype(series) or 
        isinstance(series.dtype, pd.CategoricalDtype) or 
        pd.api.types.is_bool_dtype(series)
    )
    
    if is_categorical:
        total_count = series.count()
        if total_count == 0:
            return 'Otro'
        
        unique_count = series.nunique()
        if unique_count <= 20 or (unique_count / total_count) < 0.5:
            return 'Cualitativa'
        else:
            return 'Mixta/Texto'
    else:
        return 'Otro'

def classify_variables(df):
    variable_types = {}
    for col in df.columns:
        var_type = detect_variable_type(df[col])
        if var_type not in variable_types:
            variable_types[var_type] = []
        variable_types[var_type].append(col)
    return variable_types

def compute_location_stats(series):
    stats = series.agg(['mean', 'median', 'min', 'max'])
    return {
        'Media': stats['mean'],
        'Media Recortada (10%)': trim_mean(series, 0.1),
        'Mediana': stats['median'],
        'Min': stats['min'],
        'Max': stats['max'],
    }


def compute_variability_stats(series):
    stats = series.agg(['std', 'var', 'mean', 'median', 'min', 'max'])
    quantiles = series.quantile([0.25, 0.75])
    
    cv_value = np.nan
    if stats['mean'] != 0 and not pd.isna(stats['std']):
        cv_value = (stats['std'] / abs(stats['mean'])) * 100
    
    return {
        'Desv. Est.': stats['std'],
        'Varianza': stats['var'],
        'MAD': (series - stats['median']).abs().median(),
        'Rango': stats['max'] - stats['min'],
        'RIC (IQR)': quantiles[0.75] - quantiles[0.25],
        'CV (%)': cv_value,
    }


def compute_distribution_stats(series):
    """
    Calcula estadísticas de distribución
    """
    percentiles = series.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        'P5%': percentiles[0.05],
        'P25%': percentiles[0.25],
        'P50% (Mediana)': percentiles[0.5],
        'P75%': percentiles[0.75],
        'P95%': percentiles[0.95],
        'Asimetría': series.skew(),
        'Curtosis': series.kurtosis(),
    }


def compute_weighted_stats(df, val_col, weight_col):
    mask = df[val_col].notna() & df[weight_col].notna() & (df[weight_col] > 0)
    
    if not mask.any():
        return np.nan, np.nan
        
    values = df.loc[mask, val_col].values
    weights = df.loc[mask, weight_col].values
    
    weights_sum = weights.sum()
    if weights_sum <= 0:
        return np.nan, np.nan
    
    weighted_mean = np.average(values, weights=weights)
    
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]
    
    cum_weights = np.cumsum(sorted_weights)
    cutoff = weights_sum / 2.0
    
    if cum_weights[-1] == 0:
        return weighted_mean, np.nan
    
    idx = np.searchsorted(cum_weights, cutoff)
    idx = min(idx, len(sorted_values) - 1)
    
    if idx > 0 and idx < len(cum_weights) and cum_weights[idx] != cutoff:
        w1, w2 = cum_weights[idx - 1], cum_weights[idx]
        weight_diff = w2 - w1
        if weight_diff != 0:
            weighted_median = (sorted_values[idx - 1] * (w2 - cutoff) + sorted_values[idx] * (cutoff - w1)) / weight_diff
        else:
            weighted_median = sorted_values[idx]
    else:
        weighted_median = sorted_values[idx]
    
    return weighted_mean, weighted_median


def plot_histogram(series, bins=30):
    """
    Genera visualización de distribución de frecuencias mediante histograma.
    
    Parámetros:
        series: pd.Series - Variable numérica a visualizar
        bins: int - Número de intervalos (por defecto 30)
    
    Retorna:
        matplotlib.figure.Figure - Objeto figura con histograma renderizado
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    series.plot.hist(bins=bins, ax=ax, edgecolor='black', alpha=0.7)
    ax.set_xlabel(series.name)
    ax.set_ylabel('Frecuencia')
    ax.set_title(f'Histograma: {series.name}')
    ax.grid(axis='y', alpha=0.3)
    return fig


def plot_histogram_density(series, bins=20):
    """
    Superpone histograma normalizado con estimador de densidad kernel (KDE).
    
    Útil para identificar multimodalidad y características de la distribución
    más allá de la discretización inherente a bins fijos.
    
    Parámetros:
        series: pd.Series - Variable numérica
        bins: int - Número de intervalos (por defecto 20)
    
    Retorna:
        matplotlib.figure.Figure - Figura con histograma + curva de densidad KDE
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    series.plot.hist(density=True, bins=bins, ax=ax, alpha=0.6, edgecolor='black')
    series.plot.density(ax=ax, linewidth=2, color='#FFB7B2')
    ax.set_xlabel(series.name)
    ax.set_ylabel('Densidad')
    ax.set_title(f'Histograma con Densidad: {series.name}')
    ax.grid(axis='y', alpha=0.3)
    return fig

def plot_boxplot(series):
    fig = px.box(
        series,
        y=series.name,
        points="all",                 # Muestra todos los puntos al costado de la caja
        template="plotly_white",       # Fondo limpio y profesional
        title=f"Distribución de <b>{series.name}</b>"
    )

    fig.update_traces(
        boxmean=True,                  # Añade línea punteada para la media
        marker=dict(size=4, opacity=0.6), # Estiliza los puntos individuales
        line=dict(width=1.5),          # Contornos más nítidos en la caja
        jitter=0.2                     # Dispersión sutil de los puntos
    )

    fig.update_layout(
        title_x=0.5,
        showlegend=False,
        yaxis=dict(
            title=series.name,
            showgrid=True,
            gridcolor="#E5E7EB",
            zeroline=False
        ),
        xaxis=dict(showticklabels=False),
        margin=dict(l=50, r=50, t=60, b=40)
    )

    return fig

def plot_countplot(series, top_n=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    value_counts = series.value_counts()
    if top_n:
        value_counts = value_counts.head(top_n)
    
    value_counts.plot(kind='bar', ax=ax, edgecolor='black', alpha=0.7, color='#AEC6CF')
    ax.set_xlabel(series.name)
    ax.set_ylabel('Conteo')
    ax.set_title(f'Conteos: {series.name}')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig

def plot_pieplot(series, top_n=None):
    fig, ax = plt.subplots(figsize=(10, 7))
    
    value_counts = series.value_counts()
    if top_n and len(value_counts) > top_n:
        top_values = value_counts.head(top_n)
        others_count = value_counts.iloc[top_n:].sum()
        others_series = pd.Series([others_count], index=['Otros'])
        value_counts = pd.concat([top_values, others_series])
        
    colors = sns.color_palette("pastel", len(value_counts))
    wedges, texts, autotexts = ax.pie(value_counts, labels=value_counts.index, 
                                        autopct='%1.1f%%', colors=colors, 
                                        startangle=90)
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)
    
    ax.set_title(f'Distribución: {series.name}')
    plt.tight_layout()
    return fig

def plot_frequency_table(series, top_n=None):
    freq_table = series.value_counts()
    freq_pct = series.value_counts(normalize=True) * 100
    
    if top_n:
        freq_table = freq_table.head(top_n)
        freq_pct = freq_pct.head(top_n)
        
    result_df = pd.DataFrame({
        'Categoría': freq_table.index,
        'Conteo': freq_table.values,
        'Porcentaje': freq_pct.values
    }).reset_index(drop=True)
    
    return result_df

def generate_markdown_report(df, var_types, metadata_df):
    md = []
    md.append("# 📊 Reporte Completo de Análisis Exploratorio de Datos (EDA)\n")
    md.append("---\n\n")
    
    md.append("## ⚙️ Configuración del Dataset\n\n")
    md.append(f"- **Variables Cuantitativas Seleccionadas:** {len(var_types.get('Cuantitativa', []))}\n")
    md.append(f"- **Variables Cualitativas Seleccionadas:** {len(var_types.get('Cualitativa', []))}\n")
    md.append(f"- **Otras Variables / Texto:** {len(var_types.get('Mixta/Texto', []))}\n\n")
    
    md.append("### 📋 Tabla de Configuración\n\n")
    md.append("| " + " | ".join(metadata_df.columns) + " |\n")
    md.append("|" + "|".join(["---"] * len(metadata_df.columns)) + "|\n")
    for _, row in metadata_df.iterrows():
        md.append("| " + " | ".join(str(x) for x in row.values) + " |\n")
    md.append("\n")
    
    md.append("### 📝 Descripciones de Variables\n\n")
    descripciones_presentes = False
    for _, row in metadata_df.iterrows():
        if pd.notna(row['Descripción']) and str(row['Descripción']).strip() != '':
            md.append(f"**{row['Nuevo Nombre']}:** {row['Descripción']}\n\n")
            descripciones_presentes = True
    
    if not descripciones_presentes:
        md.append("*No hay descripciones disponibles. Complete la columna 'Descripción' en la pestaña Configuración para visualizar detalles de las variables.*\n\n")
    
    md.append("---\n\n")
    
    md.append("## 📋 Descripción General del Dataset\n\n")
    md.append(f"- **Filas:** {df.shape[0]}\n")
    md.append(f"- **Columnas:** {df.shape[1]}\n")
    md.append(f"- **Valores Faltantes Totales:** {df.isnull().sum().sum()}\n\n")
    
    md.append("💡 **Concepto Didáctico:** La descripción estadística consolidada proporciona un panorama distributivo inmediato, resumiendo la tendencia central, la dispersión y la forma subyacente de los datos.\n\n")
    
    md.append("### 📐 Nota Educativa: Resumen de los Cinco Números\n\n")
    md.append("**Definición:** Conjunto de descriptores estandarizados que trazan la distribución empírica en el espacio probabilístico.\n\n")
    md.append("$$S = \\{\\min(X), Q_1, \\tilde{x}, Q_3, \\max(X)\\}$$\n\n")
    
    md.append("### 💻 Nota Educativa: Implementación en Python\n\n")
    md.append("```python\nimport pandas as pd\n\nresumen = df.describe()\n```\n\n")
    
    md.append("✅ **Recomendación:** Utilice esta tabla generada como un escáner inicial veloz para detectar anomalías obvias, tales como valores mínimos negativos en variables que deberían ser estrictamente positivas.\n\n")
    
    md.append("---\n\n")
    
    quant_vars = var_types.get('Cuantitativa', [])
    qual_vars = var_types.get('Cualitativa', [])
    
    if quant_vars:
        md.append("### 🔢 Resumen Estadístico Global (Variables Cuantitativas)\n\n")
        desc_df = df[quant_vars].describe().reset_index()
        md.append("| Estadística | " + " | ".join(str(col) for col in desc_df.columns[1:]) + " |\n")
        md.append("|" + "|".join(["---"] * len(desc_df.columns)) + "|\n")
        for _, row in desc_df.iterrows():
            md.append("| " + " | ".join(str(round(x, 6)) if isinstance(x, (int, float)) else str(x) for x in row.values) + " |\n")
        md.append("\n\n")
    
    md.append("---\n\n")
    
    if quant_vars:
        md.append("## 🔢 Análisis Detallado de Variables Cuantitativas\n\n")
        for var in quant_vars:
            series = df[var].dropna()
            md.append(f"### Variable: {var}\n\n")
            
            md.append("#### 📊 Visualizaciones y Distribución Analítica\n\n")
            md.append("💡 **Concepto Didáctico:** Las descomposiciones visuales facilitan la apreciación inmediata de proporciones, asimetrías y comportamientos probabilísticos del espacio vectorial.\n\n")
            
            md.append("#### 📈 Información del Histograma y Tabla de Frecuencias\n\n")
            md.append("##### 📐 Nota Educativa: Frecuencia Empírica\n\n")
            md.append("**Definición:** Cálculo iterativo de la distribución agrupada en intervalos continuos.\n\n")
            md.append("$$\\text{Frecuencia}(x) = \\sum_{i=1}^{n} I(x_i \\in \\text{bin})$$\n\n")
            
            md.append("### 💻 Nota Educativa: Implementación en Python\n\n")
            md.append("```python\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\ndf['columna'].plot.hist(bins=30, edgecolor='black')\n```\n\n")
            
            md.append("✅ **Recomendación:** Ajustar la magnitud de los intervalos es clave para no ocultar fluctuaciones reales ni generar densidades falsas.\n\n")
            
            counts, bin_edges = np.histogram(series, bins=30)
            md.append("**Tabla de Frecuencias por Intervalos**\n\n")
            md.append("| Rango Inicio | Rango Fin | Conteo |\n")
            md.append("|---|---|---|\n")
            for i in range(len(counts)):
                md.append(f"| {bin_edges[i]:.6f} | {bin_edges[i+1]:.6f} | {counts[i]} |\n")
            md.append("\n")
            
            md.append("#### 📈 Información del Histograma + Densidad (KDE)\n\n")
            md.append("##### 📐 Nota Educativa: Estimación Kernel (KDE)\n\n")
            md.append("**Definición:** Suavizado matemático probabilístico del histograma subyacente de la variable.\n\n")
            md.append("$$\\hat{f}(x; h) = \\frac{1}{nh} \\sum_{i=1}^{n} K\\left(\\frac{x - x_i}{h}\\right)$$\n\n")
            
            md.append("##### 💻 Nota Educativa: Implementación en Python\n\n")
            md.append("```python\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\ndf['columna'].plot.hist(density=True, bins=20)\ndf['columna'].plot.density()\n```\n\n")
            
            md.append("✅ **Conclusión:** La superposición probabilística faculta comparar directamente contra curvas normales estandarizadas.\n\n")
            
            md.append("#### 📦 Información del Boxplot y Tabla de Datos Estructural\n\n")
            md.append("##### 📐 Nota Educativa: Rango Intercuartílico e Interpretación\n\n")
            md.append("**Definición:** Evaluación posicional exenta de sesgo perimetral midiendo el 50% intermedio.\n\n")
            md.append("$$\\text{RIC} = Q_3 - Q_1 \\quad \\text{Bigotes} = [Q_1 - 1.5 \\times \\text{RIC}, Q_3 + 1.5 \\times \\text{RIC}]$$\n\n")
            
            md.append("### 💻 Nota Educativa: Implementación en Python\n\n")
            md.append("```python\nimport plotly.express as px\n\npx.box(df, y='columna')\n```\n\n")
            
            md.append("✅ **Recomendación:** Un boxplot comprimido con extensa fragmentación más allá de los bigotes alerta de anomalías profundas en la ingesta.\n\n")
            
            desc_stats = series.describe()
            md.append("**Estadísticas de Distribución**\n\n")
            md.append("| Estadística | Valor |\n")
            md.append("|---|---|\n")
            for idx, val in desc_stats.items():
                md.append(f"| {idx} | {val:.6f} |\n")
            md.append("\n")
            
            loc_stats = compute_location_stats(series)
            md.append("#### 📈 Estadísticas de Localización\n\n")
            md.append("💡 **Concepto Didáctico:** Las medidas de tendencia central resumen el conjunto de datos en un valor representativo. La media convencional es altamente sensible a valores atípicos, mientras que la mediana y la media recortada ofrecen alternativas robustas.\n\n")
            
            md.append("##### 📐 Nota Educativa: Fórmulas de Tendencia Central\n\n")
            md.append("**Definición:** Descomposición matemática integral de las métricas principales para dimensionar el punto de equilibrio escalar en una distribución asimétrica.\n\n")
            md.append("$$\\text{Media: } \\bar{x} = \\frac{1}{n}\\sum_{i=1}^{n}x_i \\quad\\quad \\text{Mediana: } \\tilde{x} = x_{\\frac{n+1}{2}} \\quad\\quad \\text{Media Recortada: } \\bar{x}_{\\text{trim}} = \\frac{1}{n-2k}\\sum_{i=k+1}^{n-k}x_{(i)}$$\n\n")
            
            md.append("##### 💻 Nota Educativa: Implementación en Python\n\n")
            md.append("```python\nimport pandas as pd\nfrom scipy.stats import trim_mean\n\nmedia = df['columna'].mean()\nmediana = df['columna'].median()\nmedia_recortada = trim_mean(df['columna'].dropna(), 0.1)\n```\n\n")
            
            md.append("##### 📐 Nota Educativa: Extremos de la Distribución\n\n")
            md.append("**Definición:** Identificación de los límites inferiores y superiores absolutos del espacio vectorial para evaluar el rango global.\n\n")
            md.append("$$\\text{Mínimo: } \\min(X) \\quad\\quad \\text{Máximo: } \\max(X)$$\n\n")
            
            md.append("##### 💻 Nota Educativa: Implementación en Python\n\n")
            md.append("```python\nimport pandas as pd\n\nminimo = df['columna'].min()\nmaximo = df['columna'].max()\n```\n\n")
            
            for k, v in loc_stats.items():
                md.append(f"- **{k}:** {v:,.6f}\n")
            md.append("\n")
            
            md.append("✅ **Conclusión:** La disonancia entre la media y la mediana permite inferir la asimetría de la distribución subyacente; brechas considerables justifican estadísticamente el uso de métricas recortadas.\n\n")
            
            var_stats = compute_variability_stats(series)
            md.append("#### 📐 Estadísticas de Variabilidad\n\n")
            md.append("💡 **Concepto Didáctico:** La variabilidad dimensiona la dispersión matemática. La Desviación Estándar penaliza severamente los atípicos elevándolos al cuadrado, mientras que la Desviación Absoluta Mediana (MAD) conserva un carácter inquebrantable frente al ruido.\n\n")
            
            md.append("##### 📐 Nota Educativa: Ecuaciones de Dispersión\n\n")
            md.append("**Definición:** Funciones matemáticas para dimensionar la lejanía estandarizada de los puntos respecto a un pivote central.\n\n")
            md.append("$$s^2 = \\frac{\\sum (x_i - \\bar{x})^2}{n-1} \\quad s = \\sqrt{s^2} \\quad \\text{MAD} = \\text{Mediana}(\\vert{}x_i - \\tilde{x}\\vert{})$$\n\n")
            
            md.append("##### 💻 Nota Educativa: Implementación en Python\n\n")
            md.append("```python\nimport pandas as pd\n\ndesv = df['columna'].std()\nvar = df['columna'].var()\nmad = (df['columna'] - df['columna'].median()).abs().median()\n```\n\n")
            
            md.append("##### 📐 Nota Educativa: Métricas Complementarias de Dispersión\n\n")
            md.append("**Definición:** Ecuaciones para el rango absoluto, el rango intercuartílico (RIC) y la normalización de la dispersión frente a la media (CV).\n\n")
            md.append("$$\\text{Rango} = \\max - \\min \\quad \\text{RIC} = Q_3 - Q_1 \\quad \\text{CV} = \\frac{s}{\\bar{x}} \\times 100$$\n\n")
            
            for k, v in var_stats.items():
                md.append(f"- **{k}:** {v:,.6f}\n")
            md.append("\n")
            
            md.append("✅ **Recomendación:** Coteje la desviación estándar contra el indicador MAD; si excede sustancialmente el valor del MAD, valide obligatoriamente la integridad y existencia de extremos severos.\n\n")
            
            dist_stats = compute_distribution_stats(series)
            md.append("#### 📍 Estadísticas de Distribución\n\n")
            md.append("💡 **Concepto Didáctico:** Los percentiles seccionan la distribución en tramos relativos de densidad probabilística. La asimetría (Skewness) revela la deformación lateral, y la curtosis dimensiona la pesadez estocástica de las colas.\n\n")
            
            md.append("##### 📐 Nota Educativa: Cuantiles\n\n")
            md.append("**Definición:** Medida de posición no central que divide el conjunto de datos ordenados en partes porcentuales estandarizadas.\n\n")
            md.append("$$P_k = \\text{Valor } x \\text{ tal que } P(X \\le x) = \\frac{k}{100}$$\n\n")
            
            md.append("##### 💻 Nota Educativa: Implementación en Python\n\n")
            md.append("```python\nimport pandas as pd\n\npercentiles = df['columna'].quantile([0.05, 0.25, 0.5, 0.75, 0.95])\n```\n\n")
            
            md.append("##### 📐 Nota Educativa: Momentos de la Distribución\n\n")
            md.append("**Definición:** Cálculos estandarizados de tercer y cuarto orden para dimensionar direccionalidad de sesgo lateral y exceso de curtosis.\n\n")
            md.append("$$\\text{Asimetría} = \\frac{\\mu_3}{\\sigma^3} \\quad \\text{Curtosis} = \\frac{\\mu_4}{\\sigma^4} - 3$$\n\n")
            
            md.append("##### 💻 Nota Educativa: Implementación en Python\n\n")
            md.append("```python\nimport pandas as pd\n\nasimetria = df['columna'].skew()\ncurtosis = df['columna'].kurtosis()\n```\n\n")
            
            for k, v in dist_stats.items():
                md.append(f"- **{k}:** {v:,.6f}\n")
            md.append("\n")
            
            md.append("✅ **Conclusión:** Valores de asimetría fuera del umbral [-1, 1] evidencian distribuciones radicalmente sesgadas que pueden comprometer modelos de machine learning. Una curtosis superior a cero (leptocúrtica) es un marcador claro de alta incidencia de colas extremas.\n\n")
            
            md.append("#### 🔍 Análisis Detallado (Calidad de Datos)\n\n")
            md.append("💡 **Concepto Didáctico:** El perfilamiento integral de calidad de datos permite auditar la integridad muestral, identificando colisiones (duplicados) y vacíos de información (faltantes) que pueden introducir sesgos algorítmicos.\n\n")
            
            md.append("##### 📐 Nota Educativa: Calidad de Datos\n\n")
            md.append("**Definición:** Tasa de completitud y nivel cardinal de un vector de datos.\n\n")
            md.append("$$\\text{Completitud} = 1 - \\frac{\\text{Nulos}}{N} \\quad\\quad \\text{Cardinalidad} = \\vert \\{x_1, x_2, ..., x_k\\} \\vert$$\n\n")
            
            md.append("##### 💻 Nota Educativa: Implementación en Python\n\n")
            md.append("```python\nimport pandas as pd\n\nunicos = df['columna'].nunique()\nfaltantes = df['columna'].isnull().sum()\nduplicados = df['columna'].duplicated().sum()\n```\n\n")
            
            md.append(f"- **Total de Observaciones:** {len(series)}\n")
            md.append(f"- **Valores Únicos:** {series.nunique()}\n")
            md.append(f"- **Faltantes (del original):** {df[var].isnull().sum()}\n")
            md.append(f"- **Duplicados:** {df[var].duplicated().sum()}\n")
            md.append("\n")
            
            md.append("✅ **Recomendación:** Monitoree estrictamente los valores faltantes reportados en esta pestaña. Un nivel superior al 5-10% requiere técnicas de imputación avanzadas en lugar de la simple eliminación estadística.\n\n")
            
            md.append("#### ⚖️ Estadísticas Ponderadas (Teoría)\n\n")
            md.append("💡 **Concepto Didáctico:** Las estadísticas ponderadas corrigen desbalances de muestreo o agrupan subpoblaciones calibrando el peso algorítmico individual de los registros.\n\n")
            
            md.append("##### 📐 Nota Educativa: Promedios Ponderados\n\n")
            md.append("**Definición:** Ecuaciones robustas para determinar el equilibrio del centro considerando el nivel de impacto particular (peso).\n\n")
            md.append("$$\\bar{x}_w = \\frac{\\sum_{i=1}^{n} w_i x_i}{\\sum_{i=1}^{n} w_i} \\quad \\text{Mediana}_w = \\text{Valor donde } \\sum w_i \\geq \\frac{\\sum w}{2}$$\n\n")
            
            md.append("##### 💻 Nota Educativa: Implementación en Python\n\n")
            md.append("```python\nimport numpy as np\n\nnp.average(df['columna_valor'], weights=df['columna_peso'])\n```\n\n")
            
            md.append("✅ **Recomendación:** Evalúe la brecha absoluta entre métricas simples y ponderadas; discrepancias altas acentúan que la variable de impacto o factor de expansión altera significativamente la visión inicial del mercado.\n\n")
            
            md.append("---\n\n")
    
    if qual_vars:
        md.append("## 🏷️ Análisis Detallado de Variables Cualitativas\n\n")
        for var in qual_vars:
            series = df[var].dropna()
            md.append(f"### Variable: {var}\n\n")
            
            md.append("#### 📊 Visualizaciones Categóricas\n\n")
            md.append("💡 **Concepto Didáctico:** Las descomposiciones visuales categóricas facilitan la apreciación inmediata de proporciones volumétricas y comportamientos modales primarios.\n\n")
            
            md.append("##### 📐 Nota Educativa: Frecuencia Absoluta\n\n")
            md.append("**Definición:** Sumatoria de ocurrencias exactas en base condicional para cada categoría discreta del espacio muestral.\n\n")
            md.append("$$\\text{Frecuencia Absoluta} = \\sum_{i=1}^{n} I(x_i = C_j)$$\n\n")
            
            md.append("##### 💻 Nota Educativa: Implementación en Python\n\n")
            md.append("```python\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\ndf['columna_categorica'].value_counts().plot(kind='bar')\n```\n\n")
            
            md.append("##### 📐 Nota Educativa: Proporción Circular\n\n")
            md.append("**Definición:** Asignación de fragmentos en grados geométricos en base al porcentaje de participación relativa contra la suma total.\n\n")
            md.append("$$\\text{Proporción Sector} = \\frac{\\text{Frecuencia de } C_j}{\\sum \\text{Frecuencias}} \\times 360^\\circ$$\n\n")
            
            md.append("##### 💻 Nota Educativa: Implementación en Python\n\n")
            md.append("```python\nimport matplotlib.pyplot as plt\n\nvalue_counts = df['columna_categorica'].value_counts()\nax.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%')\n```\n\n")
            
            md.append("✅ **Recomendación:** Privilegie gráficos de barras en lugar del formato pastel para garantizar legibilidad cuantitativa al diferenciar magnitudes nominales en orden de impacto.\n\n")
            
            freq_df = plot_frequency_table(series)
            md.append("#### 📋 Tabla de Frecuencias\n\n")
            md.append("💡 **Concepto Didáctico:** El reporte tabular de frecuencias articula numéricamente la descomposición total, proyectando incidencias relativas fundamentales para un modelado probabilístico inicial.\n\n")
            
            md.append("##### 📐 Nota Educativa: Frecuencia Relativa y Porcentual\n\n")
            md.append("**Definición:** Relación proporcional de la frecuencia absoluta frente al total de observaciones en la muestra categórica.\n\n")
            md.append("$$f_i = \\frac{n_i}{N} \\times 100$$\n\n")
            
            md.append("##### 💻 Nota Educativa: Implementación en Python\n\n")
            md.append("```python\nimport pandas as pd\n\nfreq_table = df['columna_categorica'].value_counts()\nfreq_pct = (100 * freq_table / len(df['columna_categorica'].dropna()))\n```\n\n")
            
            md.append("| Categoría | Conteo | Porcentaje |\n")
            md.append("|---|---|---|\n")
            for _, row in freq_df.iterrows():
                md.append(f"| {row['Categoría']} | {row['Conteo']} | {row['Porcentaje']:.6f}% |\n")
            md.append("\n")
            
            md.append("✅ **Recomendación:** Analice los porcentajes acumulados para identificar rápidamente qué pocas categorías concentran la mayor parte de las observaciones totales.\n\n")
            
            md.append("#### 📊 Análisis Detallado\n\n")
            md.append("💡 **Concepto Didáctico:** La **Moda** representa la categoría con mayor ocurrencia. Las **Probabilidades** empíricas reflejan la proporción estandarizada de cada categoría. El **Valor Esperado** (distribución uniforme) traza la frecuencia teórica si todas las categorías lograran ser igualmente probables.\n\n")
            
            md.append("##### 📐 Nota Educativa: Probabilidad Empírica y Valor Esperado\n\n")
            md.append("**Definición:** Cuantificación de la probabilidad basada en la frecuencia relativa observada y estimación teórica bajo distribución uniforme.\n\n")
            md.append("$$P(x_i) = \\frac{f_i}{N} \\quad\\quad E[X]_{\\text{uniforme}} = \\frac{N}{k}$$\n\n")
            
            md.append("##### 💻 Nota Educativa: Implementación en Python\n\n")
            md.append("```python\nimport pandas as pd\n\nprobabilidades = df['columna_categorica'].value_counts(normalize=True)\nmoda = df['columna_categorica'].mode()[0]\n```\n\n")
            
            total_obs = len(series)
            cat_unicas = series.nunique()
            modas_calc = series.mode()
            moda_val = modas_calc[0] if len(modas_calc) > 0 else "N/A"
            valor_esperado = total_obs / cat_unicas if cat_unicas > 0 else 0
            
            md.append(f"- **Total Observaciones (N):** {total_obs}\n")
            md.append(f"- **Categorías Únicas (k):** {cat_unicas}\n")
            md.append(f"- **Moda:** {moda_val}\n")
            md.append(f"- **Valor Esperado:** {valor_esperado:,.2f}\n")
            md.append(f"- **Faltantes (del original):** {df[var].isnull().sum()}\n")
            md.append(f"- **Duplicados:** {df[var].duplicated().sum()}\n")
            md.append("\n")
            
            md.append("✅ **Conclusión:** Las divergencias estadísticamente significativas entre la frecuencia observada en la Moda y el Valor Esperado sugieren una distribución asimétrica que se aleja de la equiprobabilidad estricta, marcando un sesgo latente en el entorno muestral original.\n\n")
            
            md.append("---\n\n")
    
    md.append("## 📊 Análisis Multivariado\n\n")
    md.append("💡 **Concepto Didáctico:** El análisis multivariado explora las relaciones simultáneas entre múltiples variables, evaluando la interdependencia lineal mediante matrices de correlación y visualizando patrones bivariados.\n\n")
    
    quant_vars = var_types.get('Cuantitativa', [])
    qual_vars = var_types.get('Cualitativa', [])
    
    if len(quant_vars) >= 2:
        md.append("### 📈 Correlación entre Variables Cuantitativas\n\n")
        md.append("#### 📐 Nota Educativa: Coeficiente de Correlación de Pearson\n\n")
        md.append("**Definición:** Métrica algorítmica que cuantifica la dependencia lineal estructural entre dos vectores cuantitativos independientes.\n\n")
        md.append("$$r_{xy} = \\frac{\\sum (x_i - \\bar{x})(y_i - \\bar{y})}{\\sqrt{\\sum (x_i - \\bar{x})^2 \\sum (y_i - \\bar{y})^2}}$$\n\n")
        
        md.append("#### 💻 Nota Educativa: Implementación en Python\n\n")
        md.append("```python\nimport pandas as pd\nimport seaborn as sns\n\ncorr_matrix = df[['columna_1', 'columna_2']].corr()\nsns.heatmap(corr_matrix, annot=True, cmap='coolwarm')\n```\n\n")
        
        corr_matrix = df[quant_vars].corr()
        md.append("#### 📊 Matriz de Correlación entre Variables Cuantitativas\n\n")
        md.append("| Variable | " + " | ".join(str(col) for col in corr_matrix.columns) + " |\n")
        md.append("|---|" + "|".join(["---"] * len(corr_matrix.columns)) + "|\n")
        for idx, row in corr_matrix.iterrows():
            md.append(f"| {idx} | " + " | ".join(f"{x:.6f}" for x in row.values) + " |\n")
        md.append("\n")
        
        md.append("✅ **Conclusión:** Los bloques perimetrales marcados con extremada intensidad de color (cercanos a 1 o -1) advierten sobre escenarios de multicolinealidad estructural; depure las variables redundantes si procede a entrenar modelos predictivos estocásticos.\n\n")
    
    if len(quant_vars) >= 2:
        md.append("### 📍 Diagrama de Dispersión (Bivariado)\n\n")
        md.append("💡 **Concepto Didáctico:** El diagrama de dispersión proyecta pares iterados cartesianos de valores bi-dimensionales; resulta invaluable facilitando la identificación instintiva de clusters, tendencias paramétricas subyacentes y valores atípicos bivariados atípicos.\n\n")
        md.append("✅ **Conclusión:** La consolidación central de los puntos intercepta el análisis correlacional previo; las trayectorias compactas sin dispersión residual avalan robustamente la dependencia lineal bivariada deducida.\n\n")
    
    if len(quant_vars) > 0 and len(qual_vars) > 0:
        md.append("### 🔀 Comparación: Variable Cuantitativa vs Cualitativa\n\n")
        md.append("💡 **Concepto Didáctico:** Descomponer una variable cuantitativa continua seccionándola a través de perfiles categóricos expone divergencias sustanciales entre distribuciones, medias poblacionales y niveles de varianza intragrupal.\n\n")
        
        md.append("#### 📐 Nota Educativa: Rango Intercuartílico Condicional\n\n")
        md.append("**Definición:** Medida de dispersión estadística segregada por particiones discretas para evaluar la variabilidad intragrupal.\n\n")
        md.append("$$\\text{RIC}_g = Q_{3,g} - Q_{1,g} \\quad \\text{donde } g \\text{ representa cada categoría}$$\n\n")
        
        md.append("#### 💻 Nota Educativa: Implementación en Python\n\n")
        md.append("```python\nimport plotly.express as px\n\npx.box(df, x='columna_categorica', y='columna_cuantitativa')\n```\n\n")
        
        md.append("✅ **Recomendación:** Verifique los intervalos de interposición visual en los perfiles boxplot; una total disociación asimétrica de la estructura de cuartiles es un hallazgo concluyente de segmentación natural.\n\n")
    
    md.append("---\n\n")
    md.append("## 📄 Fin del Reporte\n\n")
    md.append("*Reporte generado automáticamente por Dashboard EDA*\n")
    md.append("*Basado en: Practical Statistics for Data Scientists - Capítulo 1*\n\n")
    
    return "\n".join(md)


# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================


st.title("📊 Dashboard de Análisis Exploratorio de Datos (EDA)")
st.markdown("""
Aplicación interactiva y genérica para explorar archivos CSV.
Basada en *Practical Statistics for Data Scientists* - Capítulo 1: Exploratory Data Analysis
""")

st.sidebar.header("⚙️ Configuración")

@st.cache_data
def load_dataframe(uploaded_file):
    uploaded_file.seek(0)
    compression_type = 'gzip' if uploaded_file.name.endswith('.gz') else None
    return pd.read_csv(uploaded_file, compression=compression_type, sep=None, engine='python')

@st.cache_data
def load_uciml_dataset(dataset_id):
    dataset = fetch_ucirepo(id=dataset_id)
    X = dataset.data.features
    y = dataset.data.targets
    if y is not None and not y.empty:
        return pd.concat([X, y], axis=1)
    return X

data_source = st.sidebar.selectbox(
    "Seleccione la fuente de datos",
    ["Cargar archivo CSV", "Bank Marketing", "Default of Credit Card Clients", "Online Retail"]
)

df = None

if data_source == "Cargar archivo CSV":
    uploaded_file = st.sidebar.file_uploader("Selecciona un archivo CSV", type=['csv', 'gz'])
    if uploaded_file is not None:
        df = load_dataframe(uploaded_file)
elif data_source == "Bank Marketing":
    df = load_uciml_dataset(222)
elif data_source == "Default of Credit Card Clients":
    df = load_uciml_dataset(350)
elif data_source == "Online Retail":
    df = load_uciml_dataset(352)

if df is not None:
    st.sidebar.success("✅ Datos cargados exitosamente")
    
    current_columns = list(df.columns)
    if 'prev_columns' not in st.session_state or st.session_state.prev_columns != current_columns:
        st.session_state.prev_columns = current_columns
        initial_types = classify_variables(df)
        metadata = []
        for col in current_columns:
            col_type = "Otro"
            for t, cols in initial_types.items():
                if col in cols:
                    col_type = t
                    break
            if col_type == 'Cuantitativa':
                default_type = "Cuantitativa (Continua)"
            else:
                default_type = "Cualitativa (Nominal)"
            metadata.append({
                "Columna Original": col,
                "Nuevo Nombre": col,
                "Descripción": "",
                "Tipo de Variable": default_type,
                "Incluir": True
            })
        st.session_state.metadata_df = pd.DataFrame(metadata)

    tab_config, tab1, tab2, tab3, tab4 = st.tabs([
        "⚙️ Configuración",
        "📋 Descripción General",
        "🔢 Variables Cuantitativas",
        "🏷️ Variables Cualitativas",
        "📊 Análisis Multivariado"
    ])
    
    with tab_config:
        st.header("Configuración de Dataset")
        with st.form("config_dataset_form"):
            edited_metadata = st.data_editor(
                st.session_state.metadata_df,
                column_config={
                    "Columna Original": st.column_config.Column(disabled=True),
                    "Tipo de Variable": st.column_config.SelectboxColumn(
                        options=[
                            "Cualitativa (Nominal)", 
                            "Cualitativa (Ordinal)", 
                            "Cuantitativa (Discreta)", 
                            "Cuantitativa (Continua)"
                        ],
                        required=True
                    )
                },
                width='stretch',
                hide_index=True,
                key="meta_editor"
            )
            btn_actualizar = st.form_submit_button("Actualizar Análisis EDA", type="primary")
            
        if btn_actualizar:
            st.session_state.metadata_df = edited_metadata
        else:
            edited_metadata = st.session_state.metadata_df
        
        included_cols = edited_metadata[edited_metadata["Incluir"] == True]
    df_modified = df[included_cols["Columna Original"].tolist()].copy()
    rename_dict = dict(zip(included_cols["Columna Original"], included_cols["Nuevo Nombre"]))
    df_modified.rename(columns=rename_dict, inplace=True)
    
    var_types = {'Cuantitativa': [], 'Cualitativa': [], 'Mixta/Texto': []}
    
    quant_mask = included_cols["Tipo de Variable"].str.contains("Cuantitativa")
    quant_cols = included_cols.loc[quant_mask, "Nuevo Nombre"].tolist()
    qual_cols = included_cols.loc[~quant_mask, "Nuevo Nombre"].tolist()
    
    var_types['Cuantitativa'] = quant_cols
    var_types['Cualitativa'] = qual_cols
    
    if quant_cols:
        df_modified[quant_cols] = df_modified[quant_cols].apply(pd.to_numeric, errors='coerce')
    if qual_cols:
        df_modified[qual_cols] = df_modified[qual_cols].astype(str)
    
    df = df_modified
    
    with tab_config:
        pass
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Información del Dataset")
    st.sidebar.metric("Filas", df.shape[0])
    st.sidebar.metric("Columnas", df.shape[1])
    st.sidebar.metric("Valores Faltantes", df.isnull().sum().sum())
    
    with tab1:
            st.header("Descripción General del Dataset")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Primeras Filas")
                df_head = df.head(10).copy()
                df_head.index.name = None
                numeric_cols = df_head.select_dtypes(include=np.number).columns
                st.dataframe(df_head.style.format(formatter={col: "{:,.6f}" for col in numeric_cols}).bar(subset=numeric_cols, color='#CDE0F5').set_properties(**{'background-color': '#F8FBFF', 'border': '1px solid #E2E8F0', 'color': '#1C2833'}), width='stretch')
            
            with col2:
                st.subheader("Información de Tipos de Datos")
                faltantes = df.isnull().sum()
                no_nulos = df.count()
                desc_dict = dict(zip(included_cols["Nuevo Nombre"], included_cols["Descripción"]))
                desc_values = [desc_dict.get(col, "") for col in df.columns]
                info_df = pd.DataFrame({
                    'Columna': df.columns,
                    'Descripción': desc_values,
                    'Tipo': df.dtypes.astype(str).values,
                    'No Nulos': no_nulos.values,
                    'Faltantes': faltantes.values
                })
                info_df.index.name = None
                st.dataframe(info_df.style.bar(subset=info_df.select_dtypes(include=np.number).columns, color='#FDE8C4').set_properties(**{'background-color': '#FFFCF7', 'border': '1px solid #FAD7A1', 'color': '#3E2723'}), width='stretch')
            
            st.subheader("Estadísticas Descriptivas Rápidas")
            
            st.info("💡 **Concepto Didáctico:** La descripción estadística consolidada proporciona un panorama distributivo inmediato, resumiendo la tendencia central, la dispersión y la forma subyacente de los datos.")
            
            st.info("### 📐 Nota Educativa: Resumen de los Cinco Números\n**Definición:** Conjunto de descriptores estandarizados que trazan la distribución empírica en el espacio probabilístico.\n\n$$S = \\{\\min(X), Q_1, \\tilde{x}, Q_3, \\max(X)\\}$$")
            
            st.warning("### 💻 Nota Educativa: Implementación en Python\n**Definición:** Generación de un marco de datos de resumen estadístico global utilizando el método de evaluación intrínseco de Pandas.\n\n```python\nimport pandas as pd\n\nresumen = df.describe()\n```")
            
            st.success("✅ **Recomendación:** Utilice esta tabla generada como un escáner inicial veloz para detectar anomalías obvias, tales como valores mínimos negativos en variables que deberían ser estrictamente positivas (por ejemplo, edad o precios).")

            quant_vars = var_types.get('Cuantitativa', [])
            if quant_vars:
                desc_df = df[quant_vars].describe()
                desc_df.index.name = None
                st.dataframe(desc_df.style.format("{:,.6f}").bar(subset=desc_df.columns, color='#E2F0CB').set_properties(**{'background-color': '#FAFDFA', 'border': '1px solid #D5E8D4', 'color': '#194D33'}), width='stretch')
            else:
                st.info("No hay variables cuantitativas seleccionadas para mostrar estadísticas.")
                st.subheader("Clasificación de Variables")
        
            col1, col2, col3 = st.columns(3)
            with col1:
                if 'Cuantitativa' in var_types:
                    st.metric("Variables Cuantitativas", len(var_types['Cuantitativa']))
            with col2:
                if 'Cualitativa' in var_types:
                    st.metric("Variables Cualitativas", len(var_types['Cualitativa']))
            with col3:
                if 'Mixta/Texto' in var_types:
                    st.metric("Otras Variables", len(var_types.get('Mixta/Texto', [])))
            
            # Mostrar variables por tipo
            st.subheader("Listado de Variables")
            for var_type, variables in var_types.items():
                st.write(f"**{var_type}:** {', '.join(variables)}")
    
# ========================================================================
    # TAB 2: ANÁLISIS DE VARIABLES CUANTITATIVAS
    # ========================================================================
    with tab2:
        st.header("Análisis de Variables Cuantitativas")
        
        quantitative_vars = var_types.get('Cuantitativa', [])
        
        if quantitative_vars:
            selected_var = st.selectbox(
                "Selecciona una variable cuantitativa",
                quantitative_vars,
                key="quant_var"
            )
            
            series = df[selected_var].dropna()
            
            sub_tab1, sub_tab2, sub_tab3, sub_tab4, sub_tab5, sub_tab6 = st.tabs([
                "📊 Gráficos",
                "📈 Localización",
                "📐 Variabilidad",
                "📍 Distribución",
                "🔍 Análisis Detallado",
                "⚖️ Estadísticas Ponderadas"
            ])
            
            with sub_tab1:
                st.subheader(f"Visualizaciones: {selected_var}")
                st.info("💡 **Concepto Didáctico:** La visualización de la distribución permite evaluar rápidamente la normalidad, identificar asimetrías y detectar valores atípicos.")
                
                chart_type = st.radio(
                    "Tipo de gráfico",
                    ["Histograma", "Histograma + Densidad", "Boxplot"],
                    horizontal=True
                )
                
                if chart_type == "Histograma":
                    st.info("""### 📐 Nota Educativa: Ecuación Matemática
**Definición:** Cálculo formal de la frecuencia de distribución empírica para datos continuos agrupados en intervalos fijos espaciales.

$$\\text{Frecuencia}(x) = \\sum_{i=1}^{n} I(x_i \\in \\text{bin})$$
""")
                    st.warning("""### 💻 Nota Educativa: Implementación en Python
**Definición:** Fragmento de código modular estándar utilizando la librería Pandas respaldada por el motor gráfico subyacente de Matplotlib.

```python
import pandas as pd
import matplotlib.pyplot as plt

df['columna'].plot.hist(bins=bins, edgecolor='black')
```""")
                    st.success("✅ **Recomendación:** Ajuste el número de bins. Muy pocos ocultan detalles; demasiados generan ruido visual que dificulta interpretar la tendencia subyacente.")


                    bins = st.slider("Número de bins", 5, 50, 30)
                    st.pyplot(plot_histogram(series, bins=bins))
                    counts, bin_edges = np.histogram(series, bins=bins)
                    df_hist = pd.DataFrame({"Rango_Inicio": bin_edges[:-1], "Rango_Fin": bin_edges[1:], "Conteo": counts})
                    df_hist.index.name = None
                    st.dataframe(df_hist.style.format({"Rango_Inicio": "{:,.6f}", "Rango_Fin": "{:,.6f}", "Conteo": "{:,.0f}"}).bar(subset=['Conteo'], color='#F4C2C2').set_properties(**{'background-color': '#FFF5F5', 'border': '1px solid #FADCDC', 'color': '#641E16'}), width='stretch')
                
                elif chart_type == "Histograma + Densidad":
                    st.info(r"""### 📐 Nota Educativa: Estimación de Densidad Kernel (KDE)
**Definición:** Suavizado continuo del histograma clásico para estimar la función de densidad de probabilidad de una variable aleatoria.

$$ \hat{f}(x; h) = \frac{1}{nh} \sum_{i=1}^{n} K\left(\frac{x - x_i}{h}\right) $$
""")
                    st.warning("""### 💻 Nota Educativa: Implementación en Python
**Definición:** Superposición de densidades probabilísticas utilizando el método interno de renderizado en Pandas.

```python
import pandas as pd
import matplotlib.pyplot as plt

df['columna'].plot.hist(density=True)
df['columna'].plot.density()
```""")
                    st.success("✅ **Recomendación:** Utilice la curva de densidad acoplada para comparar la forma de la distribución contra modelos teóricos estandarizados independientemente de la magnitud de la muestra.")
                    


                    bins = st.slider("Número de bins", 5, 50, 20)
                    st.pyplot(plot_histogram_density(series, bins=bins))
                    counts, bin_edges = np.histogram(series, bins=bins)
                    df_hist = pd.DataFrame({"Rango_Inicio": bin_edges[:-1], "Rango_Fin": bin_edges[1:], "Conteo": counts})
                    df_hist.index.name = None
                    st.dataframe(df_hist.style.format({"Rango_Inicio": "{:,.6f}", "Rango_Fin": "{:,.6f}", "Conteo": "{:,.6f}"}).bar(subset=['Conteo'], color='#F4C2C2').set_properties(**{'background-color': '#FFF5F5', 'border': '1px solid #FADCDC', 'color': '#641E16'}), width='stretch')
                
                else:
                    st.info(r"""### 📐 Nota Educativa: Rango Intercuartílico (RIC)
**Definición:** Medida de dispersión estadística que evalúa la amplitud del 50% central de los datos para la construcción de los límites del boxplot.

$$\text{RIC} = Q_3 - Q_1 \quad \text{Bigotes} = [Q_1 - 1.5 \times \text{RIC}, Q_3 + 1.5 \times \text{RIC}]$$
""")
                    st.warning("""### 💻 Nota Educativa: Implementación en Python
**Definición:** Generación de un diagrama de caja interactivo con Plotly Express.

```python
import plotly.express as px

px.box(df, y='columna')
```""")
                    st.success("✅ **Conclusión:** El boxplot permite identificar de inmediato valores atípicos severos (puntos aislados más allá de los bigotes) comprobando la compresión de la caja intercuartílica.")
                    st.plotly_chart(plot_boxplot(series), use_container_width=True)
                    dist_stats_df = series.describe().to_frame(name="Estadísticas de Distribución").T
                    dist_stats_df.index.name = None
                    st.dataframe(dist_stats_df.style.format("{:,.6f}").bar(color='#D4E6F1').set_properties(**{'background-color': '#F4F9F9', 'border': '1px solid #E5E8E8', 'color': '#1B4F72'}), width='stretch')
            
            with sub_tab2:
                st.subheader(f"Estadísticas de Localización: {selected_var}")
                location_stats = compute_location_stats(series)
                
                st.info("💡 **Concepto Didáctico:** Las medidas de tendencia central resumen el conjunto de datos en un valor representativo. La media convencional es altamente sensible a valores atípicos, mientras que la mediana y la media recortada ofrecen alternativas robustas.")
                
                st.info("""### 📐 Nota Educativa: Fórmulas de Tendencia Central
**Definición:** Descomposición matemática integral de las métricas principales para dimensionar el punto de equilibrio escalar en una distribución asimétrica.

$$\\text{Media: } \\bar{x} = \\frac{1}{n}\\sum_{i=1}^{n}x_i \\quad\\quad \\text{Mediana: } \\tilde{x} = x_{\\frac{n+1}{2}} \\quad\\quad \\text{Media Recortada: } \\bar{x}_{\\text{trim}} = \\frac{1}{n-2k}\\sum_{i=k+1}^{n-k}x_{(i)}$$
""")
                
                st.warning("""### 💻 Nota Educativa: Implementación en Python
**Definición:** Script declarativo para la extracción de medidas posicionales utilizando Pandas y el módulo estadístico avanzado de SciPy.

```python
import pandas as pd
from scipy.stats import trim_mean

media = df['columna'].mean()
mediana = df['columna'].median()
media_recortada = trim_mean(df['columna'].dropna(), 0.1)
```""")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Media", f"{location_stats['Media']:,.6f}")
                with col2:
                    st.metric("Mediana", f"{location_stats['Mediana']:,.6f}")
                with col3:
                    st.metric("Media Recortada (10%)", f"{location_stats['Media Recortada (10%)']:,.6f}")
                
                st.info(r"""### 📐 Nota Educativa: Extremos de la Distribución
**Definición:** Identificación de los límites inferiores y superiores absolutos del espacio vectorial para evaluar el rango global.

$$\text{Mínimo: } \min(X) \quad\quad \text{Máximo: } \max(X)$$
""")
                
                st.warning("""### 💻 Nota Educativa: Implementación en Python
**Definición:** Extracción de valores extremos mediante las funciones de agregación nativas de Pandas.

```python
import pandas as pd

minimo = df['columna'].min()
maximo = df['columna'].max()
```""")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mínimo", f"{location_stats['Min']:,.6f}")
                with col2:
                    st.metric("Máximo", f"{location_stats['Max']:,.6f}")
                
                st.success("✅ **Conclusión:** La disonancia entre la media y la mediana permite inferir la asimetría de la distribución subyacente; brechas considerables justifican estadísticamente el uso de métricas recortadas.")
                
                stats_df = pd.DataFrame(list(location_stats.items()), columns=['Estadística', 'Valor'])
                stats_df.index.name = None
                st.dataframe(stats_df.style.format({"Valor": "{:,.6f}"}).bar(subset=['Valor'], color='#E8DAEF').set_properties(**{'background-color': '#FDF9FF', 'border': '1px solid #EBDEF0', 'color': '#4A235A'}), width='stretch')
            
            with sub_tab3:
                st.subheader(f"Estadísticas de Variabilidad: {selected_var}")
                
                st.info("💡 **Concepto Didáctico:** La variabilidad dimensiona la dispersión matemática. La Desviación Estándar penaliza severamente los atípicos elevándolos al cuadrado, mientras que la Desviación Absoluta Mediana (MAD) conserva un carácter inquebrantable frente al ruido.")
                st.info(r"""### 📐 Nota Educativa: Ecuaciones de Dispersión
**Definición:** Funciones matemáticas para dimensionar la lejanía estandarizada de los puntos respecto a un pivote central.

$$s^2 = \frac{\sum (x_i - \bar{x})^2}{n-1} \quad s = \sqrt{s^2} \quad \text{MAD} = \text{Mediana}(\vert{}x_i - \tilde{x}\vert{})$$
""")
                st.warning("""### 💻 Nota Educativa: Implementación en Python
**Definición:** Cálculo vectorizado de varianza, desviación estándar y MAD.

```python
import pandas as pd

desv = df['columna'].std()
var = df['columna'].var()
mad = (df['columna'] - df['columna'].median()).abs().median()
```""")
                
                variability_stats = compute_variability_stats(series)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Desv. Estándar", f"{variability_stats['Desv. Est.']:,.6f}")
                with col2:
                    st.metric("Varianza", f"{variability_stats['Varianza']:,.6f}")
                with col3:
                    st.metric("MAD", f"{variability_stats['MAD']:,.6f}")
                
                st.info(r"""### 📐 Nota Educativa: Métricas Complementarias de Dispersión
**Definición:** Ecuaciones para el rango absoluto, el rango intercuartílico (RIC) y la normalización de la dispersión frente a la media (CV).

$$\text{Rango} = \max - \min \quad \text{RIC} = Q_3 - Q_1 \quad \text{CV} = \frac{s}{\bar{x}} \times 100$$
""")
                
                col_rango, col_ric, col_cv = st.columns(3)
                with col_rango:
                    st.metric("Rango", f"{variability_stats['Rango']:,.6f}")
                with col_ric:
                    st.metric("RIC (IQR)", f"{variability_stats['RIC (IQR)']:,.6f}")
                with col_cv:
                    st.metric("CV (%)", f"{variability_stats['CV (%)']:,.6f}")
                
                st.success("✅ **Recomendación:** Coteje la desviación estándar contra el indicador MAD; si excede sustancialmente el valor del MAD, valide obligatoriamente la integridad y existencia de extremos severos.")
            
            with sub_tab4:
                st.subheader(f"Estadísticas de Distribución: {selected_var}")
                dist_stats = compute_distribution_stats(series)
                
                st.info("💡 **Concepto Didáctico:** Los percentiles seccionan la distribución en tramos relativos de densidad probabilística. La asimetría (Skewness) revela la deformación lateral, y la curtosis dimensiona la pesadez estocástica de las colas.")
                st.info(r"""### 📐 Nota Educativa: Cuantiles
**Definición:** Medida de posición no central que divide el conjunto de datos ordenados en partes porcentuales estandarizadas.

$$P_k = \text{Valor } x \text{ tal que } P(X \le x) = \frac{k}{100}$$
""")
                st.warning("""### 💻 Nota Educativa: Implementación en Python
**Definición:** Extracción simultánea de múltiples percentiles clave para evaluar los márgenes de densidad poblacional.

```python
import pandas as pd

percentiles = df['columna'].quantile([0.05, 0.25, 0.5, 0.75, 0.95])
```""")
                
                col1, col2, col3 = st.columns(3)
                percentiles = {
                    'P5%': dist_stats['P5%'],
                    'P25%': dist_stats['P25%'],
                    'P50%': dist_stats['P50% (Mediana)']
                }
                
                for i, (label, value) in enumerate(percentiles.items()):
                    with st.columns(3)[i % 3]:
                        st.metric(label, f"{value:,.6f}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("P75%", f"{dist_stats['P75%']:,.6f}")
                with col2:
                    st.metric("P95%", f"{dist_stats['P95%']:,.6f}")
                
                st.info(r"""### 📐 Nota Educativa: Momentos de la Distribución
**Definición:** Cálculos estandarizados de tercer y cuarto orden para dimensionar direccionalidad de sesgo lateral y exceso de curtosis.

$$\text{Asimetría} = \frac{\mu_3}{\sigma^3} \quad \text{Curtosis} = \frac{\mu_4}{\sigma^4} - 3$$
""")
                st.warning("""### 💻 Nota Educativa: Implementación en Python
**Definición:** Obtención empírica de asimetría y curtosis (ajustada de Fisher) extraídas del marco univariado de datos.

```python
import pandas as pd

asimetria = df['columna'].skew()
curtosis = df['columna'].kurtosis()
```""")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Asimetría (Skewness)", f"{dist_stats['Asimetría']:,.6f}")
                with col2:
                    st.metric("Curtosis", f"{dist_stats['Curtosis']:,.6f}")
                
                st.success("✅ **Conclusión:** Valores de asimetría fuera del umbral [-1, 1] evidencian distribuciones radicalmente sesgadas que pueden comprometer modelos de machine learning. Una curtosis superior a cero (leptocúrtica) es un marcador claro de alta incidencia de colas extremas.")
                
                dist_df = pd.DataFrame(list(dist_stats.items()), columns=['Estadística', 'Valor'])
                dist_df.index.name = None
                st.dataframe(dist_df.style.format({"Valor": "{:,.6f}"}).bar(subset=['Valor'], color='#D1F2EB').set_properties(**{'background-color': '#F4FCF9', 'border': '1px solid #E8F8F5', 'color': '#0E6251'}), width='stretch')
            
            with sub_tab5:
                st.subheader(f"Análisis Detallado: {selected_var}")
                
                st.info("💡 **Concepto Didáctico:** El perfilamiento integral de calidad de datos permite auditar la integridad muestral, identificando colisiones (duplicados) y vacíos de información (faltantes) que pueden introducir sesgos algorítmicos.")
                
                st.info("### 📐 Nota Educativa: Calidad de Datos\n**Definición:** Tasa de completitud y nivel cardinal de un vector de datos.\n\n$$\\text{Completitud} = 1 - \\frac{\\text{Nulos}}{N} \\quad\\quad \\text{Cardinalidad} = \\vert \\{x_1, x_2, ..., x_k\\} \\vert$$")
                
                st.warning("### 💻 Nota Educativa: Implementación en Python\n**Definición:** Métodos integrados en la librería Pandas para auditoría estructural de calidad de datos.\n\n```python\nimport pandas as pd\n\nunicos = df['columna'].nunique()\nfaltantes = df['columna'].isnull().sum()\nduplicados = df['columna'].duplicated().sum()\n```")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total de Observaciones", len(series))
                    st.metric("Valores Únicos", series.nunique())
                
                with col2:
                    st.metric("Faltantes (del original)", df[selected_var].isnull().sum())
                    st.metric("Duplicados", df[selected_var].duplicated().sum())
                    
                st.success("✅ **Recomendación:** Monitoree estrictamente los valores faltantes reportados en esta pestaña. Un nivel superior al 5-10% requiere técnicas de imputación avanzadas en lugar de la simple eliminación estadística.")
            
            with sub_tab6:
                st.subheader("Análisis Ponderado")
                weight_vars = [v for v in quantitative_vars if v != selected_var]
                
                if weight_vars:
                    weight_col = st.selectbox("Variable de Peso (Weights)", weight_vars)
                    
                    st.info("💡 **Concepto Didáctico:** Las estadísticas ponderadas corrigen desbalances de muestreo o agrupan subpoblaciones calibrando el peso algorítmico individual de los registros.")
                    st.info(r"""### 📐 Nota Educativa: Promedios Ponderados
**Definición:** Ecuaciones robustas para determinar el equilibrio del centro considerando el nivel de impacto particular (peso).
- $w_i$: Peso de la observación
- $x_i$: Valor de la observación

$$ \bar{x}_w = \frac{\sum_{i=1}^{n} w_i x_i}{\sum_{i=1}^{n} w_i} \quad \text{Mediana}_w = \text{Valor donde } \sum w_i \geq \frac{\sum w}{2} $$
""")
                    st.warning("""### 💻 Nota Educativa: Implementación en Python
**Definición:** Integración del módulo de cálculo científico de Numpy para derivar la calibración paramétrica.

```python
import numpy as np

np.average(df['columna_valor'], weights=df['columna_peso'])
```""")
                    
                    w_mean, w_median = compute_weighted_stats(df, selected_var, weight_col)
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("Media Ponderada", f"{w_mean:,.6f}" if pd.notnull(w_mean) else "N/A")
                    with c2:
                        st.metric("Mediana Ponderada", f"{w_median:,.6f}" if pd.notnull(w_median) else "N/A")
                    st.success("✅ **Recomendación:** Evalué la brecha absoluta entre métricas simples y ponderadas; discrepancias altas acentúan que la variable de impacto o factor de expansión altera significativamente la visión inicial del mercado.")
                else:
                    st.info("No hay variables cuantitativas adicionales para ponderación.")
        else:
            st.warning("No hay variables cuantitativas en el dataset")
    
# ========================================================================
    # TAB 3: ANÁLISIS DE VARIABLES CUALITATIVAS
    # ========================================================================
    with tab3:
        st.header("Análisis de Variables Cualitativas")
        
        qualitative_vars = var_types.get('Cualitativa', [])
        
        if qualitative_vars:
            selected_var = st.selectbox(
                "Selecciona una variable cualitativa",
                qualitative_vars,
                key="qual_var"
            )
            
            series = df[selected_var].dropna()
            
            sub_tab1, sub_tab2, sub_tab3 = st.tabs([
                "📊 Gráficos",
                "📋 Tabla de Frecuencias",
                "📊 Análisis Detallado"
            ])
            
            with sub_tab1:
                st.subheader(f"Visualizaciones: {selected_var}")
                st.info("💡 **Concepto Didáctico:** Las descomposiciones visuales categóricas facilitan la apreciación inmediata de proporciones volumétricas y comportamientos modales primarios.")
                
                top_n = st.slider("Mostrar top N categorías", 1, min(20, len(series.unique())), 10, key="slider_vis_qual")
                
                chart_type = st.radio(
                    "Tipo de gráfico",
                    ["Gráfico de Barras (Countplot)", "Gráfico de Pastel (Pie)"],
                    horizontal=True
                )
                
                if chart_type == "Gráfico de Barras (Countplot)":
                    st.info(r"""### 📐 Nota Educativa: Frecuencia Absoluta
**Definición:** Sumatoria de ocurrencias exactas en base condicional para cada categoría discreta del espacio muestral.

$$\text{Frecuencia Absoluta} = \sum_{i=1}^{n} I(x_i = C_j)$$
""")
                    st.warning("""### 💻 Nota Educativa: Implementación en Python
**Definición:** Extracción iterativa de recuentos e inyección perimetral en el motor de gráficos de barras.

```python
import pandas as pd
import matplotlib.pyplot as plt

df['columna_categorica'].value_counts().plot(kind='bar')
```""")
                    st.success("✅ **Recomendación:** Privilegie gráficos de barras en lugar del formato pastel para garantizar legibilidad cuantitativa al diferenciar magnitudes nominales en orden de impacto.")
                    st.pyplot(plot_countplot(series, top_n=top_n))
                    # ... [tail omitido para brevedad]
                else:
                    st.info(r"""### 📐 Nota Educativa: Proporción Circular
**Definición:** Asignación de fragmentos en grados geométricos en base al porcentaje de participación relativa contra la suma total.

$$ \text{Proporción Sector} = \frac{\text{Frecuencia de } C_j}{\sum \text{Frecuencias}} \times 360^\circ $$
""")
                    st.warning("""### 💻 Nota Educativa: Implementación en Python
**Definición:** Diagramación del esquema perimetral circular nativo utilizando los sub-procesos de Matplotlib.

```python
import matplotlib.pyplot as plt

value_counts = df['columna_categorica'].value_counts()
ax.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%')
```""")
                    st.success("✅ **Recomendación:** Reduzca la utilización de diagramas de pastel a variables que consoliden exclusivamente entre dos y cuatro cardinalidades limitadas.")
                    st.pyplot(plot_pieplot(series, top_n=top_n))
                    df_counts = series.value_counts().head(top_n).to_frame(name="Conteo")
                    df_counts.index.name = None
                    st.dataframe(df_counts.style.format("{:,.0f}").bar(subset=['Conteo'], color='#AEC6CF'), width='stretch')
            
            with sub_tab2:
                st.subheader(f"Tabla de Frecuencias: {selected_var}")
                st.info("💡 **Concepto Didáctico:** El reporte tabular de frecuencias articula numéricamente la descomposición total, proyectando incidencias relativas fundamentales para un modelado probabilístico inicial.")
                st.info(r"""### 📐 Nota Educativa: Frecuencia Relativa y Porcentual
**Definición:** Relación proporcional de la frecuencia absoluta frente al total de observaciones en la muestra categórica.

$$f_i = \frac{n_i}{N} \times 100$$
""")
                st.warning("""### 💻 Nota Educativa: Implementación en Python
**Definición:** Cálculo vectorizado de recuentos absolutos y porcentajes normalizados mediante Pandas.

```python
import pandas as pd

freq_table = df['columna_categorica'].value_counts()
freq_pct = (100 * freq_table / len(df['columna_categorica'].dropna()))
```""")
                st.success("✅ **Recomendación:** Analice los porcentajes acumulados para identificar rápidamente qué pocas categorías concentran la mayor parte de las observaciones totales.")
                
                top_n = st.slider("Mostrar top N categorías", 1, min(30, len(series.unique())), 10, key="slider_freq_qual")
                
                freq_df = plot_frequency_table(series, top_n=top_n)
                freq_df.index.name = None
                st.dataframe(freq_df.style.format({"Conteo": "{:,.6f}", "Porcentaje": "{:,.6f}"}).bar(subset=['Conteo', 'Porcentaje'], color='#D4E6F1').set_properties(**{'background-color': '#F8F9FA', 'border': '1px solid #EAECEE', 'color': '#154360'}), width='stretch')
            
            with sub_tab3:
                st.subheader(f"Análisis Detallado: {selected_var}")
                
                st.info("💡 **Concepto Didáctico:** La **Moda** representa la categoría con mayor ocurrencia. Las **Probabilidades** empíricas reflejan la proporción estandarizada de cada categoría. El **Valor Esperado** (distribución uniforme) traza la frecuencia teórica si todas las categorías lograran ser igualmente probables.")
                st.info(r"""### 📐 Nota Educativa: Probabilidad Empírica y Valor Esperado
**Definición:** Cuantificación de la probabilidad basada en la frecuencia relativa observada y estimación teórica bajo distribución uniforme.

$$P(x_i) = \frac{f_i}{N} \quad\quad E[X]_{\text{uniforme}} = \frac{N}{k}$$
""")
                st.warning("""### 💻 Nota Educativa: Implementación en Python
**Definición:** Extracción de probabilidades normalizadas y estadísticas descriptivas categóricas con Pandas.

```python
import pandas as pd

probabilidades = df['columna_categorica'].value_counts(normalize=True)
moda = df['columna_categorica'].mode()[0]
```""")
                st.success("✅ **Conclusión:** Las divergencias estadísticamente significativas entre la frecuencia observada en la Moda y el Valor Esperado sugieren una distribución asimétrica que se aleja de la equiprobabilidad estricta.")
                
                total_obs = len(series)
                cat_unicas = series.nunique()
                modas_calc = series.mode()
                moda_val = modas_calc[0] if len(modas_calc) > 0 else "N/A"
                valor_esperado = total_obs / cat_unicas if cat_unicas > 0 else 0
                probabilidades = series.value_counts(normalize=True).to_frame(name="Probabilidad")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Observaciones (N)", total_obs)
                with col2:
                    st.metric("Categorías Únicas (k)", cat_unicas)
                with col3:
                    st.metric("Moda", moda_val)
                with col4:
                    st.metric("Valor Esperado", f"{valor_esperado:,.2f}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Faltantes (del original)", df[selected_var].isnull().sum())
                with col2:
                    st.metric("Duplicados", df[selected_var].duplicated().sum())
                
                st.markdown("**Probabilidades Empíricas por Categoría:**")
                probabilidades.index.name = None
                st.dataframe(probabilidades.style.format("{:,.6f}").bar(color='#FCF3CF').set_properties(**{'background-color': '#FEF9E7', 'border': '1px solid #FCF3CF', 'color': '#7D6608'}), width='stretch')
                
                st.success("✅ **Conclusión:** Las divergencias estadísticamente significativas entre la frecuencia observada en la Moda y el Valor Esperado sugieren una distribución asimétrica que se aleja de la equiprobabilidad estricta, marcando un sesgo latente en el entorno muestral original.")
        else:
            st.warning("No hay variables cualitativas en el dataset")
    
# ========================================================================
    # TAB 4: ANÁLISIS MULTIVARIADO
    # ========================================================================
    with tab4:
        st.header("Análisis Multivariado")
        
        quantitative_vars = var_types.get('Cuantitativa', [])
        qualitative_vars = var_types.get('Cualitativa', [])
        
        analysis_type = st.radio(
            "Tipo de análisis multivariado",
            ["Correlación entre Variables Cuantitativas", 
             "Diagrama de Dispersión (Bivariado)",
             "Comparación: Variable Cuantitativa vs Cualitativa"],
            horizontal=True
        )
        
        if analysis_type == "Correlación entre Variables Cuantitativas":
            if len(quantitative_vars) >= 2:
                st.subheader("Matriz de Correlación")
                
                st.info("💡 **Concepto Didáctico:** El coeficiente de correlación de Pearson evalúa formalmente la fuerza y vector direccional de la interdependencia estandarizada lineal entre variables continuas de manera simultánea.")
                st.info(r"""### 📐 Nota Educativa: Coeficiente de Correlación de Pearson
**Definición:** Métrica algorítmica que cuantifica la dependencia lineal estructural entre dos vectores cuantitativos independientes.

$$r_{xy} = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}$$
""")
                st.warning("""### 💻 Nota Educativa: Implementación en Python
**Definición:** Construcción subyacente de la matriz relacional y renderizado térmico superficial a través de Seaborn.

```python
import pandas as pd
import seaborn as sns

corr_matrix = df[['columna_1', 'columna_2']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
```""")
                
                corr_matrix = df[quantitative_vars].corr()
                
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='vlag',
                           center=0, vmin=-1, vmax=1, ax=ax, 
                           cbar_kws={'label': 'Coeficiente de Correlación'},
                           square=True, linewidths=0.5, linecolor='white')
                ax.set_title("Matriz de Correlación - Variables Cuantitativas", fontsize=14, fontweight='bold')
                st.pyplot(fig, use_container_width=True)
                
                st.subheader("Matriz de Correlación (Tabla)")
                corr_matrix.index.name = None
                st.dataframe(corr_matrix.style.format("{:,.6f}").background_gradient(cmap='PRGn', vmin=-1, vmax=1).set_properties(**{'border': '1px solid #EAECEE', 'color': '#17202A'}), width='stretch')
                
                st.success("✅ **Conclusión:** Los bloques perimetrales marcados con extremada intensidad de color (cercanos a 1 o -1) advierten sobre escenarios de multicolinealidad estructural; depure las variables redundantes si procede a entrenar modelos predictivos estocásticos.")
            else:
                st.warning("Se necesitan al menos 2 variables cuantitativas para análisis de correlación")
                
        elif analysis_type == "Diagrama de Dispersión (Bivariado)":
            if len(quantitative_vars) >= 2:
                st.subheader("Diagrama de Dispersión")
                
                st.info("💡 **Concepto Didáctico:** El diagrama de dispersión proyecta pares iterados cartesianos de valores bi-dimensionales; resulta invaluable facilitando la identificación instintiva de clusters, tendencias paramétricas subyacentes y valores atípicos bivariados atípicos.")
                col1, col2 = st.columns(2)
                with col1:
                    var_x = st.selectbox("Variable Eje X", quantitative_vars, index=0)
                with col2:
                    var_y = st.selectbox("Variable Eje Y", quantitative_vars, index=1 if len(quantitative_vars) > 1 else 0)
                
                fig = px.scatter(df, x=var_x, y=var_y, title=f"Correlación Visual: {var_x} vs {var_y}", opacity=0.7, template="plotly_white")
                fig.update_traces(marker=dict(size=6, line=dict(width=1, color='DarkSlateGrey')))
                st.plotly_chart(fig, use_container_width=True)
                
                st.success("✅ **Conclusión:** La consolidación central de los puntos intercepta el análisis correlacional previo; las trayectorias compactas sin dispersión residual avalan robustamente la dependencia lineal bivariada deducida.")
            else:
                st.warning("Se necesitan al menos 2 variables cuantitativas para generar un diagrama de dispersión.")
        
        else:
            if len(quantitative_vars) > 0 and len(qualitative_vars) > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    quant_var = st.selectbox("Variable Cuantitativa", quantitative_vars)
                
                with col2:
                    qual_var = st.selectbox("Variable Cualitativa", qualitative_vars)
                
                st.subheader(f"{quant_var} por {qual_var}")
                
                st.info("💡 **Concepto Didáctico:** Descomponer una variable cuantitativa continua seccionándola a través de perfiles categóricos expone divergencias sustanciales entre distribuciones, medias poblacionales y niveles de varianza intragrupal.")
                
                st.info(r"""### 📐 Nota Educativa: Rango Intercuartílico Condicional
**Definición:** Medida de dispersión estadística segregada por particiones discretas para evaluar la variabilidad intragrupal.

$$\text{RIC}_g = Q_{3,g} - Q_{1,g} \quad \text{donde } g \text{ representa cada categoría}$$
""")

                st.warning(r"""### 💻 Nota Educativa: Implementación en Python
**Definición:** Generación de un diagrama de caja condicionado utilizando la gramática de gráficos de Plotly Express.

```python
import plotly.express as px

px.box(df, x='columna_categorica', y='columna_cuantitativa')
```""")
                
                fig = px.box(df, x=qual_var, y=quant_var, title=f"Distribución de {quant_var} por {qual_var}")
                fig.update_layout(xaxis_title=qual_var, yaxis_title=quant_var, height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader(f"Estadísticas de {quant_var} por {qual_var}")
                grouped_stats = df.groupby(qual_var)[quant_var].describe()
                grouped_stats.index.name = None
                st.dataframe(grouped_stats.style.format("{:,.6f}").bar(color='#D5D8DC').set_properties(**{'background-color': '#F2F3F4', 'border': '1px solid #E5E7E9', 'color': '#212F3D'}), width='stretch')
                st.success("✅ **Recomendación:** Verifique los intervalos de interposición visual en los perfiles boxplot; una total disociación asimétrica de la estructura de cuartiles es un hallazgo concluyente de segmentación natural.")
            else:
                st.warning("Se necesita al menos una variable cuantitativa y una cualitativa")
else:
    st.info("👈 Por favor, seleccione una fuente de datos o cargue un archivo CSV desde la barra lateral para comenzar el análisis")

# ============================================================================
# PIE DE PÁGINA
# ============================================================================
st.markdown("---")
st.markdown("""
**Dashboard EDA - Análisis Exploratorio de Datos**  
Basado en: *Practical Statistics for Data Scientists* - Capítulo 1  
Desarrollado con Streamlit, Pandas, Matplotlib y Seaborn
""")