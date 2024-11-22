# graph_functions.py
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from PyQt5.QtGui import QPixmap
import os
import pandas as pd
import seaborn as sns
from scipy.stats import skew, kurtosis, norm
import numpy as np
import itertools
from io import BytesIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import io
from tkinter import messagebox
from Clase_turbinaV2 import Turbina
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import PhotoImage
from functools import partial
from PIL import Image, ImageTk

def mostrar_imagen_en_etiqueta(ruta_imagen, etiqueta):
    """
    Redimensiona la imagen para ajustarse a la etiqueta de Tkinter y la establece como imagen de la etiqueta.
    
    Args:
        ruta_imagen (str): Ruta del archivo de imagen.
        etiqueta (tk.Label): Etiqueta de Tkinter donde se mostrará la imagen.
    """
    try:
        # Eliminar la imagen anterior, si existe
        etiqueta.config(image=None)
        etiqueta.image = None  # Liberar la referencia a la imagen anterior

        # Cargar la imagen usando PIL
        imagen = Image.open(ruta_imagen)

        # Obtener el tamaño de la etiqueta
        ancho_etiqueta = etiqueta.winfo_width()
        alto_etiqueta = (etiqueta.winfo_height())-4

        if ancho_etiqueta == 1 or alto_etiqueta == 1:
            # Manejo en caso de que el tamaño de la etiqueta no esté inicializado
            print("Advertencia: El tamaño de la etiqueta aún no está inicializado. Intente después de mostrar la ventana.")
            return

        # Redimensionar la imagen para ajustarse al tamaño exacto de la etiqueta
        imagen_redimensionada = imagen.resize((ancho_etiqueta, alto_etiqueta), Image.Resampling.LANCZOS)

        # Convertir la imagen redimensionada a un formato compatible con Tkinter
        imagen_tk = ImageTk.PhotoImage(imagen_redimensionada)

        # Configurar la imagen en la etiqueta
        etiqueta.config(image=imagen_tk)  # No ajustar width ni height
        etiqueta.image = imagen_tk  # Mantener una referencia para evitar la recolección de basura
    except Exception as e:
        print(f"Error al mostrar la imagen en la etiqueta: {e}")

def graficar_datos_en_etiqueta(instancia, datos, etiqueta_y, titulo, ajustar_unidades=False):
    """Graficar los datos y mostrarlos en la QLabel con estadísticas en QTextEdit."""
    
    # Ajustar las unidades para los datos de 'Nivel' si es necesario
    if ajustar_unidades:
        datos = datos.copy()  # Copiar para evitar modificar los datos originales
        datos["Valor"] = datos["Valor"] / 100
    
    # Crear el gráfico
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(datos['Fecha'], datos['Valor'])
    ax.set_xlabel('Fecha')
    ax.set_ylabel(etiqueta_y)
    ax.set_title(titulo)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Guardar el gráfico como un archivo temporal
    nombre_temporal = "grafico_temporal.png"
    fig.savefig(nombre_temporal)

    # Mostrar la imagen en la etiqueta correspondiente
    mostrar_imagen_en_etiqueta(nombre_temporal, instancia.lblGraph)

    # Eliminar el archivo temporal
    os.remove(nombre_temporal)
    
    # Mostrar estadísticas en QTextEdit
    estadísticas = datos.describe()
    instancia.mostrar_información(f"Estadísticas de los datos de {etiqueta_y}:\n{estadísticas.to_string()}")
    
    # Cerrar la figura para liberar memoria
    plt.close(fig)


def graficar_mapa_de_calor(instancia, conjunto_datos, etiqueta):
    """Graficar un mapa de calor para valores faltantes en el conjunto de datos y mostrarlo en lblGraph."""
    conjunto_datos['Fecha'] = pd.to_datetime(conjunto_datos['Fecha'], errors='coerce')
    conjunto_datos['Año'] = conjunto_datos['Fecha'].dt.year
    conjunto_datos['DíaDelAño'] = conjunto_datos['Fecha'].dt.dayofyear

    # Agrupar el conjunto de datos por Año y Día del Año
    conjunto_agrupado = conjunto_datos.groupby(['Año', 'DíaDelAño'])['Valor'].mean().reset_index()

    # Crear la tabla pivote para el mapa de calor
    conjunto_pivote = conjunto_agrupado.pivot(index='Año', columns='DíaDelAño', values='Valor')

    # Crear el mapa de calor
    plt.figure(figsize=(10, 6))
    sns.heatmap(conjunto_pivote.isnull(), cmap=sns.color_palette(["#add8e6", "#000000"]), cbar=False)
    plt.title(f'Visualización de Valores Faltantes en el DataFrame de {etiqueta}')
    plt.xlabel('Día del Año')
    plt.ylabel('Año')
    
    # Guardar el gráfico como archivo temporal
    nombre_temporal = "mapa_calor_temporal.png"
    plt.savefig(nombre_temporal)

    # Mostrar la imagen en la etiqueta correspondiente
    mostrar_imagen_en_etiqueta(nombre_temporal, instancia.lblGraph)

    # Eliminar el archivo temporal
    os.remove(nombre_temporal)

    # Crear la descripción estadística y mostrarla
    descripción = conjunto_agrupado['Valor'].describe()
    información_grafico = ''
    información_grafico += f"Datos de {etiqueta} - Mapa de Calor de Valores Faltantes:\n"
    información_grafico += f"Visualización de datos faltantes por día del año a través de los años.\n"
    información_grafico += f"Estadísticas Descriptivas de los Datos de {etiqueta}:\n"
    información_grafico += f"Conteo: {descripción['count']}\nPromedio: {descripción['mean']}\nDesviación Estándar: {descripción['std']}\nMínimo: {descripción['min']}\nMáximo: {descripción['max']}"
    instancia.mostrar_información(f"{información_grafico}")

def graficar_dispersión_caudal_por_década(instancia=None, instancia_resultados=None, titulo='Serie Decadal de Caudal', décadas=[], bandera=False):
    """Generar y mostrar un gráfico de dispersión de datos de caudal para una década específica."""
    try:
        if bandera:
            décadas = décadas[0]
            print(décadas)

        # Ordenar los datos por la columna 'Fecha'
        caudal_procesado = instancia.caudal_process.sort_values(by='Fecha')
        fin_década = décadas + 9

        # Filtrar los datos para la década seleccionada
        datos_década = caudal_procesado[
            (caudal_procesado['Fecha'].dt.year >= décadas) & (caudal_procesado['Fecha'].dt.year <= fin_década)
        ]

        if datos_década.empty:
            if bandera:
                instancia_resultados.registrar_mensaje(f"No hay datos disponibles para la década de {décadas}.")
            else:
                instancia.registrar_mensaje(f"No hay datos disponibles para la década de {décadas}.")
            return

        # Crear el gráfico de dispersión
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(datos_década['Fecha'], datos_década['Valor'], label=f'Década {décadas}s', alpha=0.5)
        ax.set_title(f'{titulo} - Década {décadas}s')
        ax.set_xlabel('Fecha')
        ax.set_ylabel('Caudal Medio (m³/s)')
        ax.legend()

        # Formatear el eje x
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.xticks(rotation=45)
        ax.grid(True)
        plt.tight_layout()

        # Guardar el gráfico como archivo temporal
        nombre_temporal = "grafico_temporal.png"
        fig.savefig(nombre_temporal)

        # Usar la función auxiliar para mostrar la imagen en la etiqueta correspondiente
        if bandera:
            mostrar_imagen_en_etiqueta(nombre_temporal, instancia_resultados.lblGraph)
        else:
            mostrar_imagen_en_etiqueta(nombre_temporal, instancia.lblGraph)

        # Mostrar estadísticas en el widget correspondiente
        estadísticas = datos_década['Valor'].describe()
        mensaje_estadísticas = f"{titulo} - Década {décadas}s\n\nEstadísticas:\n{estadísticas.to_string()}"

        if bandera:
            instancia_resultados.limpiar_información()
            instancia_resultados. mostrar_información(mensaje_estadísticas)
        else:
            instancia.limpiar_información()
            instancia.mostrar_información(mensaje_estadísticas)

        # Eliminar el archivo temporal
        os.remove(nombre_temporal)
        plt.close(fig)

    except Exception as e:
        if bandera:
            instancia_resultados.registrar_mensaje(f"Error en graficar_dispersión_caudal_por_década: {e}")
        else:
            instancia.registrar_mensaje(f"Error en graficar_dispersión_caudal_por_década: {e}")
def graficar_dispersión_nivel_por_década(instancia=None, instancia_resultados=None, titulo='Serie Decadal de Nivel', décadas=[], bandera=False):
    """Generar y mostrar un gráfico de dispersión de datos de nivel para una década específica."""
    try:
        if bandera:
            décadas = décadas[0]  # Usar el primer valor de las décadas si bandera es True

        # Renombrar las columnas de los DataFrames
        df_copycaudal = instancia.caudal_process
        df_copycaudal = df_copycaudal.rename(columns={'Valor': 'Caudal'})
        df_copynivel = instancia.nivel_process
        df_copynivel = df_copynivel.rename(columns={'Valor': 'Nivel'})

        # Unir los DataFrames de caudal y nivel por la columna 'Fecha'
        instancia.df_merge = pd.merge(df_copycaudal, df_copynivel, on='Fecha', how='outer')

        # Ordenar los datos por 'Fecha'
        nivel_procesado = instancia.nivel_process.sort_values(by='Fecha')
        fin_década = décadas + 9

        # Filtrar los datos para la década seleccionada
        datos_década = nivel_procesado[
            (nivel_procesado['Fecha'].dt.year >= décadas) & (nivel_procesado['Fecha'].dt.year <= fin_década)
        ]

        if datos_década.empty:
            if bandera:
                instancia_resultados.registrar_mensaje(f"No hay datos disponibles para la década de {décadas}.")
            else:
                instancia.registrar_mensaje(f"No hay datos disponibles para la década de {décadas}.")
            return

        # Crear el gráfico de dispersión
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(datos_década['Fecha'], datos_década['Valor'], label=f'Década {décadas}s', alpha=0.5)
        ax.set_title(f'{titulo} - Década {décadas}s')
        ax.set_xlabel('Fecha')
        ax.set_ylabel('Nivel Medio (m)')
        ax.legend()

        # Formatear el eje x
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.xticks(rotation=45)
        ax.grid(True)
        plt.tight_layout()

        # Guardar el gráfico como archivo temporal
        nombre_temporal = "grafico_temporal.png"
        fig.savefig(nombre_temporal)

        # Usar la función auxiliar para mostrar la imagen en la etiqueta correspondiente
        if bandera:
            mostrar_imagen_en_etiqueta(nombre_temporal, instancia_resultados.lblGraph)
        else:
            mostrar_imagen_en_etiqueta(nombre_temporal, instancia.lblGraph)

        # Mostrar estadísticas en el widget correspondiente
        estadísticas = datos_década['Valor'].describe()
        mensaje_estadísticas = f"{titulo} - Década {décadas}s\n\nEstadísticas:\n{estadísticas.to_string()}"

        if bandera:
            instancia_resultados.limpiar_información()
            instancia_resultados. mostrar_información(mensaje_estadísticas)
        else:
            instancia.limpiar_información()
            instancia.mostrar_información(mensaje_estadísticas)

        # Eliminar el archivo temporal
        os.remove(nombre_temporal)
        plt.close(fig)

    except Exception as e:
        if bandera:
            instancia_resultados.registrar_mensaje(f"Error en graficar_dispersión_nivel_por_década: {e}")
        else:
            instancia.registrar_mensaje(f"Error en graficar_dispersión_nivel_por_década: {e}")
def graficar_dispersion_anual_caudal(instancia=None, instancia_resultados=None, titulo='Serie Anual de Caudal', décadas=[], bandera=False):
    """Generar y mostrar un gráfico de dispersión de datos de caudal por año para todos los años."""
    try:
        

        # Ordenar los datos de caudal por la columna 'Fecha'
        caudal_procesado = instancia.caudal_process.sort_values(by='Fecha')

        # Obtener los años únicos de los datos
        años = caudal_procesado['Fecha'].dt.year.unique()

        # Crear el gráfico de dispersión
        fig, axs = plt.subplots(nrows=(len(años) + 1) // 2, ncols=2, figsize=(10, (len(años) // 2 + 1) * 3))

        # Aplanar los ejes para facilitar la iteración si hay más de una fila
        axs = axs.flatten()

        # Iterar sobre los años y crear los gráficos de dispersión en los subgráficos
        for i, año in enumerate(años):
            datos_año = caudal_procesado[caudal_procesado['Fecha'].dt.year == año]

            if not datos_año.empty:
                ax = axs[i]  # Obtener el eje correspondiente al año actual
                ax.scatter(datos_año['Fecha'], datos_año['Valor'], label=f'Año {año}', alpha=0.5)
                ax.set_title(f'Año {año}')
                ax.set_xlabel('Fecha')
                ax.set_ylabel('Valor en m³/s')

                # Formatear el eje x para mostrar los meses
                ax.xaxis.set_major_locator(mdates.MonthLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # Mostrar los nombres de los meses abreviados
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True)

        # Eliminar cualquier subgráfico vacío si el número de años es impar
        if len(años) % 2 != 0:
            fig.delaxes(axs[-1])

        # Ajustar el diseño para un mejor espaciado
        plt.tight_layout()

        # Guardar el gráfico como archivo temporal
        nombre_temporal = "grafico_temporal.png"
        fig.savefig(nombre_temporal)

        # Usar la función auxiliar para mostrar la imagen en la etiqueta correspondiente
        if bandera:
            mostrar_imagen_en_etiqueta(nombre_temporal, instancia_resultados.lblGraph)
        else:
            mostrar_imagen_en_etiqueta(nombre_temporal, instancia.lblGraph)

        # Mostrar estadísticas en el widget correspondiente
        estadísticas = caudal_procesado['Valor'].describe()
        mensaje_estadísticas = f"{titulo} ({décadas}s)\n\nEstadísticas:\n{estadísticas.to_string()}"

        if bandera:
            instancia_resultados.limpiar_información()
            instancia_resultados. mostrar_información(mensaje_estadísticas)
        else:
            instancia.limpiar_información()
            instancia.mostrar_información(mensaje_estadísticas)

        # Eliminar el archivo temporal
        os.remove(nombre_temporal)
        plt.close(fig)

    except Exception as e:
        if bandera:
            instancia_resultados.registrar_mensaje(f"Error en graficar_dispersión_nivel_por_década: {e}")
        else:
            instancia.registrar_mensaje(f"Error en graficar_dispersión_nivel_por_década: {e}")
def graficar_dispersión_anual_nivel(instancia=None, instancia_resultados=None, titulo='Serie Anual de Nivel', décadas=[], bandera=False):
    """Generar y mostrar un gráfico de dispersión de datos de nivel para todos los años."""
    try:
        
        # Ordenar los datos de nivel por la columna 'Fecha'
        nivel_procesado = instancia.nivel_process.sort_values(by='Fecha')

        # Obtener los años únicos de los datos
        años = nivel_procesado['Fecha'].dt.year.unique()

        # Crear el gráfico de dispersión
        fig, axs = plt.subplots(nrows=(len(años) + 1) // 2, ncols=2, figsize=(10, (len(años) // 2 + 1) * 3))

        # Aplanar los ejes para facilitar la iteración si hay más de una fila
        axs = axs.flatten()

        # Iterar sobre los años y crear los gráficos de dispersión en los subgráficos
        for i, año in enumerate(años):
            datos_año = nivel_procesado[nivel_procesado['Fecha'].dt.year == año]

            if not datos_año.empty:
                ax = axs[i]  # Obtener el eje correspondiente al año actual
                ax.scatter(datos_año['Fecha'], datos_año['Valor'], label=f'Año {año}', alpha=0.5)
                ax.set_title(f'Año {año}')
                ax.set_xlabel('Fecha')
                ax.set_ylabel('Nivel Medio (cm)')

                # Formatear el eje x para mostrar los meses
                ax.xaxis.set_major_locator(mdates.MonthLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # Mostrar los nombres de los meses abreviados
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True)

        # Eliminar cualquier subgráfico vacío si el número de años es impar
        if len(años) % 2 != 0:
            fig.delaxes(axs[-1])

        # Ajustar el diseño para un mejor espaciado
        plt.tight_layout()

        # Guardar el gráfico como archivo temporal
        nombre_temporal = "grafico_temporal.png"
        fig.savefig(nombre_temporal)

        # Usar la función auxiliar para mostrar la imagen en la etiqueta correspondiente
        if bandera:
            mostrar_imagen_en_etiqueta(nombre_temporal, instancia_resultados.lblGraph)
        else:
            mostrar_imagen_en_etiqueta(nombre_temporal, instancia.lblGraph)

        # Mostrar estadísticas en el widget correspondiente
        estadísticas = nivel_procesado['Valor'].describe()
        mensaje_estadísticas = f"{titulo} ({décadas}s)\n\nEstadísticas:\n{estadísticas.to_string()}"

        if bandera:
            instancia_resultados.limpiar_información()
            instancia_resultados. mostrar_información(mensaje_estadísticas)
        else:
            instancia.limpiar_información()
            instancia.mostrar_información(mensaje_estadísticas)

        # Eliminar el archivo temporal
        os.remove(nombre_temporal)
        plt.close(fig)

    except Exception as e:
        if bandera:
            instancia_resultados.registrar_mensaje(f"Error en graficar_dispersión_anual_nivel: {e}")
        else:
            instancia.registrar_mensaje(f"Error en graficar_dispersión_anual_nivel: {e}")
def mostrar_estadísticas(instancia=None, instancia_resultados=None, titulo="Estadísticas", décadas=[], bandera=False):
    """
    Calcular y mostrar las estadísticas para los datos de caudal y nivel en el widget QTextEdit proporcionado.
    :param info_textedit: QTextEdit donde se mostrarán las estadísticas.
    """
    # Lista de años a eliminar
    años_a_eliminar = [1969, 1985, 1987, 1991, 2001, 2002, 2003, 2004]

    # Filtrar los datos de caudal y nivel excluyendo los años especificados
    caudal_sel = instancia.caudal_process[~instancia.caudal_process['Fecha'].dt.year.isin(años_a_eliminar)]
    nivel_sel = instancia.nivel_process[~instancia.nivel_process['Fecha'].dt.year.isin(años_a_eliminar)]

    # Calcular las estadísticas para caudal
    caudal_media = caudal_sel['Valor'].mean()
    caudal_std = caudal_sel['Valor'].std()
    caudal_rango = caudal_sel['Valor'].max() - caudal_sel['Valor'].min()
    caudal_cv = (caudal_std / caudal_media) * 100

    # Calcular las estadísticas para nivel
    nivel_media = nivel_sel['Valor'].mean()
    nivel_std = nivel_sel['Valor'].std()
    nivel_rango = nivel_sel['Valor'].max() - nivel_sel['Valor'].min()
    nivel_cv = (nivel_std / nivel_media) * 100

    # Formatear las estadísticas para mostrar
    estadísticas_caudal = (
        f"Estadísticas de Caudal:\n"
        f"- Media: {caudal_media:.2f}\n"
        f"- Desviación Estándar: {caudal_std:.2f}\n"
        f"- Rango: {caudal_rango:.2f}\n"
        f"- Coeficiente de Variación: {caudal_cv:.2f}%\n\n"
    )

    estadísticas_nivel = (
        f"Estadísticas de Nivel:\n"
        f"- Media: {nivel_media:.2f}\n"
        f"- Desviación Estándar: {nivel_std:.2f}\n"
        f"- Rango: {nivel_rango:.2f}\n"
        f"- Coeficiente de Variación: {nivel_cv:.2f}%\n"
    )

    # Formar el mensaje de estadísticas
    mensaje_estadísticas = estadísticas_caudal + estadísticas_nivel

    # Mostrar las estadísticas en el widget correspondiente
    if bandera:
        instancia_resultados.limpiar_información()
        instancia_resultados. mostrar_información(mensaje_estadísticas)
    else:
        instancia.limpiar_información()
        instancia.mostrar_información(mensaje_estadísticas)

def graficar_distribucion_caudal(instancia=None, instancia_resultados=None, titulo='', décadas=[], bandera=False):
    """Graficar histograma y KDE de los datos de caudal con superposición de distribución normal en lblGraph y mostrar asimetría y curtosis en graphInformation."""
    try:
        if bandera == True:
            print("Bandera Verdadera")
        # Filtrar valores NaN
        datos_caudal = instancia.caudal_process['Valor'].dropna()

        # Calcular asimetría y curtosis
        asimetria = skew(datos_caudal)
        curtosis_valor = kurtosis(datos_caudal)

        # Crear histograma y gráfico KDE
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(datos_caudal, kde=True, bins=30, color='skyblue', stat="density", linewidth=0, ax=ax)

        # Calcular y graficar la superposición de la distribución normal
        mu, std = datos_caudal.mean(), datos_caudal.std()
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        ax.plot(x, p, 'r', linewidth=2, label=f'Dist. Normal\nμ={mu:.2f}, σ={std:.2f}')

        # Configurar títulos, etiquetas y leyenda
        ax.set_title(titulo)
        ax.set_xlabel('Tasa de flujo (Caudal)')
        ax.set_ylabel('Densidad')
        ax.legend()

        # Convertir gráfico a QPixmap y mostrar en lblGraph
        nombre_temporal = "grafico_temporal.png"
        fig.savefig(nombre_temporal)

        # Convertir la imagen guardada a QPixmap y mostrarla usando la función auxiliar
        if bandera:
            mostrar_imagen_en_etiqueta(nombre_temporal, instancia_resultados.lblGraph)
        else:
            mostrar_imagen_en_etiqueta(nombre_temporal, instancia.lblGraph)

        # Preparar el mensaje de estadísticas a mostrar
        mensaje_estadisticas = (
            f"{titulo}\n"
            f"Asimetría (Skewness): {asimetria:.2f}\n"
            f"Curtosis (Kurtosis): {curtosis_valor:.2f}\n"
            f"Nota: La asimetría mide la simetría; la curtosis indica la pesadez de las colas."
        )

        # Mostrar las estadísticas en la etiqueta correspondiente
        if bandera:
            instancia_resultados.limpiar_información()
            instancia_resultados. mostrar_información(mensaje_estadisticas)
        else:
            instancia.limpiar_información()
            instancia.mostrar_información(mensaje_estadisticas)

        # Registrar la acción

        # Eliminar el archivo temporal
        os.remove(nombre_temporal)
        plt.close(fig)

    except Exception as e:
        # Registrar error
        if bandera:
            instancia_resultados.registrar_mensaje(f"Error en graficar_distribucion_caudal: {e}")
        else:
            instancia.registrar_mensaje(f"Error en graficar_distribucion_caudal: {e}")
def graficar_distribución_nivel(instancia=None, instancia_resultados=None, titulo='', décadas=[], bandera=False):
    """Graficar el histograma y KDE de los datos de nivel con superposición de distribución normal en lblGraph y mostrar asimetría y curtosis en graphInformation."""
    try:
        # Filtrar valores NaN
        datos_nivel = instancia.nivel_process['Valor'].dropna()

        # Calcular asimetría y curtosis
        asimetría = skew(datos_nivel)
        curtosis = kurtosis(datos_nivel)

        # Crear el gráfico de histograma y KDE
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(datos_nivel, kde=True, bins=30, color='skyblue', stat="density", linewidth=0, ax=ax)

        # Calcular y graficar la superposición de la distribución normal
        mu, std = datos_nivel.mean(), datos_nivel.std()
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        ax.plot(x, p, 'r', linewidth=2, label=f'Dist. Normal\nμ={mu:.2f}, σ={std:.2f}')

        # Establecer títulos, etiquetas y leyenda
        ax.set_title(titulo)
        ax.set_xlabel('Nivel')
        ax.set_ylabel('Densidad')
        ax.legend()

        # Guardar el gráfico como archivo temporal
        nombre_temporal = "grafico_temporal.png"
        fig.savefig(nombre_temporal)

        # Usar la función auxiliar para mostrar la imagen en la etiqueta correspondiente
        if bandera:
            mostrar_imagen_en_etiqueta(nombre_temporal, instancia_resultados.lblGraph)
        else:
            mostrar_imagen_en_etiqueta(nombre_temporal, instancia.lblGraph)

        # Mostrar estadísticas en el widget correspondiente
        mensaje_estadísticas = (
            f"{titulo}\n"
            f"Asimetría (Skewness): {asimetría:.2f}\n"
            f"Curtosis (Kurtosis): {curtosis:.2f}\n"
            f"Nota: La asimetría mide la simetría de la distribución; la curtosis indica la densidad de las colas."
        )

        if bandera:
            instancia_resultados.limpiar_información()
            instancia_resultados. mostrar_información(mensaje_estadísticas)
        else:
            instancia.limpiar_información()
            instancia.mostrar_información(mensaje_estadísticas)

        # Eliminar el archivo temporal
        os.remove(nombre_temporal)

        # Cerrar la figura
        plt.close(fig)

    except Exception as e:
        # Registrar error
        if bandera: 
            instancia_resultados.registrar_mensaje(f"Error en graficar_distribución_nivel: {e}")
        else:
            instancia.registrar_mensaje(f"Error en graficar_distribución_nivel: {e}")
def graficar_densidad_probabilidad_caudal_por_década(instancia=None, instancia_resultados=None, titulo='', décadas=[], bandera=False):
    """Generar gráficos de densidad de probabilidad para los datos de caudal para cada década en lblGraph."""
    try:
        if not bandera:
            décadas = [décadas]
        
        # Recorrer cada década en la lista de décadas
        for inicio_década in décadas:
            fin_década = inicio_década + 9
            datos_década = instancia.caudal_process[ 
                (instancia.caudal_process['Fecha'].dt.year >= inicio_década) & 
                (instancia.caudal_process['Fecha'].dt.year <= fin_década)
            ]

            # Verificar si hay datos para la década
            if datos_década.empty:
                if bandera:
                    instancia_resultados.registrar_mensaje(f"No hay datos disponibles para la década de {inicio_década}s.")
                else:
                    instancia.registrar_mensaje(f"No hay datos disponibles para la década de {inicio_década}s.")
                continue

            # Crear gráfico KDE para densidad de probabilidad
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.kdeplot(data=datos_década['Valor'], fill=True, label=f'{inicio_década}s', alpha=0.5, ax=ax)
            ax.set_title(f'{titulo} - {inicio_década}s')
            ax.set_xlabel("Caudal Promedio (m³/s)")
            ax.set_ylabel("Densidad de Probabilidad - Caudal")
            ax.legend()

            # Guardar el gráfico como archivo temporal
            nombre_temporal = "grafico_temporal.png"
            fig.savefig(nombre_temporal)
            if bandera:
                mostrar_imagen_en_etiqueta(nombre_temporal, instancia_resultados.lblGraph)
            else:
                mostrar_imagen_en_etiqueta(nombre_temporal, instancia.lblGraph)

            estadísticas = datos_década['Valor'].describe()
            mensaje_estadísticas = f"{titulo} ({inicio_década}s)\nEstadísticas:\n{estadísticas.to_string()}"

            if bandera:
                instancia_resultados.limpiar_información()
                instancia_resultados. mostrar_información(mensaje_estadísticas)
            else:
                instancia.limpiar_información()
                instancia.mostrar_información(mensaje_estadísticas)

            # Eliminar el archivo temporal
            os.remove(nombre_temporal)

            # Cerrar la figura para liberar memoria
            plt.close(fig)

    except Exception as e:
        # Registrar el error y también imprimirlo en la consola para debug
        if bandera:
            instancia_resultados.registrar_mensaje(f"Error en graficar_densidad_probabilidad_caudal_por_década: {e}")
        else:
            instancia.registrar_mensaje(f"Error en graficar_densidad_probabilidad_caudal_por_década: {e}")
        print(f"Error en graficar_densidad_probabilidad_caudal_por_década: {e}")  # This will print the error to the console

def graficar_densidad_probabilidad_nivel_por_década(instancia=None, instancia_resultados=None,titulo='', décadas=[], bandera=False):
    """Generar gráficos de densidad de probabilidad para los datos de nivel para cada década en lblGraph."""
    try:
        if not bandera:
            décadas = [décadas]
        
        for inicio_década in décadas:
            fin_década = inicio_década + 9
            datos_década = instancia.nivel_process[
                (instancia.nivel_process['Fecha'].dt.year >= inicio_década) & 
                (instancia.nivel_process['Fecha'].dt.year <= fin_década)
            ]

            if datos_década.empty:
                if bandera:
                    instancia_resultados.registrar_mensaje(f"No hay datos disponibles para la década de {inicio_década}s.")
                else:
                    instancia.registrar_mensaje(f"No hay datos disponibles para la década de {inicio_década}s.")
                continue

            # Crear gráfico KDE para densidad de probabilidad
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.kdeplot(data=datos_década['Valor'], fill=True, label=f'{inicio_década}s', alpha=0.5, ax=ax)
            ax.set_title(f'{titulo} - {inicio_década}s')
            ax.set_xlabel("Nivel Promedio (m)")
            ax.set_ylabel("Densidad de Probabilidad - Nivel")
            ax.legend()

            # Guardar el gráfico como archivo temporal
            nombre_temporal = "grafico_temporal.png"
            fig.savefig(nombre_temporal)

            # Usar la función auxiliar para mostrar la imagen en la etiqueta correspondiente
            if bandera:
                mostrar_imagen_en_etiqueta(nombre_temporal, instancia_resultados.lblGraph)
                instancia_resultados.limpiar_información()
                instancia_resultados. mostrar_información(f"{titulo} ({inicio_década}s)\nDensidad de Probabilidad por Década")
            else:
                mostrar_imagen_en_etiqueta(nombre_temporal, instancia.lblGraph)
                instancia.limpiar_información()
                instancia.mostrar_información(f"{titulo} ({inicio_década}s)\nDensidad de Probabilidad por Década")

            # Eliminar el archivo temporal
            os.remove(nombre_temporal)

            # Cerrar la figura para liberar memoria
            plt.close(fig)

    except Exception as e:
        # Registrar error
        if bandera:
            instancia_resultados.registrar_mensaje(f"Error en graficar_densidad_probabilidad_nivel_por_década: {e}")
        else:
            instancia.registrar_mensaje(f"Error en graficar_densidad_probabilidad_nivel_por_década: {e}")
def graficar_comportamiento_anual_por_década_caudal(instancia=None, instancia_resultados=None,titulo='', décadas=[], bandera=False):
    """Generar gráfico de comportamiento anual de caudal para cada década en lblGraph."""
    try:
        if not bandera:
            décadas = [décadas]
        
        # Ciclo de colores y estilos de línea para diferenciación visual
        colores = itertools.cycle(['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'brown'])
        estilos_linea = itertools.cycle(['-', '--', '-.', ':'])

        # Crear una nueva columna para "DiaMes" (Día y Mes) para fines de graficado
        instancia.caudal_process['DiaMes'] = instancia.caudal_process['Fecha'].apply(lambda x: x.replace(year=2000))

        # Recorrer cada década especificada y graficar los datos para cada año dentro de esa década
        for inicio_década in décadas:
            fin_década = inicio_década + 9
            datos_década = instancia.caudal_process[
                (instancia.caudal_process['Fecha'].dt.year >= inicio_década) & 
                (instancia.caudal_process['Fecha'].dt.year <= fin_década)
            ]

            if datos_década.empty:
                if bandera:
                    instancia_resultados.registrar_mensaje(f"No hay datos disponibles para la década de {inicio_década}s.")
                else:
                    instancia.registrar_mensaje(f"No hay datos disponibles para la década de {inicio_década}s.")
                continue

            # Crear una figura para cada década
            fig, ax = plt.subplots(figsize=(8, 6))

            # Graficar los datos para cada año dentro de la década
            for año in range(inicio_década, fin_década + 1):
                datos_año = datos_década[datos_década['Fecha'].dt.year == año]
                if not datos_año.empty:
                    ax.plot(
                        datos_año['DiaMes'],
                        datos_año['Valor'],
                        label=str(año),
                        color=next(colores),
                        linestyle=next(estilos_linea),
                        linewidth=2,
                        marker='o',
                        markersize=4,
                        alpha=0.7
                    )

            # Título, etiquetas y leyenda
            ax.set_title(f'{titulo} - {inicio_década}s')
            ax.set_xlabel('Día y Mes')
            ax.set_ylabel('Caudal (m³/s)')
            ax.legend(title='Año', loc='upper left', bbox_to_anchor=(1, 1))
            ax.grid(True)

            # Formatear el eje X para mostrar solo día y mes
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Guardar el gráfico como archivo temporal
            nombre_temporal = "grafico_temporal.png"
            fig.savefig(nombre_temporal)
            if bandera:
                mostrar_imagen_en_etiqueta(nombre_temporal, instancia_resultados.lblGraph)
            else:
                mostrar_imagen_en_etiqueta(nombre_temporal, instancia.lblGraph)
            # Usar la función auxiliar para mostrar la imagen en la etiqueta correspondiente
            if bandera:
                instancia_resultados.limpiar_información()
                instancia_resultados. mostrar_información(f"{titulo} ({inicio_década}s)\nComportamiento Anual por Década")
            else:
                instancia.limpiar_información()
                instancia.mostrar_información(f"{titulo} ({inicio_década}s)\nComportamiento Anual por Década")

            # Eliminar el archivo temporal
            os.remove(nombre_temporal)

            # Cerrar la figura para liberar memoria
            plt.close(fig)

    except Exception as e:
        # Registrar error
        if bandera:
            instancia_resultados.registrar_mensaje(f"Error en graficar_comportamiento_anual_por_década_caudal: {e}")
        else:
            instancia.registrar_mensaje(f"Error en graficar_comportamiento_anual_por_década_caudal: {e}")
def graficar_comportamiento_anual_por_década_nivel(instancia=None, instancia_resultados=None,titulo='', décadas=[], bandera=False):
    """Generar gráfico de comportamiento anual de nivel para cada década en lblGraph."""
    try:
        if bandera:
            décadas = décadas[0]
        
        # Filtrar los datos para la década especificada
        nivel_process = instancia.nivel_process.sort_values(by='Fecha')
        fin_década = décadas + 9
        datos_década = nivel_process[(nivel_process['Fecha'].dt.year >= décadas) & 
                                     (nivel_process['Fecha'].dt.year <= fin_década)]

        if datos_década.empty:
            if bandera:
                instancia_resultados.registrar_mensaje(f"No hay datos disponibles para la década de {décadas}s.\n")
            else:
                instancia.registrar_mensaje(f"No hay datos disponibles para la década de {décadas}s.\n")
            instancia.registrar_mensaje(f"No hay datos disponibles para la década de {décadas}s.\n")
            return

        # Crear la columna DiaMes para representar día y mes
        datos_década['DiaMes'] = datos_década['Fecha'].apply(lambda x: x.replace(year=2000))

        fig, ax = plt.subplots(figsize=(8, 6))
        colores = itertools.cycle(['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'brown'])
        estilos_linea = itertools.cycle(['-', '--', '-.', ':'])

        # Graficar los datos para cada año dentro de la década
        for año in range(décadas, fin_década + 1):
            datos_año = datos_década[datos_década['Fecha'].dt.year == año]
            if not datos_año.empty:
                ax.plot(
                    datos_año['DiaMes'], 
                    datos_año['Valor'], 
                    label=str(año),
                    color=next(colores),
                    linestyle=next(estilos_linea),
                    linewidth=2,
                    marker='o',
                    markersize=4,
                    alpha=0.7
                )

        ax.set_title(f'{titulo} - {décadas}s')
        ax.set_xlabel('Día y Mes')
        ax.set_ylabel('Nivel (m)')
        ax.legend(title='Año', loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(True)
        plt.tight_layout()

        # Formatear el eje X para mostrar solo día y mes
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b-%d'))
        plt.xticks(rotation=45)

        # Guardar el gráfico como archivo temporal
        nombre_temporal = "grafico_temporal.png"
        fig.savefig(nombre_temporal)
        if bandera:
            mostrar_imagen_en_etiqueta(nombre_temporal, instancia_resultados.lblGraph)
        else:
            mostrar_imagen_en_etiqueta(nombre_temporal, instancia.lblGraph)
        # Usar la función auxiliar para mostrar la imagen en la etiqueta correspondiente
        if bandera:
            instancia_resultados.limpiar_información()
            instancia_resultados. mostrar_información(f"{titulo} ({décadas}s)\nComportamiento Anual de Nivel")
        else:
            instancia.limpiar_información()
            instancia.mostrar_información(f"{titulo} ({décadas}s)\nComportamiento Anual de Nivel")

        # Eliminar el archivo temporal
        os.remove(nombre_temporal)

        # Cerrar la figura para liberar memoria
        plt.close(fig)

    except Exception as e:
        # Registrar error
        if bandera:
            instancia_resultados.registrar_mensaje(f"Error en graficar_comportamiento_anual_por_década_nivel: {e}")
        else:
            instancia.registrar_mensaje(f"Error en graficar_comportamiento_anual_por_década_nivel: {e}")
def graficar_perfil_hidrológico_caudal(instancia=None, instancia_resultados=None, titulo="Perfil Hidrológico Anual - Caudal", décadas=[], bandera=False):
    """Generar gráfico de perfil hidrológico anual por mes para los casos seco, húmedo y normal de caudal en lblGraph."""
    try:
        # Asegurar que 'Fecha' esté en formato datetime
        instancia.caudal_process['Fecha'] = pd.to_datetime(instancia.caudal_process['Fecha'], errors='coerce')
        
        # Agregar columnas 'Mes' y 'Año'
        instancia.caudal_process['Mes'] = instancia.caudal_process['Fecha'].dt.month
        instancia.caudal_process['Year'] = instancia.caudal_process['Fecha'].dt.year

        # Función para calcular el promedio mensual
        def obtener_promedio_mensual(lista_años):
            df_filtrado = instancia.caudal_process[instancia.caudal_process['Year'].isin(lista_años)]
            return df_filtrado.groupby('Mes')['Valor'].mean()

        # Calcular promedios mensuales para cada categoría
        promedio_mensual_humedo = obtener_promedio_mensual(instancia.años_humedos)
        promedio_mensual_normal = obtener_promedio_mensual(instancia.años_normales)
        promedio_mensual_seco = obtener_promedio_mensual(instancia.años_secos)

        # Configurar la figura y graficar cada categoría como un gráfico de barras
        fig, axs = plt.subplots(3, 1, figsize=(8, 6))
        limites_y = (0, 400)
        marcas_y = range(0, 401, 50)
        etiquetas_mensuales = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']

        # Graficar cada perfil hidrológico
        for ax, datos, color, texto_titulo in zip(
            axs,
            [promedio_mensual_humedo, promedio_mensual_normal, promedio_mensual_seco],
            ['b', 'g', 'r'],
            ['Caso Húmedo', 'Caso Base', 'Caso Seco']
        ):
            barras = ax.bar(datos.index, datos.values, color=color)
            ax.set_title(f'{titulo} - {texto_titulo}', fontsize=14)
            ax.set_ylabel('Caudal Promedio (m³/s)')
            ax.set_ylim(limites_y)
            ax.set_yticks(marcas_y)
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels(etiquetas_mensuales)
            for barra in barras:
                yval = barra.get_height()
                ax.text(barra.get_x() + barra.get_width() / 2, yval + 0.1, round(yval, 2), ha='center', va='bottom')

        axs[-1].set_xlabel('Mes')
        plt.tight_layout()

        # Convertir el gráfico a QPixmap y mostrarlo en lblGraph
        nombre_temporal = "grafico_temporal.png"
        fig.savefig(nombre_temporal)
        if bandera:
            mostrar_imagen_en_etiqueta(nombre_temporal, instancia_resultados.lblGraph)
        else:
            mostrar_imagen_en_etiqueta(nombre_temporal, instancia.lblGraph)
        # Convertir la imagen guardada a QPixmap y mostrarla
        if bandera:
            instancia_resultados.limpiar_información()
            instancia_resultados. mostrar_información(f"{titulo}\nPerfil Hidrológico Mensual para los Casos Húmedo, Normal y Seco")
        else:
            instancia.limpiar_información()
            instancia.mostrar_información(f"{titulo}\nPerfil Hidrológico Mensual para los Casos Húmedo, Normal y Seco")

        # Eliminar el archivo temporal
        os.remove(nombre_temporal)
        plt.close(fig)
    except Exception as e:
        if bandera:
            instancia_resultados.registrar_mensaje(f"Error en graficar_perfil_hidrológico_caudal: {e}")
        else:
            instancia.registrar_mensaje(f"Error en graficar_perfil_hidrológico_caudal: {e}")
def graficar_perfil_hidrológico_nivel(instancia=None, instancia_resultados=None, titulo='', décadas=[], bandera=False):
    """Generar gráfico del perfil hidrológico anual (Nivel) por mes para diferentes categorías (Húmedo, Base, Seco)."""

    try:
        # Asegurar que 'Fecha' esté en formato datetime y agregar columnas para 'Mes' y 'Año'
        instancia.nivel_process['Fecha'] = pd.to_datetime(instancia.nivel_process['Fecha'], errors='coerce')
        instancia.nivel_process['Mes'] = instancia.nivel_process['Fecha'].dt.month
        instancia.nivel_process['Year'] = instancia.nivel_process['Fecha'].dt.year

        # Definir la función para obtener promedios mensuales para una lista específica de años
        def obtener_promedio_mensual(nivel_sel, lista_años):
            df_casosNV = nivel_sel[nivel_sel['Year'].isin(lista_años)]
            return df_casosNV.groupby('Mes')['Valor'].mean()

        # Calcular promedios mensuales para cada caso
        promedio_mensual_humedo = obtener_promedio_mensual(instancia.nivel_process, instancia.años_humedos)
        promedio_mensual_normal = obtener_promedio_mensual(instancia.nivel_process, instancia.años_normales)
        promedio_mensual_seco = obtener_promedio_mensual(instancia.nivel_process, instancia.años_secos)

        # Configuración de la gráfica
        fig, ejes = plt.subplots(3, 1, figsize=(8, 6))
        limites_yNV = (0, 4)
        marcas_yNV = [i * 0.5 for i in range(9)]
        categorias = [
            {"avg": promedio_mensual_humedo, "color": 'b', "title": "Caso Húmedo", "ax": ejes[0]},
            {"avg": promedio_mensual_normal, "color": 'g', "title": "Caso Base", "ax": ejes[1]},
            {"avg": promedio_mensual_seco, "color": 'r', "title": "Caso Seco", "ax": ejes[2]},
        ]

        # Graficar las barras para cada categoría
        for categoria in categorias:
            ax = categoria["ax"]
            barras = ax.bar(categoria["avg"].index, categoria["avg"].values, color=categoria["color"])
            ax.set_title(f'Perfil Hidrológico Anual - {categoria["title"]} NIVEL', fontsize=14)
            ax.set_ylabel('Nivel Promedio (m)')
            ax.set_ylim(limites_yNV)
            ax.set_yticks(marcas_yNV)
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels(['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'])
            for barra in barras:
                yval = barra.get_height()
                ax.text(barra.get_x() + barra.get_width() / 2, yval + 0.1, round(yval, 2), ha='center', va='bottom')

        ejes[2].set_xlabel('Mes')
        plt.tight_layout()

        # Convertir el gráfico a QPixmap y mostrarlo en lblGraph
        nombre_temporal = "grafico_temporal.png"
        fig.savefig(nombre_temporal)

        # Convertir la imagen guardada a QPixmap y mostrarla
        if bandera:
            mostrar_imagen_en_etiqueta(nombre_temporal, instancia_resultados.lblGraph)
        else:
            mostrar_imagen_en_etiqueta(nombre_temporal, instancia.lblGraph)
        if bandera:
            instancia_resultados.limpiar_información()
            instancia_resultados. mostrar_información(f"Mostrado {titulo}\n")
        else:
            instancia.limpiar_información()
            instancia.mostrar_información(f"Mostrado {titulo}\n")

        # Eliminar el archivo temporal
        os.remove(nombre_temporal)
        plt.close(fig)

    except Exception as e:
        if bandera:
            instancia_resultados.registrar_mensaje(f"Error en graficar_perfil_hidrológico_nivel: {e}")
        else:
            instancia.registrar_mensaje(f"Error en graficar_perfil_hidrológico_nivel: {e}")
def graficar_perfil_anual_dias_caudal(instancia=None, instancia_resultados=None,titulo='', décadas=[], bandera=False):
    """Generar gráfico del perfil hidrológico anual por día del año (Caudal) para diferentes categorías (Húmedo, Base, Seco)."""

    try:
        # Asegurar que 'Fecha' esté en formato datetime y agregar columnas para 'Año' y 'Día_del_Año'
        instancia.caudal_process['Fecha'] = pd.to_datetime(instancia.caudal_process['Fecha'], errors='coerce')
        instancia.caudal_process['Year'] = instancia.caudal_process['Fecha'].dt.year
        instancia.caudal_process['Day_of_Year'] = instancia.caudal_process['Fecha'].dt.dayofyear

        # Definir la función para obtener promedios diarios para una lista específica de años
        def obtener_promedio_diario(caudal_sel, lista_años):
            df_casosQ = caudal_sel[caudal_sel['Year'].isin(lista_años)]
            return df_casosQ.groupby('Day_of_Year')['Valor'].mean()

        # Calcular promedios diarios para cada caso
        promedio_diario_humedo = obtener_promedio_diario(instancia.caudal_process, instancia.años_humedos)
        promedio_diario_normal = obtener_promedio_diario(instancia.caudal_process, instancia.años_normales)
        promedio_diario_seco = obtener_promedio_diario(instancia.caudal_process, instancia.años_secos)

        # Configuración de la gráfica
        fig, ejes = plt.subplots(3, 1, figsize=(8, 6))
        limites_yQ = (0, 500)
        marcas_yQ = range(0, 501, 50)
        categorias = [
            {"avg": promedio_diario_humedo, "color": 'b', "title": "Caso Húmedo", "ax": ejes[0]},
            {"avg": promedio_diario_normal, "color": 'g', "title": "Caso Base", "ax": ejes[1]},
            {"avg": promedio_diario_seco, "color": 'r', "title": "Caso Seco", "ax": ejes[2]},
        ]

        # Graficar las barras para cada categoría
        for categoria in categorias:
            ax = categoria["ax"]
            ax.bar(categoria["avg"].index, categoria["avg"].values, color=categoria["color"])
            ax.set_title(f'Perfil Hidrológico Anual CAUDAL - {categoria["title"]}', fontsize=14)
            ax.set_ylabel('Caudal Promedio (m³/s)')
            ax.set_xlabel('Día del Año')
            ax.set_ylim(limites_yQ)
            ax.set_yticks(marcas_yQ)

        plt.tight_layout()

        # Convertir el gráfico a QPixmap y mostrarlo en lblGraph
        nombre_temporal = "grafico_temporal.png"
        fig.savefig(nombre_temporal)

        # Convertir la imagen guardada a QPixmap y mostrarla
        if bandera:
            mostrar_imagen_en_etiqueta(nombre_temporal, instancia_resultados.lblGraph)
        else:
            mostrar_imagen_en_etiqueta(nombre_temporal, instancia.lblGraph)
        if bandera:
            instancia_resultados.limpiar_información()
            instancia_resultados. mostrar_información(f"Mostrado {titulo}\n")
        else:
            instancia.limpiar_información()
            instancia.mostrar_información(f"Mostrado {titulo}\n")

        # Eliminar el archivo temporal
        os.remove(nombre_temporal)
        plt.close(fig)

    except Exception as e:
        if bandera:
            instancia_resultados.registrar_mensaje(f"Error en graficar_perfil_anual_dias_caudal: {e}")
        else:
            instancia.registrar_mensaje(f"Error en graficar_perfil_anual_dias_caudal: {e}")
def graficar_perfil_anual_dias_nivel(instancia=None, instancia_resultados=None,titulo='', décadas=[], bandera=False):
    """Generar gráfico del perfil hidrológico anual por día del año (Nivel) para diferentes categorías (Húmedo, Base, Seco)."""

    try:
        # Asegurar que 'Fecha' esté en formato datetime y agregar columnas para 'Año' y 'Día_del_Año'
        instancia.nivel_process['Fecha'] = pd.to_datetime(instancia.nivel_process['Fecha'], errors='coerce')
        instancia.nivel_process['Year'] = instancia.nivel_process['Fecha'].dt.year
        instancia.nivel_process['Day_of_Year'] = instancia.nivel_process['Fecha'].dt.dayofyear

        # Definir la función para obtener promedios diarios para una lista específica de años
        def obtener_promedio_diario(nivel_sel, lista_años):
            df_casosNV = nivel_sel[nivel_sel['Year'].isin(lista_años)]
            return df_casosNV.groupby('Day_of_Year')['Valor'].mean()

        # Calcular promedios diarios para cada caso
        promedio_diario_humedo = obtener_promedio_diario(instancia.nivel_process, instancia.años_humedos)
        promedio_diario_normal = obtener_promedio_diario(instancia.nivel_process, instancia.años_normales)
        promedio_diario_seco = obtener_promedio_diario(instancia.nivel_process, instancia.años_secos)

        # Configuración de la gráfica
        fig, ejes = plt.subplots(3, 1, figsize=(8, 6))
        limites_yNV = (0, 5)
        marcas_yNV = [i * 0.5 for i in range(11)]
        categorias = [
            {"avg": promedio_diario_humedo, "color": 'b', "title": "Caso Húmedo", "ax": ejes[0]},
            {"avg": promedio_diario_normal, "color": 'g', "title": "Caso Base", "ax": ejes[1]},
            {"avg": promedio_diario_seco, "color": 'r', "title": "Caso Seco", "ax": ejes[2]},
        ]

        # Graficar las barras para cada categoría
        for categoria in categorias:
            ax = categoria["ax"]
            ax.bar(categoria["avg"].index, categoria["avg"].values, color=categoria["color"])
            ax.set_title(f'Perfil Hidrológico Anual NIVEL - {categoria["title"]}', fontsize=14)
            ax.set_ylabel('Nivel Promedio (m)')
            ax.set_xlabel('Día del Año')
            ax.set_ylim(limites_yNV)
            ax.set_yticks(marcas_yNV)

        plt.tight_layout()

        # Convertir el gráfico a QPixmap y mostrarlo en lblGraph
        nombre_temporal = "grafico_temporal.png"
        fig.savefig(nombre_temporal)
        if bandera:
            mostrar_imagen_en_etiqueta(nombre_temporal, instancia_resultados.lblGraph)
        else:
            mostrar_imagen_en_etiqueta(nombre_temporal, instancia.lblGraph)
        if bandera:
            instancia_resultados.limpiar_información()
            instancia_resultados. mostrar_información(f"Mostrado {titulo}\n")
        else:
            instancia.limpiar_información()
            instancia.mostrar_información(f"Mostrado {titulo}\n")

        # Eliminar el archivo temporal
        os.remove(nombre_temporal)
        plt.close(fig)

    except Exception as e:
        if bandera:
            instancia_resultados.registrar_mensaje(f"Error en graficar_perfil_anual_dias_nivel: {e}")
        else:
            instancia.registrar_mensaje(f"Error en graficar_perfil_anual_dias_nivel: {e}")
def mostrar_estadísticas_nominales(instancia=None, instancia_resultados=None,titulo='', décadas=[], bandera=False):
    """Calcular y mostrar estadísticas del caudal y nivel nominal (máximo, promedio, mínimo) para cada categoría hidrológica."""

    try:
        # Asegurar que 'Fecha' esté en formato datetime y agregar la columna 'Año'
        instancia.caudal_process['Fecha'] = pd.to_datetime(instancia.caudal_process['Fecha'], errors='coerce')
        instancia.caudal_process['Year'] = instancia.caudal_process['Fecha'].dt.year
        instancia.nivel_process['Fecha'] = pd.to_datetime(instancia.nivel_process['Fecha'], errors='coerce')
        instancia.nivel_process['Year'] = instancia.nivel_process['Fecha'].dt.year

        # Función para calcular máximo, promedio, mínimo para caudal o nivel
        def calcular_estadísticas(df, lista_años):
            df_filtrado = df[df['Year'].isin(lista_años)]
            max_val = df_filtrado['Valor'].max()
            mean_val = df_filtrado['Valor'].mean()
            min_val = df_filtrado['Valor'].min()
            return max_val, mean_val, min_val

        # Calcular estadísticas de caudal (flow) para cada categoría
        estadisticas_caudal_h2 = calcular_estadísticas(instancia.caudal_process, instancia.años_humedos)
        estadisticas_caudal_b2 = calcular_estadísticas(instancia.caudal_process, instancia.años_normales)
        estadisticas_caudal_s2 = calcular_estadísticas(instancia.caudal_process, instancia.años_secos)

        # Calcular estadísticas de nivel (level) para cada categoría
        estadisticas_nivel_h2 = calcular_estadísticas(instancia.nivel_process, instancia.años_humedos)
        estadisticas_nivel_b2 = calcular_estadísticas(instancia.nivel_process, instancia.años_normales)
        estadisticas_nivel_s2 = calcular_estadísticas(instancia.nivel_process, instancia.años_secos)

        # Formatear las estadísticas para mostrarlas
        texto_estadísticas = (
            f"**Estadísticas de Caudal (Flow)**\n"
            f"Húmedo (Wet) - Máx: {estadisticas_caudal_h2[0]:.2f}, Prom: {estadisticas_caudal_h2[1]:.2f}, Mín: {estadisticas_caudal_h2[2]:.2f}\n"
            f"Base (Normal) - Máx: {estadisticas_caudal_b2[0]:.2f}, Prom: {estadisticas_caudal_b2[1]:.2f}, Mín: {estadisticas_caudal_b2[2]:.2f}\n"
            f"Seco (Dry) - Máx: {estadisticas_caudal_s2[0]:.2f}, Prom: {estadisticas_caudal_s2[1]:.2f}, Mín: {estadisticas_caudal_s2[2]:.2f}\n\n"
            f"**Estadísticas de Nivel (Level)**\n"
            f"Húmedo (Wet) - Máx: {estadisticas_nivel_h2[0]:.2f}, Prom: {estadisticas_nivel_h2[1]:.2f}, Mín: {estadisticas_nivel_h2[2]:.2f}\n"
            f"Base (Normal) - Máx: {estadisticas_nivel_b2[0]:.2f}, Prom: {estadisticas_nivel_b2[1]:.2f}, Mín: {estadisticas_nivel_b2[2]:.2f}\n"
            f"Seco (Dry) - Máx: {estadisticas_nivel_s2[0]:.2f}, Prom: {estadisticas_nivel_s2[1]:.2f}, Mín: {estadisticas_nivel_s2[2]:.2f}"
        )
        if bandera:
            # instancia_resultados.lblGraph.clear()
            instancia.limpiar_información()
            instancia.mostrar_información(texto_estadísticas)
        else:
            # instancia.lblGraph.clear()
            instancia.limpiar_información()
            instancia.mostrar_información(texto_estadísticas)

    except Exception as e:
        if bandera:
            instancia_resultados.registrar_mensaje(f"Error en mostrar_estadísticas_nominales: {e}")
        else:
            instancia.registrar_mensaje(f"Error en mostrar_estadísticas_nominales: {e}")





def calcular_P95_y_mostrar(instancia=None, instancia_resultados=None,titulo='', décadas=[], bandera=False, escala=(800, 600)):
    """Calcular y mostrar la probabilidad de excedencia P95 y su gráfico correspondiente."""

    try:
        # Lista de años del DataFrame 'caudal_sel'
        lista_anosQ = instancia.caudal_process['Fecha'].dt.year.tolist()

        # Inicializar listas y diccionarios
        caudales_P95 = []  # Lista para los valores P95
        Caudales = instancia.caudal_process['Valor'].tolist()
        Fechas_compQ = instancia.caudal_process['Fecha'].tolist()

        # Convertir fechas a objetos datetime si no lo están ya
        Fechas_compQ = [pd.to_datetime(fecha) for fecha in Fechas_compQ]

        # Diccionario para agrupar los caudales por año
        caudales_por_ano = {ano: [] for ano in lista_anosQ}

        # Agrupar los caudales por año
        for fecha, caudal in zip(Fechas_compQ, Caudales):
            ano = fecha.year
            if ano in caudales_por_ano:
                caudales_por_ano[ano].append(caudal)

        # Calcular el P95 para cada año
        for ano in lista_anosQ:
            if caudales_por_ano[ano]:  # Si hay datos para el año
                # Ordenar los caudales para el año actual
                caudales_ordenados = sorted(caudales_por_ano[ano])

                # Calcular el percentil 95 (P95)
                P95 = np.percentile(caudales_ordenados, 95)
                caudales_P95.append(P95)
            else:
                caudales_P95.append(None)  # No hay datos para el año

        # Crear un DataFrame para los resultados
        P95_dfQ = pd.DataFrame({
            'Año': lista_anosQ,
            'Caudal_P95': caudales_P95})

        # Imprimir el DataFrame para verificar
        print(P95_dfQ.head())

        # Graficar los resultados
        fig = plt.figure(figsize=(8, 6))
        plt.plot(P95_dfQ['Año'], P95_dfQ['Caudal_P95'], label='Caudal (m³/s)', color='blueviolet')
        plt.xlabel('Año')
        plt.ylabel('Caudal (m³/s)')
        plt.title('Caudal (Q95) vs Décadas')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Guardar el gráfico temporalmente como una imagen
        nombre_temporal = "grafico_temporal.png"
        fig.savefig(nombre_temporal)
        plt.close()  # Cerrar la figura para liberar recursos
        if bandera:
            mostrar_imagen_en_etiqueta(nombre_temporal, instancia_resultados.lblGraph)
        else:
            mostrar_imagen_en_etiqueta(nombre_temporal, instancia.lblGraph)
        # Mostrar la imagen en el label adecuado utilizando la función mostrar_imagen_en_etiqueta
        if bandera:
            instancia_resultados.limpiar_información()
            instancia_resultados. mostrar_información(f"Max P95: {np.max(caudales_P95):.2f}\nMin P95: {np.min(caudales_P95):.2f}\nPromedio P95: {np.mean(caudales_P95):.2f}")
        else:
            instancia.limpiar_información()
            instancia.mostrar_información(f"Max P95: {np.max(caudales_P95):.2f}\nMin P95: {np.min(caudales_P95):.2f}\nPromedio P95: {np.mean(caudales_P95):.2f}")
        os.remove(nombre_temporal)

    except Exception as e:
        if bandera:
            instancia_resultados.registrar_mensaje(f"Error en calcular_P95_y_mostrar: {e}")
        else:
            instancia.registrar_mensaje(f"Error en calcular_P95_y_mostrar: {e}")
        print(f"Error en calcular_P95_y_mostrar: {e}")  # Imprimir el error en la consola para debug
def mostrar_caudal_promedio(instancia=None, instancia_resultados=None,titulo='', décadas=[], bandera=False):
    """Calcular y mostrar el caudal promedio (caudal promedio) a lo largo de los años."""
    
    try:
        # **CAUDALES PROMEDIOS**
        # Convertir 'Fecha' a datetime y establecerla como índice
        caudal_sel = instancia.caudal_process  # Suponiendo que instancia.caudal_process es su conjunto de datos
        caudal_sel['Fecha'] = pd.to_datetime(caudal_sel['Fecha'])
        caudal_sel = caudal_sel.set_index('Fecha')

        # Re-muestrear con frecuencia anual y calcular el promedio de caudal
        promedios_anuales = caudal_sel['Valor'].resample('A').mean()

        # Convertir años y promedios anuales en listas
        lista_anosQ = promedios_anuales.index.year.tolist()  # Lista de años
        lista_promedios_anualesQ = promedios_anuales.tolist()  # Lista de promedios anuales

        # Imprimir para depuración
        print("Años:", lista_anosQ)
        print("Promedios anuales:", lista_promedios_anualesQ)

        # Crear el gráfico de caudal promedio
        fig = plt.figure(figsize=(8, 6))
        plt.plot(lista_anosQ, lista_promedios_anualesQ, label='Caudal m³/s', color='blueviolet')
        plt.xlabel('Año')
        plt.ylabel('Caudal (m³/s)')
        plt.title('Caudal Promedio vs Años')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Guardar el gráfico temporalmente como una imagen
        nombre_temporal = "grafico_temporal.png"
        fig.savefig(nombre_temporal)
        plt.close()  # Cerrar la figura para liberar recursos

        if bandera:
            mostrar_imagen_en_etiqueta(nombre_temporal, instancia_resultados.lblGraph)
        else:
            mostrar_imagen_en_etiqueta(nombre_temporal, instancia.lblGraph)

        # Mostrar las estadísticas de caudal
        max_caudal = np.max(lista_promedios_anualesQ)
        min_caudal = np.min(lista_promedios_anualesQ)
        mean_caudal = np.mean(lista_promedios_anualesQ)

        if bandera:
            instancia_resultados.limpiar_información()
            instancia_resultados. mostrar_información(f"Max Caudal: {max_caudal:.2f} m³/s\nMin Caudal: {min_caudal:.2f} m³/s\nCaudal Promedio: {mean_caudal:.2f} m³/s")
        else:
            instancia.limpiar_información()
            instancia.mostrar_información(f"Max Caudal: {max_caudal:.2f} m³/s\nMin Caudal: {min_caudal:.2f} m³/s\nCaudal Promedio: {mean_caudal:.2f} m³/s")

        os.remove(nombre_temporal)

    except Exception as e:
        if bandera:
            instancia_resultados.registrar_mensaje(f"Error en mostrar_caudal_promedio: {e}")
        else:
            instancia.registrar_mensaje(f"Error en mostrar_caudal_promedio: {e}")
def mostrar_nivel_P95(instancia=None, instancia_resultados=None,titulo='', décadas=[], bandera=False):
    """Calcular y mostrar los niveles P95 para cada año."""
    
    try:
        # **Nivel P95**
        # Lista de años de las diferentes categorías (H2, B2, S2)
        # Asegúrese de que tanto los archivos como los datos procesados estén disponibles
        if not instancia.años_secos or not instancia.años_humedos or not instancia.años_normales:
            messagebox.showerror("Años Faltantes", "Por favor seleccione los años Secos, Húmedos y Normales antes.")
            return
        total_years = instancia.años_secos | instancia.años_humedos | instancia.años_normales  # Usando el operador de unión
        total_years = sorted(total_years)
        print("Total de Años:", len(total_years))

        # Extraer años de la columna 'Fecha'
        anos_listaNV = instancia.nivel_process['Fecha'].dt.year.tolist()

        # Inicializar listas para los valores P95
        nivel_P95 = []
        niveles = instancia.nivel_process['Valor'].tolist()  # Lista de valores de nivel
        Fechas_compNV = instancia.nivel_process['Fecha'].tolist()

        # Convertir las fechas a datetime si no lo están ya
        Fechas_compNV = [pd.to_datetime(fecha) for fecha in Fechas_compNV]

        # Agrupar los niveles por año
        niveles_por_ano = {ano: [] for ano in anos_listaNV}
        for fecha, lvl in zip(Fechas_compNV, niveles):
            ano = fecha.year
            if ano in niveles_por_ano:
                niveles_por_ano[ano].append(lvl)

        # Calcular P95 para cada año
        for ano in anos_listaNV:
            if niveles_por_ano[ano]:  # Si hay datos para el año
                niveles_ordenados = sorted(niveles_por_ano[ano])
                P95 = np.percentile(niveles_ordenados, 95)
                nivel_P95.append(P95)
            else:
                nivel_P95.append(None)  # No hay datos para el año

        # Crear el DataFrame para los niveles P95
        NivelP95_df = pd.DataFrame({
            'Año': anos_listaNV,
            'Nivel_P95': nivel_P95
        })

        # Filtrar el DataFrame utilizando la lista de años totales
        NivelP95_df['Año'] = pd.to_datetime(NivelP95_df['Año'], errors='coerce')
        NivelP95_df_fil = NivelP95_df[NivelP95_df['Año'].dt.year.isin(total_years)]

        # Mostrar el DataFrame para depuración
        print(NivelP95_df.head())

        # Graficar los resultados de nivel P95
        fig= plt.figure(figsize=(8, 6))
        plt.plot(NivelP95_df['Año'], NivelP95_df['Nivel_P95'], label='Nivel cm', color='blueviolet')
        plt.xlabel('Año')
        plt.ylabel('Nivel (cm)')
        plt.title('Nivel P95 vs Décadas')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        # Guardar el gráfico temporalmente como una imagen
        nombre_temporal = "grafico_temporal.png"
        fig.savefig(nombre_temporal)
        plt.close()  # Cerrar la figura para liberar recursos

        if bandera:
            mostrar_imagen_en_etiqueta(nombre_temporal, instancia_resultados.lblGraph)
        else:
            mostrar_imagen_en_etiqueta(nombre_temporal, instancia.lblGraph)

        # Aquí sigue el patrón exacto solicitado para manejar la imagen y la información:
        if bandera:
            instancia_resultados.limpiar_información()
            instancia_resultados. mostrar_información("Mostrados los gráficos y estadísticas de Nivel P95.\n")
        else:
            instancia.limpiar_información()
            instancia.mostrar_información("Mostrados los gráficos y estadísticas de Nivel P95.\n")

    except Exception as e:
        if bandera:
            instancia_resultados.registrar_mensaje(f"Error en mostrar_nivel_P95: {e}")
        else:
            instancia.registrar_mensaje(f"Error en mostrar_nivel_P95: {e}")
def mostrar_velocidad_flujo(instancia=None, instancia_resultados=None,titulo='', décadas=[], bandera=False):
    """Calcular y mostrar la velocidad de flujo (Velocidad) gráfica."""

    try:
        # **Velocidad de Flujo**
        nivel_cha_cha = instancia.df_merge['Nivel'].tolist()
        caudal_cha_cha = instancia.df_merge['Caudal'].tolist()
        fecha_cha_cha = instancia.df_merge['Fecha'].tolist()

        print('¿Aja?')
        print(len(fecha_cha_cha))
        print(len(nivel_cha_cha))
        print(len(caudal_cha_cha))
        print('¿O no aja?')

        lvl = 0
        longitud = 142.73  # metros (por ejemplo)
        Lvlm = [lvl / 100 for lvl in nivel_cha_cha]

        # Calcular la variable área según el nivel
        Area_variable = [elemento * longitud if elemento >= 0.5 else 0 for elemento in Lvlm]

        # Calcular la velocidad
        instancia.velocidad = []
        ultimo_valor = None  # Variable para almacenar el último valor válido de velocidad

        for a, b in zip(caudal_cha_cha, Area_variable):
            try:
                if pd.isna(b):
                    # Usar el último valor válido de velocidad si b es NaN
                    instancia.velocidad.append(ultimo_valor if ultimo_valor is not None else 0)
                else:
                    # Calcular la velocidad
                    valor_actual = a / b
                    instancia.velocidad.append(valor_actual)
                    ultimo_valor = valor_actual  # Actualizar el último valor válido
            except ZeroDivisionError:
                # Asignar cero en caso de una división por cero
                instancia.velocidad.append(0)

        # Agregar la velocidad calculada como una nueva columna en el DataFrame
        instancia.df_merge['Velocidad'] = instancia.velocidad

        # Graficar la velocidad
        fig = plt.figure(figsize=(8, 6))
        plt.plot(fecha_cha_cha, instancia.velocidad, label='Velocidad m/s', color='blueviolet')
        plt.xlabel('Año')
        plt.ylabel('Velocidad (m/s)')
        plt.title('Velocidad vs Décadas')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Guardar el gráfico temporalmente como una imagen
        nombre_temporal = "grafico_temporal.png"
        fig.savefig(nombre_temporal)
        plt.close()  # Cerrar la figura para liberar recursos
        # Mostrar la imagen de acuerdo al valor de bandera
        if bandera:
            mostrar_imagen_en_etiqueta(nombre_temporal, instancia_resultados.lblGraph)
        else:
            mostrar_imagen_en_etiqueta(nombre_temporal, instancia.lblGraph)
        os.remove(nombre_temporal)

        # Mostrar la información de estadísticas de velocidad
        max_velocidad = np.nanmax(instancia.velocidad) if instancia.velocidad else None
        min_velocidad = np.nanmin(instancia.velocidad) if instancia.velocidad else None
        mean_velocidad = np.nanmean(instancia.velocidad) if instancia.velocidad else None

        texto_informacion_velocidad = f"Max Velocidad: {max_velocidad:.2f} m/s\nMin Velocidad: {min_velocidad:.2f} m/s\nPromedio Velocidad: {mean_velocidad:.2f} m/s"

        if bandera:
            instancia_resultados.limpiar_información()
            instancia_resultados. mostrar_información(texto_informacion_velocidad)
        else:
            instancia.limpiar_información()
            instancia.mostrar_información(texto_informacion_velocidad)

        

    except Exception as e:
        if bandera:
            instancia_resultados.registrar_mensaje(f"Error en mostrar_velocidad_flujo: {e}")
        else:
            instancia.registrar_mensaje(f"Error en mostrar_velocidad_flujo: {e}")
def mostrar_comportamiento_mensual(instancia=None, instancia_resultados=None,titulo='', décadas=[], bandera=False):
    """Calcular y mostrar el comportamiento mensual de la velocidad promedio diaria por mes y categoría."""

    try:
        # Asegurarse de que la columna 'Fecha' esté en formato datetime
        instancia.df_merge['Fecha'] = pd.to_datetime(instancia.df_merge['Fecha'])

        # Preparar los datos para 'Año', 'Mes' y 'Día del Año'
        instancia.df_merge['Año'] = instancia.df_merge['Fecha'].dt.year
        instancia.df_merge['Día_del_Año'] = instancia.df_merge['Fecha'].dt.dayofyear
        instancia.df_merge['Mes'] = instancia.df_merge['Fecha'].dt.month

        # Usar las listas de años para las categorías
        años_secos = instancia.años_secos  # Años secos
        años_humedos = instancia.años_humedos  # Años húmedos
        años_normales = instancia.años_normales  # Años normales

        # Función para calcular la velocidad promedio diaria por día del año
        def obtener_promedio_diario(df, lista_años):
            df_filtrado = df[df['Año'].isin(lista_años)]
            return df_filtrado.groupby('Día_del_Año')['Velocidad'].mean()

        # Obtener los promedios diarios para cada categoría
        df_merge_secos = obtener_promedio_diario(instancia.df_merge, años_secos)  # Años secos
        df_merge_humedos = obtener_promedio_diario(instancia.df_merge, años_humedos)  # Años húmedos
        df_merge_normales = obtener_promedio_diario(instancia.df_merge, años_normales)  # Años normales

        # Graficar los promedios de velocidad diaria
        mostrar_velocidad_mensual(instancia, instancia_resultados,df_merge_secos, df_merge_humedos, df_merge_normales, 'Día del Año', bandera)

    except Exception as e:
        if bandera:
            instancia_resultados.registrar_mensaje(f"Error en mostrar_comportamiento_mensual: {e}")
        else:
            instancia.registrar_mensaje(f"Error en mostrar_comportamiento_mensual: {e}")
def mostrar_velocidad_promedio_mensual(instancia=None, instancia_resultados=None,titulo='', décadas=[], bandera=False):
    """Calcular y mostrar la velocidad promedio diaria por mes y categoría."""

    try:
        # Asegurarse de que la columna 'Fecha' esté en formato datetime
        instancia.df_merge['Fecha'] = pd.to_datetime(instancia.df_merge['Fecha'])

        # Preparar los datos para 'Año', 'Mes' y 'Día del Año'
        instancia.df_merge['Año'] = instancia.df_merge['Fecha'].dt.year
        instancia.df_merge['Día_del_Año'] = instancia.df_merge['Fecha'].dt.dayofyear
        instancia.df_merge['Mes'] = instancia.df_merge['Fecha'].dt.month

        # Usar las listas de años para las categorías
        años_secos = instancia.años_secos  # Años secos
        años_humedos = instancia.años_humedos  # Años húmedos
        años_normales = instancia.años_normales  # Años normales

        # Función para calcular la velocidad promedio diaria por día del año
        def obtener_promedio_diario(df, lista_años):
            df_filtrado = df[df['Año'].isin(lista_años)]
            return df_filtrado.groupby('Día_del_Año')['Velocidad'].mean()

        # Obtener los promedios diarios para cada categoría
        df_merge_secos = obtener_promedio_diario(instancia.df_merge, años_secos)  # Años secos
        df_merge_humedos = obtener_promedio_diario(instancia.df_merge, años_humedos)  # Años húmedos
        df_merge_normales = obtener_promedio_diario(instancia.df_merge, años_normales)  # Años normales

        # Graficar los promedios de velocidad diaria
        mostrar_velocidad_mensual(instancia,instancia_resultados,df_merge_secos, df_merge_humedos, df_merge_normales, 'Mes', bandera)

    except Exception as e:
        if bandera:
            instancia_resultados.registrar_mensaje(f"Error en mostrar_velocidad_promedio_mensual: {e}")
        else:
            instancia.registrar_mensaje(f"Error en mostrar_velocidad_promedio_mensual: {e}")
def mostrar_velocidad_mensual(instancia, instancia_resultados, df_merge_secos, df_merge_humedos, df_merge_normales, caso, bandera):
    """Graficar la velocidad promedio diaria por mes para cada categoría."""

    try:
        # Crear la figura para los gráficos de barras
        fig=plt.figure(figsize=(8, 6))

        # Graficar para los años secos (instancia.dry)
        plt.subplot(3, 1, 1)
        barras_secos = plt.bar(df_merge_secos.index, df_merge_secos.values, color='b')
        plt.title('Perfil Hidrológico Anual Velocidad - Caso Seco', fontsize=12)
        plt.ylabel('Velocidad Promedio (m/s)')
        if caso == 'Mes':
            plt.xticks(range(1, 13), ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'])
        plt.xlabel(caso)

        # Agregar etiquetas de datos a las barras
        if caso == 'Mes':
            for barra in barras_secos:
                valor_y = barra.get_height()
                plt.text(barra.get_x() + barra.get_width()/2, valor_y + 0.01, f'{valor_y:.3f}', 
                        ha='center', va='bottom', fontsize=8)

        # Graficar para los años húmedos (instancia.wet)
        plt.subplot(3, 1, 2)
        barras_humedos = plt.bar(df_merge_humedos.index, df_merge_humedos.values, color='g')
        plt.title('Perfil Hidrológico Anual Velocidad - Caso Húmedo', fontsize=12)
        plt.ylabel('Velocidad Promedio (m/s)')
        if caso == 'Mes':
            plt.xticks(range(1, 13), ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'])
        plt.xlabel(caso)

        # Agregar etiquetas de datos a las barras
        if caso == 'Mes':
            for barra in barras_humedos:
                valor_y = barra.get_height()
                plt.text(barra.get_x() + barra.get_width()/2, valor_y + 0.01, f'{valor_y:.3f}', 
                        ha='center', va='bottom', fontsize=8)

        # Graficar para los años normales (instancia.normal)
        plt.subplot(3, 1, 3)
        barras_normales = plt.bar(df_merge_normales.index, df_merge_normales.values, color='r')
        plt.title('Perfil Hidrológico Anual Velocidad - Caso Normal', fontsize=12)
        plt.xlabel(caso)
        if caso == 'Mes':
            plt.xticks(range(1, 13), ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'])
        plt.ylabel('Velocidad Promedio (m/s)')

        # Agregar etiquetas de datos a las barras
        if caso == 'Mes':
            for barra in barras_normales:
                valor_y = barra.get_height()
                plt.text(barra.get_x() + barra.get_width()/2, valor_y + 0.01, f'{valor_y:.3f}', 
                        ha='center', va='bottom', fontsize=8)

        # Ajustar el espacio entre los gráficos
        plt.tight_layout(pad=2)

        ## Guardar el gráfico temporalmente como una imagen
        nombre_temporal = "grafico_temporal.png"
        fig.savefig(nombre_temporal)
        plt.close()  # Cerrar la figura para liberar recursos

        # Mostrar la imagen de acuerdo al valor de bandera
        if bandera:
            mostrar_imagen_en_etiqueta(nombre_temporal, instancia_resultados.lblGraph)
        else:
            mostrar_imagen_en_etiqueta(nombre_temporal, instancia.lblGraph)

        os.remove(nombre_temporal)

        # Mostrar la información de estadísticas de velocidad
        if bandera:
            instancia_resultados.limpiar_información()
            instancia_resultados. mostrar_información("Displayed daily average velocity graph for categories.\n")
        else:
            instancia.limpiar_información()
            instancia.mostrar_información("Displayed daily average velocity graph for categories.\n")

    except Exception as e:
        if bandera:
            instancia_resultados.registrar_mensaje(f"Error en mostrar_velocidad_mensual: {e}")
        else:
            instancia.registrar_mensaje(f"Error en mostrar_velocidad_mensual: {e}")
def calculate_and_display_turbine_power(instancia=None, instancia_resultados=None, turbine_options=None, titulo='', bandera=False,index=0):
    """Calculate and display power output for the specified turbine models."""
    try:
        # Default turbine options if none are provided
        if turbine_options is None:
            turbine_options = [
                "SmartFreestream",
                "SmartMonofloat",
                "EnviroGen005series",
                "Hydroquest1.4",
                "EVG-050H",
                "EVG-025H"
            ]
        else:
            turbine_options = [turbine_options]  # Ensure it's a list

        # Load turbine data
        file_name_datasheet = r'Datasheet_V2.csv'
        file_name_powercurve = r'PowerCurves.xlsx'
        df_datasheet = pd.read_csv(file_name_datasheet, delimiter=',')
        df_powercurve = pd.read_excel(file_name_powercurve)

        # Clean up the column names to remove leading/trailing spaces
        df_datasheet.columns = df_datasheet.columns.str.strip()
        # Initialize turbines
        turbines = {}
        for turbine_name in turbine_options:
            # Check if the turbine name exists in the columns
            if turbine_name not in df_datasheet.columns:
                error_message = f"Turbine '{turbine_name}' not found in data sheet columns."
                handle_error(instancia,instancia_resultados,error_message, bandera)
                return

            # Extract the relevant turbine data from the datasheet
            data = df_datasheet[turbine_name].tolist()
            filteredpc = df_powercurve[df_powercurve['Type'] == turbine_name]

            turbines[turbine_name] = Turbina(
                turbine_name, *data[:7], filteredpc
            )

        # Extract velocity data
        velocity = instancia.df_merge['Velocidad'].tolist()
        list_of_turb_power = []

        # Pre-compute power outputs for each turbine
        for turbine_name in turbine_options:
            turbine = turbines[turbine_name]
            power_output = turbine.PowerOut(velocity)
            list_of_turb_power.append(power_output)

        # If turbine_options is provided, display all turbine graphs immediately
        if turbine_options is not None:
            for t, turbine_name in enumerate(turbine_options):
                update_turbine_plot(instancia,instancia_resultados,t, list_of_turb_power, turbine_options, bandera,index)

        # If turbine_options is None, display turbine graphs with a delay of 5 seconds between each
        else:
            # If turbine_options is provided, display all turbine graphs immediately
            if turbine_options is not None:
                # Display turbine plots with a delay
                for t in range(len(turbine_options)):
                    instancia.parent.after(
                        t * 5000,  # Delay each turbine plot by 5 seconds
                        partial(update_turbine_plot, instancia,index, t, list_of_turb_power, turbine_options, bandera,index)
                    )
                if bandera:
                    instancia_resultados.registrar_mensaje(f"Displaying all turbine power plots with delay.")
                else:
                    instancia.registrar_mensaje(f"Displaying all turbine power plots with delay.")


    except ValueError as ve:
        error_message = f"Value Error: {ve}"
        handle_error(instancia,instancia_resultados,error_message, bandera)
    except KeyError as ke:
        error_message = f"Key Error: {ke}"
        handle_error(instancia,instancia_resultados,error_message, bandera)
    except Exception as e:
        error_message = f"Unexpected Error: {e}"
        handle_error(instancia,instancia_resultados,error_message, bandera)


def update_turbine_plot(instancia=None, instancia_resultados=None, index=None, list_of_turb_power=None, turbine_options=None, bandera=False,idx=None):
    """Update and display plot for a specific turbine."""
    if index < len(turbine_options):
        turbine_name = turbine_options[index]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(instancia.df_merge['Fecha'], list_of_turb_power[index], label=turbine_name,
                color=['blue', 'green', 'red', 'lime', 'darkorange', 'aquamarine'][idx])
        ax.set_xlabel('Año')
        ax.set_ylabel('Potencia (kW)')
        ax.set_title(f'Modelo {turbine_name}')
        ax.legend()
        ax.grid(True)
        # Guardar el gráfico temporalmente como una imagen
        nombre_temporal = "grafico_temporal.png"
        fig.savefig(nombre_temporal)
        plt.close()  # Cerrar la figura para liberar recursos
        # Mostrar la imagen de acuerdo al valor de bandera
        if bandera:
            mostrar_imagen_en_etiqueta(nombre_temporal, instancia_resultados.lblGraph)
        else:
            mostrar_imagen_en_etiqueta(nombre_temporal, instancia.lblGraph)
        os.remove(nombre_temporal)
        

        if bandera:
            instancia_resultados.mostrar_información(f"Displayed {turbine_name} plot.\n")
        else:
            instancia.mostrar_información(f"Displayed {turbine_name} plot.\n")


def handle_error(instancia,instancia_resultados, message, bandera):
    """Handle errors by logging or showing in a specific component."""
    print(f"Error: {message}")
    if bandera:
        instancia_resultados. mostrar_información(f"{message} \n Try to Process the data first please on other page")
    else:
        instancia.mostrar_información(message)