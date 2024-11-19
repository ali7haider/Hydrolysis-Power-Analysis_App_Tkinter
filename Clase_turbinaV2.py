import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

'''
Clase turbina

Name, nombre de la tecnología, que coincide con el nombre de las columnas en la ficha tecnica
Pout, potencia nominal de salida
Lenght, Longitud
Width, ancho
Height, altura
rtor, radio del rotor
Depth, profundidad minima del rio para la instalación del dispositivo
Type, tipo de turbina
PowerCurve, Curva de potencia vs velocidad de la turbina

MÉTODOS
Power Out, calculo de la potencia de salida de la turbina considerando la velocidad promedio de flujo
get_monthly_avg, promedios mensuales del numero de turbinas necesarias para suplir la demanda especificacda por el usuario

'''


class Turbina:
    def __init__(self, Name, Pout, Lenght, Width, Height, rtor, Depth, Type, PowerCurve):
        self.Name = Name
        self.Pout = Pout
        self.Lenght = Lenght
        self.Width = Width
        self.Height = Height
        self.rtor = rtor
        self.Depth = Depth
        self.Type = Type
        self.PowerCurve = PowerCurve

    #Salida de potencia de las trubinas
    def PowerOut(self,velocity):
        df_filtered = self.PowerCurve
        x_values = df_filtered['Velocity (m/s)'].values
        y_values = df_filtered['Power kW'].values

        # Crear la función de interpolación lineal
        interpolator = interp1d(x_values, y_values, kind='linear', fill_value="extrapolate")

        # Calcular los valores de y interpolados para cada velocidad
        y_interpolated = interpolator(velocity)

        # Modificar los valores interpolados si son negativos
        y_interpolated[y_interpolated < 0] = np.nan

        # Mostrar los resultados
        return y_interpolated
    

def get_power(velocity, df_merged):

    #Rutas de los archivos de ficha tecnca y curvas de potencia

    file_name_datasheet = r'Datasheet_V2.csv'
    file_name_powercurve = r'PowerCurves.xlsx'

    df_datasheet = pd.read_csv(file_name_datasheet, delimiter=',')
    df_powercurve = pd.read_excel(file_name_powercurve)



    ''' Información que ingresa el usuario a través de un drop list o check box'''

    #Asumiendo que selecciona todas las opciones disponibles
    Turbine_options = ['SmartFreestream',	'SmartMonofloat',	'EnviroGen005series', 'Hydroquest1.4', 'EVG-050H', 'EVG-025H']

    #Potencia requerida por el usuario
    Potencia_requerida = 50 # En kW


    ''' #Creacion del o los objetos#
    Al inicio de codigo se crean todos los objetos correspondientes a todas las turbinas disponibles

    '''

    for i in Turbine_options:
        #llamar el datasheet de cada turbina
        data = df_datasheet[i].tolist()
        if i == 'SmartFreestream':
            #filtar la información de curva de potencia
            filteredpc = df_powercurve[df_powercurve['Type'] == i]
            #Asignar la información de la ficha tecnica a los objetos
            SmartFreestream = Turbina(i,data[0],data[1],data[2], data[3],data[4],data[5],data[6],filteredpc)
        elif i == 'SmartMonofloat':
            filteredpc = df_powercurve[df_powercurve['Type'] == i]
            SmartMonofloat = Turbina(i,data[0],data[1],data[2], data[3],data[4],data[5],data[6],filteredpc)
        elif i == 'EnviroGen005series':
            filteredpc = df_powercurve[df_powercurve['Type'] == i]
            EnviroGen005series = Turbina(i,data[0],data[1],data[2], data[3],data[4],data[5],data[6],filteredpc)
        elif i == 'Hydroquest1.4':
            filteredpc = df_powercurve[df_powercurve['Type'] == i]
            Hydroquest1_4 = Turbina(i,data[0],data[1],data[2], data[3],data[4],data[5],data[6],filteredpc)
        elif i == 'EVG-050H':
            filteredpc = df_powercurve[df_powercurve['Type'] == i]
            EVG_050H = Turbina(i,data[0],data[1],data[2], data[3],data[4],data[5],data[6],filteredpc)
        elif i == 'EVG-025H':
            filteredpc = df_powercurve[df_powercurve['Type'] == i]
            EVG_025H = Turbina(i,data[0],data[1],data[2], data[3],data[4],data[5],data[6],filteredpc)


    #Seleccion que realiza el usuario de las trubinas que desea evaluar


    #Turbine_selection = ['SmartFreestream',	'SmartMonofloat', 'EnviroGen005series',	'Hydroquest1.4']
    Turbine_selection = ['SmartFreestream',	'SmartMonofloat',	'EnviroGen005series',	'Hydroquest1.4', 'EVG-050H',	'EVG-025H']


    '''Empiezan los calculos'''

    ''''Información de estrada que es este caso solo se usa para probar el codigo'''
    # Generar una lista de fechas 
    fecha = df_merged['Fecha'].tolist()

    # Generar una lista de vvelocidades
    velocity = df_merged['Velocidad'].tolist()

    '''end'''

    y1_result, y2_result, y3_result, y4_result, y5_result, y6_result = 0,0,0,0,0,0
    y1, y2, y3, y4, y5, y6 = 0,0,0,0,0,0

    df_resultados_NTurbinas = pd.DataFrame({
        "Velocidad m/s": velocity,
        "Fecha": fecha
    })


    #Dataframe de potencias
    df_potencias = df_resultados_NTurbinas

    color = ['blue','green', 'red','lime', 'darkorange','aquamarine']
    # Potencia de salida

    plt.figure(figsize=(12, 6))

    list_of_turb_power = []
    N_turb = [y1, y2, y3, y4, y5, y6]


    

    t = 0
    for i in Turbine_selection:
        if i == 'SmartFreestream':
            y1 = SmartFreestream.PowerOut(velocity)
            y1_result = [Potencia_requerida / x if x != 0 else np.nan for x in y1]
            list_of_turb_power.append(y1)
            df_resultados_NTurbinas["SmartFreestream"] = y1_result
            plt.plot(fecha, y1, label=i, color='blue', marker='o')  # Grafica la primera lista  
        elif i == 'SmartMonofloat':
            y2 = SmartMonofloat.PowerOut(velocity)
            y2_result = [Potencia_requerida / x if x != 0 else np.nan for x in y2]
            list_of_turb_power.append(y2)
            df_resultados_NTurbinas["SmartMonofloat"] = y2_result
            plt.plot(fecha, y2, label=i, color='green', marker='x')  # Grafica la segunda lista
        elif i == 'EnviroGen005series':
            y3 = EnviroGen005series.PowerOut(velocity)
            y3_result = [Potencia_requerida / x if x != 0 else np.nan for x in y3]
            list_of_turb_power.append(y3)
            df_resultados_NTurbinas["EnviroGen005series"] = y3_result
            plt.plot(fecha, y3, label=i, color='red', marker='s')   # Grafica la tercera lista
        elif i == 'Hydroquest1.4':
            y4 = Hydroquest1_4.PowerOut(velocity)
            y4_result = [Potencia_requerida / x if x != 0 else np.nan for x in y4]
            list_of_turb_power.append(y4)
            df_resultados_NTurbinas["Hydroquest1.4"] = y4_result
            plt.plot(fecha, y4, label=i, color='lime', marker='v')   # Grafica la cuarta lista
        elif i == 'EVG-050H':
            y5 = EVG_050H.PowerOut(velocity)
            y5_result = [Potencia_requerida / x if x != 0 else np.nan for x in y5]
            list_of_turb_power.append(y5)
            df_resultados_NTurbinas['EVG-050H'] = y5_result
            plt.plot(fecha, y5, label=i, color='darkorange', marker='1')   # Grafica la cuarta list
        elif i == 'EVG-025H':
            y6 = EVG_025H.PowerOut(velocity)
            y6_result = [Potencia_requerida / x if x != 0 else np.nan for x in y6]
            list_of_turb_power.append(y6)
            df_resultados_NTurbinas['EVG-025H'] = y6_result
            plt.plot(fecha, y6, label=i, color='aquamarine', marker='p') 
        t = t + 1   # Grafica la cuarta lista

    # Personalización de la gráfica

    plt.xlabel('Fecha')
    plt.ylabel('Power kW')
    plt.legend()
    plt.title('Potencia de salida turbinas')

    # Mostrar la gráfica
    plt.show()


    '''Graficar cada turbina particularmente Potencia de salida'''
    ###

    t = 0
    for i in Turbine_selection:
        plt.figure(figsize=(8, 6))
        plt.plot(fecha, list_of_turb_power[t], label=i, color = color[t])
        plt.xlabel('Año')
        plt.ylabel('Potencia (kW)')
        plt.title('Modelo ' + i)
        plt.legend()
        plt.grid(True)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        t = t + 1
        plt.show()
        

    '''
    #Numero de turbinas requeridas para suplir la potencia especificada por el usuario#
    ***Se requiere contar con los años del periodo hmedo seco y normal
    '''
    # Definir las listas de años para cada categoría
    years_H2 = [1970, 1971, 1988, 1989, 1998, 1999]  # Años de la categoría 'Húmedo'
    years_B2 = [1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981 , 1984, 1986, 1990, 1993, 1994, 1995, 1996, 2000]  # Años de la categoría 'Base'
    years_S2 = [1982, 1983, 1992, 1997, 1998]  # Años de la categoría 'Seco'

    # Agregar una columna de 'Mes' para análisis mensual
    df_resultados_NTurbinas['Mes'] = df_resultados_NTurbinas['Fecha'].dt.month
    df_resultados_NTurbinas['Year'] = df_resultados_NTurbinas['Fecha'].dt.year


    # Función para filtrar y calcular la media mensual de caudales
    def get_monthly_avg(df_filter, year_list):
        df_casos = df_filter[df_filter['Year'].isin(year_list)]
        return df_casos.groupby('Mes')[Turbine_selection].mean()


    # Obtener las medias mensuales para cada categoría
    monthly_avg_a = get_monthly_avg(df_resultados_NTurbinas, years_H2)
    monthly_avg_b = get_monthly_avg(df_resultados_NTurbinas, years_B2)
    monthly_avg_c = get_monthly_avg(df_resultados_NTurbinas, years_S2)

    scenarios = [monthly_avg_a,monthly_avg_b,monthly_avg_c]
    scenarios_names = ['Húmedo', 'Base', 'Seco']

    t = 0
    for i in scenarios:
        i.plot(kind='bar', figsize=(10, 6))
        plt.xticks(range(1, 13), ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'])
        plt.xlabel("Mes")
        plt.ylabel("Número de turbinas Promedio")
        plt.title("Promedio mensual de numero de turbinas en un escenario " + scenarios_names[t])
        plt.legend(title="Turbinas")
        plt.xticks(rotation=0)
        t = t + 1
        plt.show()

