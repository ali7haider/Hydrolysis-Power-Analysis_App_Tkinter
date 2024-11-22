import tkinter as tk
from tkinter import ttk
import os
import pandas as pd
import pandas as pd
import tkinter.filedialog as filedialog
from tkinter import messagebox
from graph_functions import (graficar_datos_en_etiqueta,graficar_mapa_de_calor,graficar_dispersión_caudal_por_década,graficar_dispersión_nivel_por_década,graficar_dispersion_anual_caudal,
                             graficar_dispersión_anual_nivel,mostrar_estadísticas,graficar_distribucion_caudal,graficar_distribución_nivel,
                             graficar_densidad_probabilidad_caudal_por_década,graficar_densidad_probabilidad_nivel_por_década,
                             graficar_comportamiento_anual_por_década_caudal,graficar_comportamiento_anual_por_década_nivel,graficar_perfil_hidrológico_caudal,
                             graficar_perfil_hidrológico_nivel,graficar_perfil_anual_dias_caudal,graficar_perfil_anual_dias_nivel,mostrar_estadísticas_nominales,
                             calcular_P95_y_mostrar,mostrar_caudal_promedio,mostrar_nivel_P95,mostrar_velocidad_flujo,mostrar_comportamiento_mensual,mostrar_velocidad_promedio_mensual,
                            calcular_y_mostrar_potencia_turbina
                             )

class HydropowerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Hydropower Workflow Application")
        self.geometry("1100x700")
        # Center the main window on the screen
        self.center_window()
        self.config(bg="white")
        self.pages = {}

        # Initialize the pages
        self.create_pages()

        # Show the home page initially
        self.show_page(HomePage)
    def center_window(self):
            """Centers the main window on the screen."""
            # Get the screen width and height
            screen_width = self.winfo_screenwidth()
            screen_height = self.winfo_screenheight()

            # Get the dimensions of the main window
            window_width = 1060
            window_height = 680

            # Calculate the position of the window to center it
            position_top = (screen_height // 2) - (window_height // 2)
            position_left = (screen_width // 2) - (window_width // 2)

            # Set the position of the main window
            self.geometry(f'{window_width}x{window_height}+{position_left}+{position_top}')
    def create_pages(self):
        # First, create the HomePage instance
        home_page = HomePage(self)
        self.pages[HomePage] = home_page
        home_page.place(relwidth=1, relheight=1)

        # Pass the HomePage instance when creating ResultsPage
        results_page = ResultsPage(self, home_page)
        self.pages[ResultsPage] = results_page
        results_page.place(relwidth=1, relheight=1)

    def show_page(self, page_class):
        # Bring the desired page to the front
        page = self.pages[page_class]
        if isinstance(page, ResultsPage):
            page.populate_tree()  # Populate the Treeview with updated data
        page.tkraise()
class HomePage(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg="white")
        self.bg = tk.Frame(self, bg="white")
        self.bg.pack(fill=tk.BOTH, expand=True)
        self.parent=parent
        # Initialize file variables and other state variables
        self.caudal_file = None
        self.nivel_file = None
        self.caudal_data = None
        self.nivel_data = None
        self.caudal_f = None
        self.nivel_f = None
        self.caudal_process = None
        self.nivel_process = None
        self.graph_index = 0
        self.graphs = None

        self.años_secos = None
        self.años_humedos = None
        self.años_normales = None
        self.df_merge = None
        self.water_tra_value = None
        self.water_density_value = None
        self.turbine_option = None
        self.velocidad = None

        self.current_graph = 0  # Start with Caudal graph
        # Define the graphs to cycle through
        self.graphs = [
        {"function": graficar_dispersión_caudal_por_década, "titulo": "Serie Decadal de Caudal", "décadas": [1970, 1980, 1990]},
        {"function": graficar_dispersión_nivel_por_década, "titulo": "Serie Decadal de Nivel", "décadas": [1970, 1980, 1990]},
        {"function": graficar_dispersion_anual_caudal, "titulo": "Serie Anual de Caudal", "décadas": []},
        {"function": graficar_dispersión_anual_nivel, "titulo": "Serie Anual de Nivel", "décadas": []},
        {"function": mostrar_estadísticas, "titulo": "Resumen General de Estadísticas", "décadas": []},
        {"function": graficar_distribucion_caudal, "titulo": "Distribución con KDE y Curva Normal - Caudal", "décadas": []},
        {"function": graficar_distribución_nivel, "titulo": "Análisis de Distribución - Nivel", "décadas": []},
        {"function": graficar_densidad_probabilidad_caudal_por_década, "titulo": "Probabilidad de Densidad Decadal - Caudal", "décadas": [1970, 1980, 1990]},
        {"function": graficar_densidad_probabilidad_nivel_por_década, "titulo": "Probabilidad de Densidad Decadal - Nivel", "décadas": [1970, 1980, 1990]},
        {"function": graficar_comportamiento_anual_por_década_caudal, "titulo": "Comportamiento Anual de Caudal por Década", "décadas": [1970, 1980, 1990]},
        {"function": graficar_comportamiento_anual_por_década_nivel, "titulo": "Comportamiento Anual de Nivel por Década", "décadas": [1970, 1980, 1990]},
        {"function": graficar_perfil_hidrológico_caudal, "titulo": "Perfil Hidrológico Anual - Caudal", "décadas": []},
        {"function": graficar_perfil_hidrológico_nivel, "titulo": "Perfil Hidrológico Anual - Nivel", "décadas": []},
        {"function": graficar_perfil_anual_dias_caudal, "titulo": "Perfil Hidrológico Anual por Días - Caudal", "décadas": []},
        {"function": graficar_perfil_anual_dias_nivel, "titulo": "Perfil Hidrológico Anual por Días - Nivel", "décadas": []},
        {"function": mostrar_estadísticas_nominales, "titulo": "Estadísticas Nominales Caudal", "décadas": []},
        {"function": calcular_P95_y_mostrar, "titulo": "Gráfico de P95", "décadas": []},
        {"function": mostrar_caudal_promedio, "titulo": "Gráfico de Caudal Promedio", "décadas": []},
        {"function": mostrar_nivel_P95, "titulo": "Gráfico de Nivel 95", "décadas": []},
        {"function": mostrar_velocidad_flujo, "titulo": "Gráfico de Velocidad de Flujo", "décadas": []},
        {"function": mostrar_comportamiento_mensual, "titulo": "Gráfico de Comportamiento Mensual", "décadas": []},
        {"function": mostrar_velocidad_promedio_mensual, "titulo": "Gráfico de Velocidad Promedio Mensual", "décadas": []},
    ]
        self.turbine_graphs = [
        {
            "function": calcular_y_mostrar_potencia_turbina,
            "titulo": "Turbine Power Output Over Time",
            "opciones_turbina": [
                "SmartFreestream",
                "SmartMonofloat",
                "EnviroGen005series",
                "Hydroquest1.4",
                "EVG-050H",
                "EVG-025H"
            ],
            "description": "Displays individual power output plots for selected turbines."
        }
    ]
        
        # Initialize the UI components
        self.initialize_ui(parent)
        self.registrar_mensaje_2("Por favor, seleccione dos archivos de datos para comenzar: El primero debe comenzar con 'Q' y el segundo debe comenzar con 'NV'.\n")
        # Setup your Tkinter components like label and graph information area
        # self.lblGraph = tk.Label(parent)
        # self.lblGraph.pack()
        
    def load_data(self):
        """Function that loads the data files when btnCarga is clicked."""
        # Clear logs for the current step
        self.clear_logs()
        self.registrar_mensaje("Iniciando la carga de datos...\n")

        # First dialog box for selecting a file
        first_file = self.select_file()

        if first_file:
            # Extract the filename to check the prefix
            first_filename = os.path.basename(first_file)

            # Check if the first file starts with 'Q'
            if first_filename.startswith("Q"):
                self.caudal_file = first_file
                self.registrar_mensaje(f"Primer archivo (Q) seleccionado: {first_filename}\n")

                # Prompt user for second file with prefix "NV"
                second_file = self.select_file("NV")
            else:
                self.registrar_mensaje("Error: El primer archivo debe comenzar con 'Q'. Por favor, inténtelo nuevamente.")

                return  # Exit if the first file does not have 'Q' prefix
            
            # Check and assign the second file if selected correctly
            if second_file:
                second_filename = os.path.basename(second_file)
                
                # Check if second file starts with 'NV'
                if second_filename.startswith("NV"):
                    self.nivel_file = second_file
                    self.registrar_mensaje(f"Segundo archivo (NV) seleccionado: {second_filename}")
                    self.registrar_mensaje_2("Por favor, haga clic en 'Pretratamiento de datos' para continuar")

                else:
                    self.registrar_mensaje("Error: El segundo archivo debe comenzar con 'NV'. Por favor, intente de nuevo.")
                    return  # Exit if the second file does not have 'NV' prefix
                
                # Load data from files
                try:
                    self.caudal_data = pd.read_csv(self.caudal_file, delimiter='|', decimal='.')
                    self.nivel_data = pd.read_csv(self.nivel_file, delimiter='|', decimal='.')
                    
                    # Convert 'Fecha' to datetime format for both caudal and nivel datasets
                    self.caudal_data["Fecha"] = pd.to_datetime(self.caudal_data["Fecha"], errors='coerce')
                    self.nivel_data["Fecha"] = pd.to_datetime(self.nivel_data["Fecha"], errors='coerce')
                    
                    # Ensure 'Valor' column exists and filter valid dates
                    self.caudal_data = self.caudal_data.dropna(subset=["Fecha", "Valor"])
                    self.nivel_data = self.nivel_data.dropna(subset=["Fecha", "Valor"])
                    
                    # Enable the next button and disable the current one
                    self.btnCarga.config(state=tk.DISABLED)  # Disable btnCarga after successful file loading
                    self.btnPreta.config(state=tk.NORMAL)  # Enable btnPreta after successful file loading
                    # # Call this in your 'Procesamiento' (Processing) stage
                    self.show_graph()


                    # # Enable the next button after both files are selected
                    # self.btnPreta.setEnabled(True)
                    # self.btnCarga.setDisabled(True)  # Disable btnCarga to prevent re-clicking
                except Exception as e:
                    self.registrar_mensaje(f"Error al cargar los datos: {e}")
            else:
                self.registrar_mensaje("Error: Segundo archivo no seleccionado correctamente. Por favor, intente de nuevo.")
        else:
            self.registrar_mensaje("Error: Primer archivo no seleccionado correctamente. Por favor, intente de nuevo.")

    def select_file(self, required_prefix=None):
        """Function for file selection using Tkinter's filedialog"""
        # Open file dialog to select a file
        file_path = filedialog.askopenfilename(title="Select File", filetypes=[("Data files", "*.data")])
        
        if file_path:  # Check if a file was selected
            file_name = os.path.basename(file_path)  # Get the file name without path
            
            # If required_prefix is given, check if the file has the correct prefix
            if required_prefix and not file_name.startswith(required_prefix):
                self.registrar_mensaje(f"Error: El archivo seleccionado debe comenzar con '{required_prefix}'. Por favor, seleccione un archivo correcto.")
                return None  # Return None if the file does not match the expected prefix
            
            # Ensure the file has not been selected already for both 'Q' and 'NV'
            if (required_prefix == "Q" and file_path == self.nivel_file) or (required_prefix == "NV" and file_path == self.caudal_file):
                self.registrar_mensaje("Error: No puede seleccionar el mismo archivo para 'Q' y 'NV'. Por favor, elija un archivo diferente.")
                return None  # Return None to indicate an error

            return file_path  # Return the selected file path if it matches the prefix
        else:
            self.registrar_mensaje("No se ha seleccionado ningún archivo. Por favor, intente de nuevo.")
            return None  # Return None if no file was selected
    def show_graph(self):
        """Update the graph sequentially every 5 seconds using Tkinter's after() method."""
        if self.current_graph == 0:
            # Display Caudal line graph
            graficar_datos_en_etiqueta(self,self.caudal_data, etiqueta_y='Caudal (m3/s)', titulo='Gráfico de Línea de Valor en el Tiempo de Caudal (datos sin tratamiento)')
            self.current_graph = 1
        elif self.current_graph == 1:
            # Display Nivel line graph
            graficar_datos_en_etiqueta(self,self.nivel_data, etiqueta_y='Nivel (m)', titulo='Gráfico de Línea de Valor en el Tiempo de Nivel (datos sin tratamiento)')
            self.current_graph = 2
        elif self.current_graph == 2:
            # Display the Caudal heatmap
            graficar_mapa_de_calor(self,self.caudal_data, 'Caudal')
            self.current_graph = 3
        elif self.current_graph == 3:
            # Display the Nivel heatmap
            graficar_mapa_de_calor(self,self.nivel_data, 'Nivel')
            self.current_graph = 4  # Set to 4 to indicate we're done

        # Schedule the next update in 5000 milliseconds (5 seconds) if not finished
        if self.current_graph < 4:
            self.parent.after(5000, self.show_graph)  # Update graph every 5 seconds
        else:
            print("All graphs have been displayed.")  # Add a message for clarity

    

    def preprocess_data(self):
        """Function for the 'Pretratamiento' step, handling data completeness checks and interpolation."""
        self.clear_logs()  # Clear previous logs for clarity
        self.registrar_mensaje("Iniciando el preprocesamiento de datos...")


        # Load data from files
        self.caudal_data = pd.read_csv(self.caudal_file, delimiter='|', decimal='.')
        self.nivel_data = pd.read_csv(self.nivel_file, delimiter='|', decimal='.')

        # Convert 'Fecha' to datetime format
        try:
            self.caudal_data["Fecha"] = pd.to_datetime(self.caudal_data["Fecha"], errors='coerce')
            self.nivel_data["Fecha"] = pd.to_datetime(self.nivel_data["Fecha"], errors='coerce')
            self.registrar_mensaje("Conversión de fechas exitosa")
        except Exception as e:
            self.registrar_mensaje(f"Error en la conversión de fechas: {e}")
            return

        # Ensure 'Valor' column exists and filter valid dates
        self.caudal_data = self.caudal_data.dropna(subset=["Fecha", "Valor"])
        self.nivel_data = self.nivel_data.dropna(subset=["Fecha", "Valor"])
        # Group data by year and calculate missing data percentage
        for label, dataset in [("Caudal", self.caudal_data), ("Nivel", self.nivel_data)]:
            yearly_data = dataset.set_index("Fecha").resample("Y").count()["Valor"]
            yearly_data = yearly_data.rename("Records").reset_index()
            yearly_data["Year"] = yearly_data["Fecha"].dt.year
            yearly_data["Missing %"] = 100 * (1 - yearly_data["Records"] / 365)

            # Log results with section headers and separator lines
            self.registrar_mensaje(f"Completitud de los datos de {label} por año:")
            for _, row in yearly_data.iterrows():
                year, records, missing_pct = row["Year"], row["Records"], row["Missing %"]
                if missing_pct > 20:
                    color = 'red'
                    status = "Marcado para exclusión."
                else:
                    color = 'green'
                    status = "Los datos son utilizables."
                self.registrar_mensaje(f"Año {year}: {missing_pct:.2f}% de datos faltantes  - {status}")

        
        # Process each dataset for interpolation and statistics
        for label, dataset in [("Caudal", self.caudal_data), ("Nivel", self.nivel_data)]:
            # Create a copy to avoid modifying original data
            dataset = dataset.copy()

            # Filter out zero and null values
            dataset = dataset[(dataset["Valor"] != 0) & (~dataset["Valor"].isna())]

            # Set 'Fecha' as index and ensure full date range
            dataset.set_index("Fecha", inplace=True)
            full_index = pd.date_range(start=dataset.index.min(), end=dataset.index.max(), freq="D")
            dataset = dataset.reindex(full_index)
            
            # Fill missing consecutive NaNs if <= 20% of the year (~73 days)
            dataset["NaN_count_pre"] = dataset["Valor"].isna().astype(int).groupby(dataset["Valor"].notna().astype(int).cumsum()).cumsum()

            def interpolate_if_needed(group):
                max_consecutive_nans = group["NaN_count_pre"].max()
                if max_consecutive_nans <= 73:
                    group["Valor"] = group["Valor"].interpolate()
                return group

            dataset = dataset.groupby(dataset.index.year, group_keys=False).apply(interpolate_if_needed)
            dataset.drop(columns=["NaN_count_pre"], inplace=True)  # Remove helper column after interpolation

            # Reset index and add decade column for summary
            dataset.reset_index(inplace=True)
            dataset.rename(columns={'index': 'Fecha'}, inplace=True)
            dataset["Decade"] = dataset["Fecha"].dt.year // 10 * 10
            print(dataset)
            if label=="Caudal":
                self.caudal_f = dataset[(dataset['Valor'] != 0) & (~dataset['Valor'].isna())]
                self.caudal_process=dataset.copy() 
            elif label=="Nivel":
                self.nivel_f = dataset[(dataset['Valor'] != 0) & (~dataset['Valor'].isna())]
                self.nivel_process=dataset.copy() 

            # Display Decade-wise statistics
            decade_stats = dataset.groupby("Decade")["Valor"].describe()
            self.registrar_mensaje(f"Estadísticas de los datos de {label} por década:")
            self.registrar_mensaje(decade_stats.to_string())

            # Mostrar estadísticas para los NaNs consecutivos después de la interpolación
            dataset.set_index("Fecha", inplace=True)
            dataset["NaN_count_post"] = dataset["Valor"].isna().astype(int).groupby(dataset["Valor"].notna().astype(int).cumsum()).cumsum()
            consecutive_nans_post = dataset.groupby(dataset.index.year)["NaN_count_post"].max()
            self.registrar_mensaje(f"{label} NaNs consecutivos por año después de la interpolación:")
            self.registrar_mensaje(consecutive_nans_post.to_string())

            # Registrar la finalización del procesamiento
            self.registrar_mensaje(f"Preprocesamiento de datos de {label} completado con resumen por décadas.")

        self.clear_logs_2()
        self.registrar_mensaje_2("Por favor, haga clic en 'Tratamiento de datos' para continuar.\n")

        # Enable the next button after preprocessing
        self.btnPreta.config(state=tk.DISABLED)  # Disable btnCarga after successful file loading
        self.btnTrata.config(state=tk.NORMAL)  # Enable btnPreta after successful file loading
    def treat_data(self):
        """Handle the Tratamiento button click to switch to page 1, filter available years, check data completeness, and fill missing values."""
        
        # Switch to the stack widget page with index 1
        
        # Initialize text to show usable years
        usable_years_text = ""
        self.clear_logs_2()
        self.registrar_mensaje_2("\n Haga clic en Confirmar para continuar. \n")


        if self.caudal_data is not None and self.nivel_data is not None:
            # Loop through both datasets to calculate completeness and handle imputation
            for label, dataset in [("Caudal", self.caudal_data), ("Nivel", self.nivel_data)]:
                # Resample data by year and calculate record counts
                yearly_data = dataset.set_index("Fecha").resample("Y").count()["Valor"]
                yearly_data = yearly_data.rename("Records").reset_index()
                yearly_data["Year"] = yearly_data["Fecha"].dt.year
                yearly_data["Missing %"] = 100 * (1 - yearly_data["Records"] / 365.25)
                
                # Process usable years (less than 20% missing data)
                usable_years = yearly_data[yearly_data["Missing %"] < 20]["Year"].tolist()
                if usable_years:
                    # Add usable years to the text display
                    usable_years_text += f"Años Utilizables de Datos {label}: {', '.join(map(str, sorted(usable_years)))}\n\n"
                else:
                    usable_years_text += f"Años Utilizables de Datos {label}: None\n"


            # Update the availableYears QTextEdit with usable years
            self.availableYears.config(state=tk.NORMAL)  # Enable text box to update logs
            self.availableYears.delete(1.0, tk.END)  # Delete all text
            self.availableYears.insert(tk.END, usable_years_text + "\n")
            self.availableYears.config(state=tk.DISABLED)  # Disable text box to make it read-only

            
       
        else:
            # Error if data not loaded
            self.registrar_mensaje("Error: Either Caudal or Nivel data is not loaded. Please load both datasets first.")
    def registrar_mensaje(self, message):
        """Helper function to add messages to the Text log area."""
        self.graphInformation.config(state=tk.NORMAL)  # Enable text box to update logs
        self.graphInformation.insert(tk.END, message + "\n")  # Append message with newline to the log
        self.graphInformation.config(state=tk.DISABLED)  # Disable text box to make it read-only

    def clear_logs(self):
        """Function to clear the log area."""
        self.graphInformation.config(state=tk.NORMAL)  # Enable text box to clear
        self.graphInformation.delete(1.0, tk.END)  # Delete all text
        self.graphInformation.config(state=tk.DISABLED)  # Disable text box to make it read-only
    def registrar_mensaje_2(self, message):
    
        """Helper function to add messages to the Text log area."""
        self.textEditLogs.config(state=tk.NORMAL)  # Enable text box to update logs
        self.textEditLogs.insert(tk.END, message + "\n")   # Append message to the log
        self.textEditLogs.config(state=tk.DISABLED)  # Disable text box to make it read-only

    def clear_logs_2(self):
        """Function to clear the log area."""
        self.textEditLogs.config(state=tk.NORMAL)  # Enable text box to clear
        self.textEditLogs.delete(1.0, tk.END)  # Delete all text
        self.textEditLogs.config(state=tk.DISABLED)  # Disable text box to make it read-only
    def mostrar_información(self, message):
        self.limpiar_información()
        """Display the given message in the graph information area (Text widget)."""
        self.graphInformation.config(state=tk.NORMAL)  # Enable text box to update
        self.graphInformation.delete(1.0, tk.END)  # Clear any previous content
        self.graphInformation.insert(tk.END, message)  # Insert new message
        self.graphInformation.config(state=tk.DISABLED)  # Disable text box to make it read-only

    def limpiar_información(self):
        """Clear the information displayed in the graph information area (Text widget)."""
        self.graphInformation.config(state=tk.NORMAL)
        self.graphInformation.delete(1.0, tk.END)  # Clear the content
        self.graphInformation.config(state=tk.DISABLED)  # Disable the text box again
    
    
    
    def get_usable_years(self, dataset):
        """Helper function to get years with less than 20% missing data from the dataset."""
        # Resample the dataset by year and calculate missing percentage
        yearly_data = dataset.set_index("Fecha").resample("Y").count()["Valor"]
        yearly_data = yearly_data.rename("Records").reset_index()
        yearly_data["Year"] = yearly_data["Fecha"].dt.year
        yearly_data["Missing %"] = 100 * (1 - yearly_data["Records"] / 365.25)
        
        # Filter years with less than 20% missing data
        usable_years = yearly_data[yearly_data["Missing %"] < 20]["Year"].tolist()
        return set(usable_years)


    def confirm_years(self):
        """Check entered years for dry, wet, and normal, and validate against available years for both Caudal and Nivel, considering missing data."""
        if self.caudal_data is None or self.nivel_data is None:
            messagebox.showerror("Error de Validación", "Los datos de Caudal o Nivel no están disponibles para la validación.")
            return False

        # Get usable years for Caudal dataset (less than 20% missing data)
        usable_years_caudal = self.get_usable_years(self.caudal_data)

        # Get usable years for Nivel dataset (less than 20% missing data)
        usable_years_nivel = self.get_usable_years(self.nivel_data)

        # Get the union of usable years from both datasets
        usable_years = usable_years_caudal | usable_years_nivel

        try:
            self.clear_logs()  # Clear any previous logs

            # Read years from Entry fields and convert them to sets of integers
            años_secos = set(map(int, self.txtDryYears.get().split(',')))
            años_humedos = set(map(int, self.txtWetYears.get().split(',')))
            años_normales = set(map(int, self.txtNormalYears.get().split(',')))

            # Check if all entered years are in usable years
            invalid_years = (años_secos | años_humedos | años_normales) - usable_years
            if invalid_years:
                messagebox.showerror(
                "Años Inválidos",
                f"Años ingresados no válidos: {', '.join(map(str, invalid_years))}.\n"
                "Por favor, ingrese únicamente años utilizables."
            )

                return False
            else:
                self.registrar_mensaje("Años Secos, Húmedos y Normales confirmados con éxito y guardados.")
                self.registrar_mensaje_2("\nPor favor, haga clic en 'Procesamiento' para continuar.\n\n")

                # Save valid years for further processing
                self.años_secos = años_secos
                self.años_humedos = años_humedos
                self.años_normales = años_normales

                print("secos años:", self.años_secos)
                print("humedos años:", self.años_humedos)
                print("normales años:", self.años_normales)
                # After processing, enable the next button and disable the current one
                self.btnProce.config(state=tk.NORMAL)
                self.btnTrata.config(state=tk.DISABLED)
                return True

        except ValueError:
            messagebox.showerror("Input Error", "Error: Please enter only numeric years, separated by commas.")
            return False


    def combined_function(self, func1, func2):
        """Call two functions sequentially."""
        if func1():  # Check if func1 returns True
            func2()  # Only call func2 if func1 was successful

    def open_popup_window(self):
        """Creates and opens a pop-up window containing the combo box and button, centered on the screen."""
        # Create a new top-level window (pop-up)
        popup = tk.Toplevel(self.bg)
        popup.title("Seleccionar Gráfico de Turbina")  # Set title for the pop-up window

        # Set the size for the pop-up window (adjust as needed)
        popup_width = 400
        popup_height = 300

        # Get the screen width and height
        screen_width = popup.winfo_screenwidth()
        screen_height = popup.winfo_screenheight()

        # Calculate the position of the window to center it
        position_top = (screen_height // 2) - (popup_height // 2)
        position_left = (screen_width // 2) - (popup_width // 2)

        # Set the position and size of the pop-up window
        popup.geometry(f'{popup_width}x{popup_height}+{position_left}+{position_top}')

        # Container for input fields (centered content)
        input_container = tk.Frame(popup, bg="white")
        input_container.pack(fill=tk.BOTH, expand=True, pady=20)  # Center the container both vertically and horizontally

        # Create a frame to center the combo box and button horizontally
        center_frame = tk.Frame(input_container, bg="white")
        center_frame.pack(expand=True)  # This will center the contents within input_container

        # Combo Box for selecting a processing method
        self.cmbxTurbineGraph = self.create_combo_box(center_frame, "Seleccionar Gráficos de Turbinas", 
                                                    ["Todos", "SmartFreestream", "SmartMonofloat", "EnviroGen005series", 
                                                    "Hydroquest1.4", "EVG-050H", "EVG-050H"])

        # Confirm button
        self.btnConfirm_2 = tk.Button(
            center_frame,
            text="Confirmar",
            bg="#4d4eba",  # Background color
            fg="white",  # Text color
            font=("MS Shell Dlg 2", 12),
            width=10,
            relief="flat",  # Flat border
            bd=2,  # Border width
            highlightbackground="#4d4eba",  # Border color
            highlightthickness=2,  # Border thickness
            cursor="hand2",  # Pointer cursor
            command=lambda: self.confirm_turbine_selection(popup)  # Wrap the function call
        )
        self.btnConfirm_2.pack(pady=20, padx=20, anchor='e')  # Padding for the button

    def confirm_turbine_selection(self, popup):
        """
        Confirms the selected turbine graph, saves it to self.turbine_option,
        updates button states, and closes the pop-up window.
        """
        # Get the selected option from the combo box
        self.turbine_option = self.cmbxTurbineGraph.get()
        
        if not self.turbine_option:
            # Show an error message if no option is selected
            from tkinter import messagebox
            messagebox.showerror("Error de Selección", "Por favor, seleccione un gráfico de turbina antes de confirmar.")
            return

        # Save the selected option
        print(f"Seleccionar Gráfico de Turbina: {self.turbine_option}")

        # Update button states
        self.btnResult.config(state=tk.NORMAL)  # Activate btnResult
        self.btnProce.config(state=tk.DISABLED)  # Disable btnProce
        self.registrar_mensaje("\nGráfico de la turbina seleccionado y guardado.")
        self.clear_logs_2()
        self.registrar_mensaje_2("\nPor favor, haga clic en 'Resultados' para continuar.\n\n")

        # Close the pop-up window
        popup.destroy()
    def show_results(self):
        """Display each decade graph for caudal, nivel, and yearly graphs with 5-second intervals."""
        self.clear_logs()
        self.clear_logs_2()
        self.registrar_mensaje_2("Mostrando resultados (Cada gráfico cambiará cada 5 segundos)\n")
        self.btnResult.config(state=tk.DISABLED)  # Disable btnResult to prevent re-clicking

        


        # Initialize variables for cycling through graphs and decades
        self.graph_index = 0
        self.decade_index = 0

        # Start displaying the first graph
        self.update_graph_display()

    def update_graph_display(self):
        """Update the graph display for each graph."""
        # Check if all graphs are displayed
        if self.graph_index >= len(self.graphs):
            # Handle turbine plots after regular graphs
            if self.turbine_option == 'Todos':
                # Define the turbine options
                opciones_turbina = [
                    "SmartFreestream",
                    "SmartMonofloat",
                    "EnviroGen005series",
                    "Hydroquest1.4",
                    "EVG-050H",
                    "EVG-025H"
                ]

                # Ensure opciones_turbina are processed one by one
                if not hasattr(self, "turbine_index"):
                    self.turbine_index = 0  # Initialize the turbine index

                if self.turbine_index < len(opciones_turbina):
                    # Display the turbine plot
                    current_turbine = opciones_turbina[self.turbine_index]
                    print(current_turbine)
                    calcular_y_mostrar_potencia_turbina(self,None, current_turbine,titulo='', bandera=False,index=self.turbine_index)

                    # Move to the next turbine
                    self.turbine_index += 1

                    # Schedule the next turbine plot after 5 seconds
                    self.parent.after(5000, self.update_graph_display)
                    return
                else:
                    # Reset turbine index after completion
                    del self.turbine_index
                    self.registrar_mensaje("Gráficos de la turbina seleccionados mostrados.")
                    return
            else:
                # Handle single turbine or default case
                calcular_y_mostrar_potencia_turbina(self,None, self.turbine_option,titulo='', bandera=False,index=0)
                return

        # Regular graph display logic
        graph_info = self.graphs[self.graph_index]
        graph_function = graph_info["function"]
        title = graph_info["titulo"]
        decades = graph_info["décadas"]

        # Call the function with the instance and additional parameters
        if decades:
            current_decade = decades[self.decade_index]
            graph_function(self, None,title, current_decade, bandera=False)
            self.decade_index += 1
            if self.decade_index >= len(decades):
                self.decade_index = 0
                self.graph_index += 1
        else:
            graph_function(self, None,title)
            self.graph_index += 1

        # Schedule the next graph update after 1 second
        self.parent.after(1000, self.update_graph_display)



    def initialize_ui(self, parent):
        """Function to initialize the UI components"""

        # Create the frame for the sidebar
        self.frame = tk.Frame(self.bg, bg="white", width=400)
        self.frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # Create a user section
        user_frame = tk.Frame(self.frame, bg="white", height=5)
        user_frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=10)
        user_label = tk.Label(user_frame, text="Usuario", bg="white", fg="black", font=("MS Shell Dlg 2", 14, "bold"), anchor="w")
        user_label.pack(fill=tk.BOTH, expand=False)

        # Create the button section
        self.button_frame = tk.Frame(self.frame, bg="white", padx=10)
        self.button_frame.pack(fill=tk.Y, expand=True, pady=(80, 0))  # Top padding: 20, Bottom padding: 0

        # Create the buttons (equivalent to btnCarga, btnPreta, etc.)
        self.btnCarga = tk.Button(self.button_frame, text="Carga de archivos", bg="blue", fg="white", font=("MS Shell Dlg 2", 12), width=20, command=self.load_data)
        self.btnCarga.pack(pady=5)
        self.apply_button_styles(self.btnCarga)

        self.btnPreta = tk.Button(self.button_frame, text="Pretratamiento de datos", bg="blue", fg="white", font=("MS Shell Dlg 2", 12), width=20, state=tk.DISABLED,command=self.preprocess_data)
        self.btnPreta.pack(pady=5)
        self.apply_button_styles(self.btnPreta)

        self.btnTrata = tk.Button(
            self.button_frame,
            text="Tratamiento de datos",
            bg="blue",
            fg="white",
            font=("MS Shell Dlg 2", 12),
            width=20,
            state=tk.DISABLED,
            command=lambda: self.combined_function(self.show_Tratamiento, self.treat_data)
        )
        self.btnTrata.pack(pady=5)
        self.apply_button_styles(self.btnTrata)

        self.btnProce = tk.Button(self.button_frame, text="Procesamiento", bg="blue", fg="white", font=("MS Shell Dlg 2", 12), width=20, state=tk.DISABLED, command=self.open_popup_window)
        self.btnProce.pack(pady=5)
        self.apply_button_styles(self.btnProce)

        self.btnResult = tk.Button(self.button_frame, text="Resultados", bg="blue", fg="white", font=("MS Shell Dlg 2", 12), width=20, state=tk.DISABLED,command=self.show_results)
        self.btnResult.pack(pady=5)
        self.apply_button_styles(self.btnResult)
        
        self.btnSimulator = tk.Button(self.button_frame, text="Simular",bg="#4d4eba",fg="white", font=("MS Shell Dlg 2", 12), width=20, relief="flat", bd=2, highlightbackground="#4d4eba", highlightthickness=2, cursor="hand2")
        self.btnSimulator.pack(pady=5, anchor='e')

        # Add hover effects specific to btnSimulator
        # self.btnSimulator.bind("<Enter>", lambda e: self.on_simulator_hover())
        # self.btnSimulator.bind("<Leave>", lambda e: self.on_simulator_leave())
        
        # Create the log section (Text widget)
        text_frame = tk.Frame(self.frame, bg="white", width=35)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.textEditLogs = tk.Text(text_frame, height=10, width=35, bg="#e9ecef",font=("MS Shell Dlg 2", 11), wrap="word",padx=10, pady=5, fg="#6c757d")
        self.textEditLogs.config(state=tk.DISABLED)  # Making it read-only
        self.textEditLogs.pack(fill=tk.BOTH, pady=10, expand=True)

        # Create the content section on the right side (like stacked widget)
        self.content_frame = tk.Frame(self.bg, bg="white")
        self.content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)  # Expands to occupy the remaining space
        
        # Create Tratamiento page (hidden by default)
        self.page_procesamiento = self.create_procesamiento_page()
        self.page_procesamiento.pack_forget()  # Initially hidden

        self.page_Tratamiento = self.create_Tratamiento_page()
        self.page_Tratamiento.pack_forget()  # Initially hidden

        # Button for "View All Graphs"
        self.btnViewAllGraphs = tk.Button(self.content_frame, text="Ver todos los gráficos", bg="#4d4eba", fg="white", font=("MS Shell Dlg 2", 12), width=20, relief="flat", bd=2, highlightbackground="#4d4eba", highlightthickness=2, cursor="hand2", command=lambda: parent.show_page(ResultsPage))
        self.btnViewAllGraphs.pack(side=tk.TOP, pady=10, padx=10, anchor="e")

        # Add hover effects specific to btnViewAllGraphs
        self.btnViewAllGraphs.bind("<Enter>", lambda e: self.on_view_all_graphs_hover())
        self.btnViewAllGraphs.bind("<Leave>", lambda e: self.on_view_all_graphs_leave())

        self.lblGraph = tk.Label(self.content_frame, text="El gráfico se mostrará aquí", bg="#f5f5f5", font=("MS Shell Dlg 2", 14), borderwidth=2, relief="solid")
        self.lblGraph.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)



       # Add a frame for the text widget
        self.info_frame = tk.Frame(self.content_frame, bg="#ffffff", height=200)  # Set a minimum height
        self.info_frame.pack(fill=tk.BOTH, padx=10, pady=10)

        # Disable auto-resizing of the frame
        self.info_frame.pack_propagate(False)

        # Add a Text widget to show graph information within the info frame
        self.graphInformation = tk.Text(self.info_frame, bg="#e9ecef", font=("MS Shell Dlg 2", 11), wrap="word",padx=10, pady=5, fg="#6c757d")
        self.graphInformation.config(state=tk.DISABLED)  # Make it read-only
        self.graphInformation.pack(fill=tk.BOTH, expand=True)


    
    def apply_button_styles(self, button):
        # Apply normal style to the button
        button.config(
            bg="white", 
            fg="black", 
            bd=2,  # border width
            relief="solid",  # solid border to replicate Qt style
            font=("MS Shell Dlg 2", 12),  # Font size and style
            padx=3,  # padding inside button
            pady=3,  # padding inside button
            highlightthickness=0,  # no focus highlight border
            width=20 , # Width of the button
            cursor="hand2",  # Pointer cursor

        )

        
        # Hover effect on button (change color and border)
        def on_hover(event):
            button.config(bg="white", fg="#4d4eba", bd=2, relief="solid")

        def on_leave(event):
            button.config(bg="white", fg="black", bd=2, relief="solid")

        # Bind hover events
        button.bind("<Enter>", on_hover)
        button.bind("<Leave>", on_leave)
    def create_Tratamiento_page(self):
        """Creates the Tratamiento page."""
        page = tk.Frame(self.bg, bg="white")  # Direct child of self.bg to replace content_frame

        # Text widget
        self.availableYears = tk.Text(
            page, height=10, width=50, bg="#e9ecef", wrap="word", font=("MS Shell Dlg 2", 11),padx=10, pady=5, fg="#6c757d"
        )
        self.availableYears.insert("1.0", "Available years information will go here.")
        self.availableYears.config(state=tk.DISABLED)
        self.availableYears.pack(fill=tk.X, pady=50, padx=20)

        # Container for input fields, centered on the page
        input_container = tk.Frame(page, bg="white")
        input_container.pack(fill=tk.BOTH, expand=True, pady=30)  # Center the container vertically

        # Input fields
        self.txtDryYears = self.create_input_field(input_container, "Años Secos", "Ingrese los Años Secos (Separados por comas)")
        self.txtWetYears = self.create_input_field(input_container, "Años Húmedos", "Ingrese los Años Húmedos (Separados por comas)")
        self.txtNormalYears = self.create_input_field(input_container, "Años Normales", "Ingrese los Años Normales (Separados por comas)")


        # Confirm button
        self.btnConfirm = tk.Button(
            input_container,
            text="Confirmar",
            bg="#4d4eba",  # Background color
            fg="white",  # Text color
            font=("MS Shell Dlg 2", 12),
            width=15,
            
            relief="flat",  # Flat border
            bd=2,  # Border width
            highlightbackground="#4d4eba",  # Border color
            highlightthickness=2,  # Border thickness
            cursor="hand2",  # Pointer cursor
            command=lambda: self.combined_function(self.confirm_years,self.show_content_frame)

        )
        self.btnConfirm.pack(pady=20,padx=20,anchor='e')  # Padding for the button

        return page

    def create_input_field(self, parent, label_text, placeholder="Enter text..."):
        """Creates a styled input field with a label above it, aligned to the left, and with a placeholder."""
        frame = tk.Frame(parent, bg="white")
        frame.pack(pady=10)  # Add vertical spacing between input fields

        # Label (above the entry field)
        label = tk.Label(frame, text=label_text, bg="white", font=("MS Shell Dlg 2", 12), anchor="w", justify="left")
        label.pack(fill=tk.X, padx=20,pady=5, anchor="w")  # Align to the left with padding

        # Entry field with placeholder functionality
        entry_frame = tk.Frame(frame, bg="white")  # Wrapper for the entry and the border
        entry_frame.pack(fill=tk.X, padx=20)

        entry = tk.Entry(
            entry_frame,
            font=("MS Shell Dlg 2", 13),
            bg="white",
            fg="grey",  # Initial placeholder color
            relief="flat",  # Remove border
            width=100,
        )
        entry.pack()

        # Simulated bottom border
        bottom_border = tk.Frame(entry_frame, height=1, bg="gray")  # Adjust color as needed
        bottom_border.pack(fill=tk.X)

        # Add placeholder functionality
        def on_focus_in(event):
            if entry.get() == placeholder:
                entry.delete(0, "end")
                entry.config(fg="black")  # Change text color to black when typing

        def on_focus_out(event):
            if not entry.get():  # If empty, show placeholder again
                entry.insert(0, placeholder)
                entry.config(fg="grey")

        entry.insert(0, placeholder)  # Set initial placeholder text
        entry.bind("<FocusIn>", on_focus_in)
        entry.bind("<FocusOut>", on_focus_out)

        return entry


    def create_procesamiento_page(self):
        """Creates the Procesamiento page."""
        page = tk.Frame(self.bg, bg="white")  # Direct child of self.bg to replace content_frame

        # Container for input fields (centered content)
        input_container = tk.Frame(page, bg="white")
        input_container.pack(fill=tk.BOTH, expand=True)  # Center the container both vertically and horizontally

        # Create a frame to center the combo box and button horizontally
        center_frame = tk.Frame(input_container, bg="white")
        center_frame.pack(expand=True)  # This will center the contents within input_container

        # Combo Box for selecting a processing method
        self.cmbxTurbineGraph = self.create_combo_box(center_frame, "Seleccionar Gráficos de Turbinas", 
                                                    ["All", "SmartFreestream", "SmartMonofloat", "EnviroGen005series", 
                                                    "Hydroquest1.4", "EVG-050H", "EVG-050H"])

        # Confirm button
        self.btnConfirm_2 = tk.Button(
            center_frame,
            text="Confirmar",
            bg="#4d4eba",  # Background color
            fg="white",  # Text color
            font=("MS Shell Dlg 2", 12),
            width=10,
            relief="flat",  # Flat border
            bd=2,  # Border width
            highlightbackground="#4d4eba",  # Border color
            highlightthickness=2,  # Border thickness
            cursor="hand2",  # Pointer cursor
            # command=self.show_content_frame  # Button command to go back to main content
        )
        self.btnConfirm_2.pack(pady=20, padx=20, anchor='e')  # Padding for the button

        return page

    def create_combo_box(self, parent, label_text, options):
        """Creates a combo box with a label above it."""
        frame = tk.Frame(parent, bg="white")
        frame.pack(pady=10)  # Add vertical spacing between input fields

        # Label (above the combo box)
        label = tk.Label(frame, text=label_text, bg="white", font=("MS Shell Dlg 2", 12), anchor="w")
        label.pack(fill=tk.X, padx=20, pady=5, anchor="w")  # Align to the left with padding

        # Combo Box for selecting options
        combo_box = ttk.Combobox(frame, values=options, font=("MS Shell Dlg 2", 13), state="readonly")
        combo_box.pack(fill=tk.X, padx=20)
        combo_box.set(options[0])  # Set default selection (optional)

        return combo_box


    def show_Tratamiento(self):
        """Hides content_frame and shows the Tratamiento page."""
        # Hide the main content frame
        if hasattr(self, 'content_frame'):
            self.content_frame.pack_forget()
        
        # Hide the Procesamiento page if it's visible
        if hasattr(self, 'page_procesamiento'):
            self.page_procesamiento.pack_forget()

        # Show the Tratamiento page
        if hasattr(self, 'page_Tratamiento'):
            self.page_Tratamiento.pack(fill=tk.BOTH, expand=True)
        return True


    def show_procesamiento(self):
        """Hides content_frame and shows the Procesamiento page."""
        # Hide the main content frame
        if hasattr(self, 'content_frame'):
            self.content_frame.pack_forget()

        # Hide the Tratamiento page if it's visible
        if hasattr(self, 'page_Tratamiento'):
            self.page_Tratamiento.pack_forget()
        
        # Show the Procesamiento page
        if hasattr(self, 'page_procesamiento'):
            self.page_procesamiento.pack(fill=tk.BOTH, expand=True)


    def show_content_frame(self):
        """Hides any active page (Tratamiento or Procesamiento) and shows the main content frame."""
        # Hide Tratamiento page if it's visible
        if hasattr(self, 'page_Tratamiento'):
            self.page_Tratamiento.pack_forget()

        # Hide Procesamiento page if it's visible
        if hasattr(self, 'page_procesamiento'):
            self.page_procesamiento.pack_forget()

        # Show the main content frame
        if hasattr(self, 'content_frame'):
            self.content_frame.pack(fill=tk.BOTH, expand=True)


    def on_view_all_graphs_hover(self):
        """Handles hover effects specifically for btnViewAllGraphs."""
        self.btnViewAllGraphs.config(bg="#6c70ca", fg="#f6f8ff")

    def on_view_all_graphs_leave(self):
        """Handles leave effects specifically for btnViewAllGraphs."""
        self.btnViewAllGraphs.config(bg="#4d4eba", fg="white")
    def on_simulator_hover(self):
        """Handles hover effects specifically for btnViewAllGraphs."""
        self.btnSimulator.config(bg="#6c70ca", fg="#f6f8ff")

    def on_simulator_leave(self):
        """Handles leave effects specifically for btnViewAllGraphs."""
        self.btnSimulator.config(bg="#4d4eba", fg="white")
class ResultsPage(tk.Frame):
    def __init__(self, parent, home_page):
        super().__init__(parent, bg="white")
        self.home_page = home_page  # Reference to HomePage instance

        # Initialize the UI components
        self.initialize_ui(parent)

    
    def handle_graph_selection(self, event):
        """Handle when a graph item is selected in the Treeview."""
        try:
            # Ensure both files and processed data are available
            if not self.home_page.caudal_file or not self.home_page.nivel_file:
                self.show_warning("Missing Files", "Please select both 'Caudal' and 'Nivel' files before proceeding.")
                return

            if self.home_page.caudal_process is None or self.home_page.nivel_process is None:
                self.show_warning("Data Not Processed", "Please process both 'Caudal' and 'Nivel' data before proceeding.")
                return

            # Get the selected item
            selected_item = self.tree.selection()
            if not selected_item:
                return
            selected_item_id = selected_item[0]
            selected_item_text = self.tree.item(selected_item_id, "text")

            # Determine if the selection is a decade or a main graph
            parent_item_id = self.tree.parent(selected_item_id)
            if selected_item_text.startswith("Decade"):
                graph_title = self.tree.item(parent_item_id, "text")
                selected_decade = int(selected_item_text.split()[-1])

                for graph in self.home_page.graphs:
                    if graph["titulo"] == graph_title:
                        params = {"titulo": graph["titulo"], "décadas": [selected_decade], "bandera": True}
                        self.display_graph(graph["function"], params)
                        break
            else:
                graph_title = selected_item_text
                for graph in self.home_page.graphs:
                    if graph["titulo"] == graph_title:
                        params = {"titulo": graph["titulo"], "décadas": graph.get("décadas", []), "bandera": True}
                        self.display_graph(graph["function"], params)
                        break
                else:
                    for tgraph in self.home_page.turbine_graphs:
                        if graph_title in tgraph["opciones_turbina"]:
                            params = {
                                "opciones_turbina": selected_item_text,  # The selected turbine
                                "titulo": tgraph["titulo"],
                                "bandera": True,
                                "index": 0
                            }
                            self.display_graph(tgraph["function"], params)
                            break

        except Exception as e:
            # self.log_message(f"Error handling graph selection: {e}\n")
            print(f"Error handling graph selection: {e}\n")

    def display_graph(self, graph_function, params=None):
        """Call the graph function and display the result."""
        try:
            # Ensure params is a dictionary to include self.home_page
            if params is None:
                params = {}
            # Add self.home_page to params
            params['instancia'] = self.home_page
            params['instancia_resultados'] = self

            # Call the graph function with the provided parameters
            graph_function(**params)
        except Exception as e:
            print(f"Error displaying graph: {e}\n")


    def show_warning(self, title, message):
        """Display a warning message."""
        tk.messagebox.showwarning(title, message)
    def populate_tree(self):
        """Populate the Treeview with graphs and turbine graphs."""
        # Clear the Treeview
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Set column width to prevent text truncation
        self.tree.column("#0", width=300, stretch=True)
        self.tree.heading("#0", text="Gráficos")  # Set a heading for the column

        # Add "Graphs" category (collapsed by default)
        graphs_parent = self.tree.insert("", "end", text="Gráficos", open=False)

        # Add main graphs
        for graph in self.home_page.graphs:
            # Determine the parent category for the graph
            graph_title = graph["titulo"].strip()  # Ensure no extra whitespace
            if "Caudal" in graph_title:
                caudal_parent = self._find_or_create_sub_parent(graphs_parent, "Gráficos de Caudal", open=False)
                graph_item = self.tree.insert(caudal_parent, "end", text=graph_title, open=False)
            elif "Nivel" in graph_title:
                nivel_parent = self._find_or_create_sub_parent(graphs_parent, "Gráficos de Nivel", open=False)
                graph_item = self.tree.insert(nivel_parent, "end", text=graph_title, open=False)
            else:
                graph_item = self.tree.insert(graphs_parent, "end", text=graph_title, open=False)

            # Add decades as sub-items under the specific graph node
            for decade in graph.get("décadas", []):
                self.tree.insert(graph_item, "end", text=f"Década {decade}", open=False)

        # Add "Turbine Graphs" category (collapsed by default)
        turbine_parent = self.tree.insert("", "end", text="Gráficos de Turbinas", open=False)

        for tgraph in self.home_page.turbine_graphs:
            turbine_item = self.tree.insert(turbine_parent, "end", text=tgraph["titulo"], open=False)
            for option in tgraph.get("opciones_turbina", []):
                self.tree.insert(turbine_item, "end", text=option, open=False)

    def _find_or_create_sub_parent(self, parent, sub_title, open=False):
        """Helper function to find or create a sub-parent node."""
        # Check if the sub-parent already exists
        for child in self.tree.get_children(parent):
            if self.tree.item(child, "text") == sub_title:
                return child
        # Create a new sub-parent if not found
        return self.tree.insert(parent, "end", text=sub_title, open=open)

    def initialize_ui(self, parent):
        """Initialize the UI components for the ResultsPage."""
        self.bg = tk.Frame(self, bg="white")
        self.bg.pack(fill=tk.BOTH, expand=True)
        
        # Create the frame for the sidebar, similar to QFrame
        self.frame = tk.Frame(self.bg, bg="white", width=400)
        self.frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # Create a frame for the QTextEdit (Text widget) in the sidebar (light gray)
        user_frame = tk.Frame(self.frame, bg="white", height=5)
        user_frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=10)

        # Create the Label for user section and align it to the left
        user_label = tk.Label(user_frame, text="Graph List", bg="white", fg="black", font=("MS Shell Dlg 2", 14, "bold"))
        user_label.pack(fill=tk.BOTH, expand=False)
        
        # Create a frame below the label to contain the Treeview
        tree_frame = tk.Frame(self.frame, bg="white")
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Add a Treeview widget to the frame
        self.tree = ttk.Treeview(tree_frame, show="tree")  # Use only the default tree column (#0)
        self.tree.column("#0", width=320, stretch=True)   # Set width for the default tree column
        self.tree.heading("#0", text="Graphs")           # Set the heading for the default tree column
        self.tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Bind the selection event to the handle_graph_selection method
        self.tree.bind("<<TreeviewSelect>>", self.handle_graph_selection)

        # Create the content section on the right side (like stacked widget)
        self.content_frame = tk.Frame(self.bg, bg="white")
        self.content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)  # Expands to occupy the remaining space

        # Back button
        self.btnBack = tk.Button(
            self.content_frame,
            text="Volver",
            bg="#4d4eba",  # Background color
            fg="white",  # Text color
            font=("MS Shell Dlg 2", 12),
            width=20,
            relief="flat",  # Flat border
            bd=2,  # Border width
            highlightbackground="#4d4eba",  # Border color
            highlightthickness=2,  # Border thickness
            cursor="hand2",  # Pointer cursor
            command=lambda: parent.show_page(HomePage)
        )
        self.btnBack.pack(side=tk.TOP, pady=10, anchor="e")

        # Add hover effects specific to btnBack
        self.btnBack.bind("<Enter>", lambda e: self.on_btnBack_hover())
        self.btnBack.bind("<Leave>", lambda e: self.on_btnBack_leave())

        # Label for graph information
        self.lblGraph = tk.Label(
            self.content_frame, 
            text="El gráfico se mostrará aquí", 
            bg="lightgray", 
            font=("MS Shell Dlg 2", 14), 
            borderwidth=2, 
            relief="solid"
        )
        self.lblGraph.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Text widget to show graph information
        self.graphInformation = tk.Text(
            self.content_frame,
            font=("MS Shell Dlg 2", 11),
            height=10,
            width=50,
            padx=10, pady=5,
            bg="#e9ecef",
            fg="#6c757d"
        )
        self.graphInformation.config(state=tk.DISABLED)  # Making it read-only
        self.graphInformation.pack(fill=tk.BOTH, padx=10, pady=10)

    def apply_button_styles(self, button):
        # Apply normal style to the button
        button.config(
            bg="white", 
            fg="black", 
            bd=2,  # border width
            relief="solid",  # solid border to replicate Qt style
            font=("MS Shell Dlg 2", 12),  # Font size and style
            padx=3,  # padding inside button
            pady=3,  # padding inside button
            highlightthickness=0,  # no focus highlight border
            width=20 , # Width of the button
            cursor="hand2",  # Pointer cursor

        )

        
        # Hover effect on button (change color and border)
        def on_hover(event):
            button.config(bg="white", fg="#4d4eba", bd=2, relief="solid")

        def on_leave(event):
            button.config(bg="white", fg="black", bd=2, relief="solid")

        # Bind hover events
        button.bind("<Enter>", on_hover)
        button.bind("<Leave>", on_leave)
    def mostrar_información(self, message):
        self.limpiar_información()
        """Display the given message in the graph information area (Text widget)."""
        self.graphInformation.config(state=tk.NORMAL)  # Enable text box to update
        self.graphInformation.delete(1.0, tk.END)  # Clear any previous content
        self.graphInformation.insert(tk.END, message)  # Insert new message
        self.graphInformation.config(state=tk.DISABLED)  # Disable text box to make it read-only

    def limpiar_información(self):
        """Clear the information displayed in the graph information area (Text widget)."""
        self.graphInformation.config(state=tk.NORMAL)
        self.graphInformation.delete(1.0, tk.END)  # Clear the content
        self.graphInformation.config(state=tk.DISABLED)  # Disable the text box again
    def registrar_mensaje(self, message):
        """Helper function to add messages to the Text log area."""
        self.graphInformation.config(state=tk.NORMAL)  # Enable text box to update logs
        self.graphInformation.insert(tk.END, message + "\n")  # Append message with newline to the log
        self.graphInformation.config(state=tk.DISABLED)  # Disable text box to make it read-only
    def on_btnBack_hover(self):
        """Handles hover effects specifically for btnViewAllGraphs."""
        self.btnBack.config(bg="#6c70ca", fg="#f6f8ff")

    def on_btnBack_leave(self):
        """Handles leave effects specifically for btnViewAllGraphs."""
        self.btnBack.config(bg="#4d4eba", fg="white")
    

    
if __name__ == "__main__":
    app = HydropowerApp()
    app.mainloop()
