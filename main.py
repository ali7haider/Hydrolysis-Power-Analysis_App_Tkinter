import sys
import os
import pandas as pd
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog,QTreeWidget, QTreeWidgetItem,QMessageBox
from PyQt5.QtWidgets import QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pandas as pd
from PyQt5.QtGui import QPixmap
from io import BytesIO
from PIL import Image
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import skew, kurtosis, norm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import itertools
from PyQt5.QtCore import Qt
import matplotlib.dates as mdates
from Clase_turbinaV2 import Turbina

class MainScreen(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainScreen, self).__init__()
        uic.loadUi('main.ui', self)  # Load the main.ui file
        self.stackedWidget.setCurrentIndex(0)
        self.stackedWidget_2.setCurrentIndex(0)

        # Disable all buttons initially except btnCarga
        self.btnPreta.setDisabled(True)
        self.btnTrata.setDisabled(True)
        self.btnProce.setDisabled(True)
        self.btnResult.setDisabled(True)
        self.btnSimulator.setDisabled(True)
        
        # Enable only the btnCarga button
        self.btnCarga.setEnabled(True)

        # Initialize file variables
        self.caudal_file = None
        self.nivel_file = None
        self.caudal_data=None
        self.nivel_data=None
        self.caudal_f=None
        self.nivel_f=None
        self.caudal_process=None
        self.nivel_process=None
        self.graph_index = 0
        self.graphs = None

        self.dry_years = None
        self.wet_years = None
        self.normal_years = None
        self.df_merge= None
        self.water_tra_value = None
        self.water_density_value = None
        self.turbine_option=None
        self.velocidad= None
        # Connect buttons to their respective functions
        self.btnCarga.clicked.connect(self.load_data)
        self.btnPreta.clicked.connect(self.preprocess_data)
        self.btnTrata.clicked.connect(self.treat_data)
        self.btnProce.clicked.connect(self.process_data)
        self.btnResult.clicked.connect(self.show_results)
        # Add confirmation button for dry/wet/normal year inputs
        self.btnConfirm.clicked.connect(self.confirm_years)
        self.btnConfirm_2.clicked.connect(self.confirm_data)
        self.graphs_list = self.findChild(QTreeWidget, 'listGraphs')
        self.btnViewAllGraphs.clicked.connect(self.load_graph_list)  
        self.btnBack.clicked.connect(self.change_page)  
        self.graphs_list.itemClicked.connect(self.handle_graph_selection)
        # Log initial message in QTextEdit
        self.log_message_2("Please select two data files to begin: First one must start with 'Q' and and second one must start with 'NV'.\n")
        self.graphs = [
        {"function": self.plot_decade_scatter_caudal_graph, "title": "Decadal Caudal Series", "decades": [1970, 1980, 1990]},
        {"function": self.plot_decade_scatter_nivel_graph, "title": "Decadal Nivel Series", "decades": [1970, 1980, 1990]},
        {"function": self.plot_yearly_scatter_caudal_graph, "title": "Yearly Caudal Series", "decades": []},
        {"function": self.plot_yearly_scatter_nivel_graph, "title": "Yearly Nivel Series", "decades": []},
        {"function": self.show_statistics, "title": "Overall Statistics Summary", "decades": []}, # Fifth entry for statistics
        {"function": self.plot_distribution_caudal_graph, "title": "Distribution with KDE and Normal Curve - Caudal","decades": []},  # New graph
        {"function": self.plot_distribution_nivel_graph, "title": "Distribution Analysis - Nivel", "decades": []},
        {"function": self.plot_decadal_density_caudal_graph, "title": "Decadal Density Probability - Caudal", "decades": [1970, 1980, 1990]},
        {"function": self.plot_decadal_density_nivel_graph, "title": "Decadal Density Probability - Nivel", "decades": [1970, 1980, 1990]},
        {"function": self.plot_annual_behavior_decadal_caudal_graph, "title": "Annual Caudal Behavior by Decade", "decades": [1970, 1980, 1990]},
        {"function": self.plot_annual_behavior_decadal_nivel_graph, "title": "Annual Nivel Behavior by Decade", "decades": [1970, 1980, 1990]},
        {"function": self.plot_hydrological_profile_caudal, "title": "Perfil Hidrológico Anual - Caudal", "decades": []},
        {"function": self.plot_hydrological_profile_nivel, "title": "Perfil Hidrológico Anual - Nivel", "decades": []},
        {"function": self.plot_annual_profile_days_caudal, "title": "Perfil Hidrológico Anual por Días - Caudal", "decades": []},
        {"function": self.plot_annual_profile_days_nivel, "title": "Perfil Hidrológico Anual por Días - Nivel", "decades": []},
        {"function": self.show_nominal_stats, "title": "Nomial Stats Cuadal - Nivel", "decades": []},
        {"function": self.calculate_P95_and_display, "title": "P95 Graph", "decades": []},
        {"function": self.display_average_flow, "title": "Average Flow Graph", "decades": []},
        {"function": self.display_level_p95, "title": "Level 95 Graph", "decades": []},
        {"function": self.display_flow_velocity, "title": "Flow Velocity Graph", "decades": []},
        {"function": self.calculate_monthly_behavior, "title": "Monthly Behavior Graph", "decades": []},
        {"function": self.calculate_average_monthly_velocity, "title": "Average Monthly Average Graph", "decades": []}
    ]
        self.turbine_graphs = [
        {
            "function": self.calculate_and_display_turbine_power,
            "title": "Turbine Power Output Over Time",
            "turbine_options": [
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

    def handle_graph_selection(self, item):
        """Handle when a graph item is clicked in the QTreeWidget."""
        try:
            # Ensure both files and processed data are available
            if not self.caudal_file or not self.nivel_file:
                QMessageBox.warning(self, "Missing Files", "Please select both 'Caudal' and 'Nivel' files before proceeding.")
                return

            if self.caudal_process is None or self.nivel_process is None:
                QMessageBox.warning(self, "Data Not Processed", "Please process both 'Caudal' and 'Nivel' data before proceeding.")
                return

            selected_item_text = item.text(0)  # Get the text of the clicked item
            print(f"Selected Item: {selected_item_text}")

            # Check if the selected item is a decade or a main graph title
            if selected_item_text.startswith("Decade"):
                # If it's a decade (like "Decade 1970"), get the parent graph title
                parent_item = item.parent()
                graph_title = parent_item.text(0)  # Parent is the main graph title
                selected_decade = selected_item_text.split()[-1]  # Get the decade number (e.g., 1970)

                # Find the corresponding graph in self.graphs based on the parent graph's title
                for graph in self.graphs:
                    if graph["title"] == graph_title:
                        # Pass the graph title, decades list (with the selected decade), and flag
                        params = {
                            "title": graph["title"],
                            "decades": [int(selected_decade)],  # Only pass the selected decade
                            "flag": True,
                        }
                        self.display_graph(graph["function"], params)
                        break

            else:
                # If it's a main graph title (not a decade), process it as usual
                selected_graph_title = selected_item_text  # This is the main graph title
                parent_item = item.parent()
                graph_title = parent_item.text(0)  # Parent is the main graph title


                for graph in self.graphs:
                    if graph["title"] == selected_graph_title:
                        # Pass the graph's title and decades as parameters
                        params = {
                            "title": graph["title"],
                            "decades": graph.get("decades", []),
                            "flag": True,
                        }
                        self.display_graph(graph["function"], params)
                        break
                else:
                    # Check in turbine graphs if it's not found in the main graphs
                    for tgraph in self.turbine_graphs:
                        if tgraph["title"] == graph_title:
                            params = {
                                "turbine_options":selected_item_text,
                                "title": tgraph["title"],
                                "flag": True,
                                
                            }
                            self.display_graph(tgraph["function"], params)
                            break

        except Exception as e:
            self.log_message(f"Error graph: {e}\n")


    def display_graph(self, graph_function, params=None):
        """Call the graph function and display the output in QLabel."""
        try:
            # Call the graph function with parameters if provided
            if params:
                graph_function(**params)  # Unpack params to pass title and decades
            else:
                graph_function()

            
        except Exception as e:
            self.log_message(f"Error displaying graph: {e}\n")

    def display_stats(self, stats_text):
        """Display the stats in QTextArea."""
        self.graphInformation_2.setPlainText(stats_text)  # Set the stats text
    def load_graph_list(self):
        """Populate the QTreeWidget with the list of graphs."""
        self.stackedWidget_2.setCurrentIndex(1)  # Switch to the graph display view
        self.graphs_list.clear()  # Clear the tree widget before populating

        # Parent nodes for "Graphs" and "Turbine Graphs"
        graph_parent = QTreeWidgetItem(self.graphs_list, ["Graphs"])
        turbine_parent = QTreeWidgetItem(self.graphs_list, ["Turbine Graphs"])

        # Add main graphs to the "Graphs" parent
        for graph in self.graphs:
            # Check if the graph has 'Caudal' or 'Nivel' in the title
            if "Caudal" in graph["title"]:
                caudal_parent = self._find_or_create_sub_parent(graph_parent, "Caudal Graphs")
                graph_item = QTreeWidgetItem(caudal_parent, [graph["title"]])
            elif "Nivel" in graph["title"]:
                nivel_parent = self._find_or_create_sub_parent(graph_parent, "Nivel Graphs")
                graph_item = QTreeWidgetItem(nivel_parent, [graph["title"]])
            else:
                # If not categorized, add directly under the parent
                graph_item = QTreeWidgetItem(graph_parent, [graph["title"]])

            # If the graph has associated decades, add them as subchildren
            if graph.get("decades"):
                for decade in graph["decades"]:
                    QTreeWidgetItem(graph_item, [f"Decade {decade}"])
        
        # Add turbine graphs under the "Turbine Graphs" parent
        for tgraph in self.turbine_graphs:
            turbine_item = QTreeWidgetItem(turbine_parent, [tgraph["title"]])
            # If specific turbine options exist, add them as subchildren
            if tgraph.get("turbine_options"):
                for turbine in tgraph["turbine_options"]:
                    QTreeWidgetItem(turbine_item, [turbine])

    def _find_or_create_sub_parent(self, parent, sub_title):
        """Helper function to find or create a sub-parent node."""
        # Iterate through existing children to check if the sub-parent already exists
        for i in range(parent.childCount()):
            if parent.child(i).text(0) == sub_title:
                return parent.child(i)
        # If not found, create a new sub-parent
        return QTreeWidgetItem(parent, [sub_title])


    


    def change_page(self):
        
        # Switch to the stack widget page with index 1
        self.stackedWidget_2.setCurrentIndex(0)
    def log_message(self, message):
        """Helper function to add messages to the QTextEdit log area."""
        self.graphInformation.append(message)
    
    def clear_logs(self):
        """Function to clear the log area."""
        self.graphInformation.clear()
        

    def log_message_2(self, message):
        """Helper function to add messages to the QTextEdit log area."""
        self.textEditLogs.append(message)
    
    def clear_logs_2(self):
        """Function to clear the log area."""
        self.textEditLogs.clear()
    def load_data(self):
        """Function that loads the data files when btnCarga is clicked."""
        # Clear logs for the current step
        self.clear_logs()
        self.log_message("<b><span style='color:blue;'>Starting data loading...</span></b>")

        # First dialog box for selecting a file
        first_file = self.select_file()

        if first_file:
            # Extract the filename to check the prefix
            first_filename = os.path.basename(first_file)

            # Check if the first file starts with 'Q'
            if first_filename.startswith("Q"):
                self.caudal_file = first_file
                self.log_message(f"<span style='color:green;'>First file (Q) selected: {first_filename}</span>")

                # Prompt user for second file with prefix "NV"
                second_file = self.select_file("NV")
            else:
                self.log_message("<span style='color:red;'>Error: The first file must start with 'Q'. Please try again.</span>")
                return  # Exit if the first file does not have 'Q' prefix
            
            # Check and assign the second file if selected correctly
            if second_file:
                second_filename = os.path.basename(second_file)
                
                # Check if second file starts with 'NV'
                if second_filename.startswith("NV"):
                    self.nivel_file = second_file
                    self.log_message(f"<span style='color:green;'>Second file (NV) selected: {second_filename}</span>")
                    self.log_message_2("<b>Please click 'Pretratamiento de datos' to continue.</b>\n")

                else:
                    self.log_message("<span style='color:red;'>Error: The second file must start with 'NV'. Please try again.</span>")
                    return  # Exit if the second file does not have 'NV' prefix
                
                # Load data from files
                try:
                    self.caudal_data = pd.read_csv(self.caudal_file, delimiter='|', decimal='.')
                    self.nivel_data = pd.read_csv(self.nivel_file, delimiter='|', decimal='.')
                    
                    # Convert 'Fecha' to datetime format for both caudal and nivel datasets
                    self.caudal_data["Fecha"] = pd.to_datetime(self.caudal_data["Fecha"], errors='coerce')
                    self.nivel_data["Fecha"] = pd.to_datetime(self.nivel_data["Fecha"], errors='coerce')
                    self.log_message("<span style='color:green;'>Date conversion successful.</span>")
                    
                    # Ensure 'Valor' column exists and filter valid dates
                    self.caudal_data = self.caudal_data.dropna(subset=["Fecha", "Valor"])
                    self.nivel_data = self.nivel_data.dropna(subset=["Fecha", "Valor"])
                    
                    # Call this in your 'Procesamiento' (Processing) stage
                    self.show_graph()


                    # Enable the next button after both files are selected
                    self.btnPreta.setEnabled(True)
                    self.btnCarga.setDisabled(True)  # Disable btnCarga to prevent re-clicking
                except Exception as e:
                    self.log_message(f"<span style='color:red;'>Error loading data: {e}</span>")
            else:
                self.log_message("<span style='color:red;'>Error: Second file not selected properly. Please try again.</span>")
        else:
            self.log_message("<span style='color:red;'>Error: First file not selected properly. Please try again.</span>")

    def select_file(self, required_prefix=None):
        """Function for file selection (using PyQt's QFileDialog)"""
        # Open file dialog to select a file
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "Data files (*.data)")
        
        if file_path:  # Check if a file was selected
            file_name = os.path.basename(file_path)  # Get the file name without path
            
            # If required_prefix is given, check if the file has the correct prefix
            if required_prefix and not file_name.startswith(required_prefix):
                self.log_message(f"<span style='color:red;'>Error: The selected file must start with '{required_prefix}'. Please select a correct file.</span>")
                return None  # Return None if the file does not match the expected prefix
            
            # Ensure the file has not been selected already for both 'Q' and 'NV'
            if (required_prefix == "Q" and file_path == self.nivel_file) or (required_prefix == "NV" and file_path == self.caudal_file):
                self.log_message("<span style='color:red;'>Error: You cannot select the same file for both 'Q' and 'NV'. Please choose a different file.</span>")
                return None  # Return None to indicate an error

            return file_path  # Return the selected file path if it matches the prefix
        else:
            self.log_message("<span style='color:orange;'>No file selected. Please try again.</span>")
            return None  # Return None if no file was selected

    def preprocess_data(self):
        """Function for the 'Pretratamiento' step, handling data completeness checks and interpolation."""
        self.clear_logs()  # Clear previous logs for clarity
        self.log_message("<b><span style='color:blue;'>Starting data preprocessing...</span></b>")

        # Load data from files
        self.caudal_data = pd.read_csv(self.caudal_file, delimiter='|', decimal='.')
        self.nivel_data = pd.read_csv(self.nivel_file, delimiter='|', decimal='.')

        # Convert 'Fecha' to datetime format
        try:
            self.caudal_data["Fecha"] = pd.to_datetime(self.caudal_data["Fecha"], errors='coerce')
            self.nivel_data["Fecha"] = pd.to_datetime(self.nivel_data["Fecha"], errors='coerce')
            self.log_message("<span style='color:green;'>Date conversion successful.</span>")
        except Exception as e:
            self.log_message(f"<span style='color:red;'>Error in date conversion: {e}</span>")
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
            self.log_message(f"<br><hr><b>{label} Data Completeness by Year:</b><hr>")
            for _, row in yearly_data.iterrows():
                year, records, missing_pct = row["Year"], row["Records"], row["Missing %"]
                if missing_pct > 20:
                    color = 'red'
                    status = "Marked for exclusion."
                else:
                    color = 'green'
                    status = "Data is usable."
                self.log_message(f"Year {year}: <span style='color:{color};'>{missing_pct:.2f}% missing</span> - {status}")
        
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
            self.log_message(f"<br><hr><b>{label} Data Statistics by Decade:</b><hr>")
            self.log_message(decade_stats.to_string())

            # Display statistics for consecutive NaNs after interpolation
            dataset.set_index("Fecha", inplace=True)
            dataset["NaN_count_post"] = dataset["Valor"].isna().astype(int).groupby(dataset["Valor"].notna().astype(int).cumsum()).cumsum()
            consecutive_nans_post = dataset.groupby(dataset.index.year)["NaN_count_post"].max()
            self.log_message(f"<br><b>{label} Consecutive NaNs per Year After Interpolation:</b><br>")
            self.log_message(consecutive_nans_post.to_string())

            # Log completion of processing
            self.log_message(f"<span style='color:green;'>{label} data preprocessing completed with decade-wise summary.</span>")
        self.clear_logs_2()
        self.log_message_2("<b>Please click 'Tratamiento de datos' to continue.</b>\n")

        # Enable the next button after preprocessing
        self.btnTrata.setEnabled(True)
        self.btnPreta.setDisabled(True)

    def generate_decade_stats(self, dataset, label):
        """Generate decade-wise statistical summary."""
        # Add a 'Decade' column for grouping
        dataset['Decade'] = dataset['Fecha'].dt.year // 10 * 10
        
        # Filter out 0 and NaN values
        dataset_filtered = dataset[(dataset['Valor'] != 0) & (~dataset['Valor'].isna())]
        
        # Group by Decade and provide descriptive statistics
        decade_stats = dataset_filtered.groupby('Decade')['Valor'].describe()
        
        # Add the decade statistics to graphInformation
        self.graphInformation.append(f"<hr><b>{label} Data - Decade-wise Statistics:</b><hr>")
        self.graphInformation.append(f"<b>Descriptive Statistics by Decade:</b><br>")
        self.graphInformation.append(decade_stats.to_html())  # Add statistics in HTML format to QTextEdit

    def save_plot_as_image(self):
        """Save the latest plot as an image file to display in graphInformation."""
        # Save the current plot to an image file (in memory or disk)
        image_path = '/path/to/save/plot.png'
        plt.savefig(image_path)
        return image_path  # Return the file path to display in the QTextEdit

    def plot_data_in_label(self, data, ylabel, title, adjust_units=False):
        """Plot the graph and show it in the QLabel with statistics in QTextEdit."""        
        # Adjust units for 'Nivel' data if needed
        if adjust_units:
            data = data.copy()  # Copy to avoid modifying original data
            data["Valor"] = data["Valor"] / 100
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data['Fecha'], data['Valor'])
        ax.set_xlabel('Fecha')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        temp_filename = "temp_plot.png"
        fig.savefig(temp_filename)

        # Convert the saved image to a QPixmap and display it
        qpixmap = QPixmap(temp_filename)
        self.lblGraph.setPixmap(qpixmap)

        # Delete the temporary file
        os.remove(temp_filename)   
        
        # Display statistics in QTextEdit
        stats = data.describe()
        self.graphInformation.clear()
        self.graphInformation.append(f"{ylabel} Data Statistics:\n{stats.to_string()}")
        
        # Close the figure to free memory
        plt.close(fig)

    def convert_canvas_to_qpixmap(self, canvas):
        """Convert the matplotlib canvas to a QPixmap image."""
        buf = BytesIO()
        canvas.print_png(buf)
        buf.seek(0)
        img = Image.open(buf)
        return QPixmap.fromImage(img.toqimage())

    def show_graph(self):
        """Initialize graph display to show only two graphs sequentially."""
        self.current_graph = 0  # Start with Caudal graph
        self.graph_timer = QTimer()
        self.graph_timer.timeout.connect(self.display_next_graph)
        self.display_next_graph()  # Display the first graph immediately
        self.graph_timer.start(5000)  # After 5 seconds, show the next graph and stop

    def display_next_graph(self):
        """Display each graph sequentially and then show the Nivel heatmap after the second graph."""
        if self.current_graph == 0:
            # Display Caudal line graph
            self.plot_data_in_label(
                self.caudal_data,
                ylabel='Caudal (m3/s)',
                title='Gráfico de Línea de Valor en el Tiempo de Caudal (datos sin tratamiento)'
            )
            self.current_graph = 1  # Move to the next graph
        elif self.current_graph == 1:
            # Display Nivel line graph
            self.plot_data_in_label(
                self.nivel_data,
                ylabel='Nivel (m)',
                title='Gráfico de Línea de Valor en el Tiempo de Nivel (datos sin tratamiento)',
                adjust_units=True
            )
            self.current_graph = 2  # Move to the heatmap
            self.graph_timer.start(5000)  # Wait 5 seconds before the next graph
        elif self.current_graph == 2:
            # Display the heatmap for missing values in Nivel data
        
            self.plot_heatmap(self.caudal_data, 'Caudal')
            self.generate_decade_stats(self.caudal_data, 'Caudal')

            self.current_graph = 3  # Move to the heatmap

        elif self.current_graph == 3:
            # Display the heatmap for missing values in Nivel data
            self.plot_heatmap(self.nivel_data, 'Nivel')
            self.generate_decade_stats(self.nivel_data, 'Nivel')

            self.graph_timer.stop()  # Stop the timer once heatmap is shown

    def plot_heatmap(self, dataset, label):
        """Plot a heatmap for missing values in the dataset and display it in lblGraph."""
        dataset['Fecha'] = pd.to_datetime(dataset['Fecha'], errors='coerce')
        dataset['Year'] = dataset['Fecha'].dt.year
        dataset['DayOfYear'] = dataset['Fecha'].dt.dayofyear
        dataset_grouped = dataset.groupby(['Year', 'DayOfYear'])['Valor'].mean().reset_index()
        dataset_pivot = dataset_grouped.pivot(index='Year', columns='DayOfYear', values='Valor')
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(dataset_pivot.isnull(), cmap=sns.color_palette(["#add8e6", "#000000"]), cbar=False)
        plt.title(f'Visualización de Valores Faltantes en el DataFrame de {label}')
        plt.xlabel('Día del Año')
        plt.ylabel('Año')
        
        # Save the figure to a temporary file
        temp_filename = "temp_heatmap.png"
        plt.savefig(temp_filename)

        # Convert the saved image to a QPixmap and display it
        qpixmap = QPixmap(temp_filename)
        self.lblGraph.setPixmap(qpixmap)

        # Delete the temporary file
        os.remove(temp_filename)
        
        description = dataset_grouped['Valor'].describe()
        
        # Add information to graphInformation
        self.graphInformation.append(f"<b>{label} Data - Heatmap of Missing Values:</b><br>")
        self.graphInformation.append(f"Visualizes missing data for each day of the year across years.<br>")
        self.graphInformation.append(f"<b>Descriptive Statistics of {label} Data:</b><br>")
        self.graphInformation.append(f"Count: {description['count']}, Mean: {description['mean']}, Std: {description['std']}, Min: {description['min']}, Max: {description['max']}<br>")
        

    def treat_data(self):
        """Handle the Tratamiento button click to switch to page 1, filter available years, check data completeness, and fill missing values."""
        
        # Switch to the stack widget page with index 1
        self.stackedWidget.setCurrentIndex(1)
        
        # Initialize text to show usable years
        usable_years_text = ""
        self.clear_logs_2()
        self.log_message_2("<span style='color:blue;'>\n Click on Confirm to Contine. \n</span>")


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
                    usable_years_text += f"{label} Data Usable Years: {', '.join(map(str, sorted(usable_years)))}\n\n"
                else:
                    usable_years_text += f"{label} Data Usable Years: None\n"


            # Update the availableYears QTextEdit with usable years
            self.availableYears.setPlainText(usable_years_text)
             
            
       
        else:
            # Error if data not loaded
            self.log_message("Error: Either Caudal or Nivel data is not loaded. Please load both datasets first.")

    def confirm_data(self):
        """Checks txtWaterTra and txtWaterDensity, saves values if valid."""
        # Retrieve text from txtWaterTra and txtWaterDensity fields
        water_tra = self.txtWaterTra.text().strip()
        water_density = self.txtWaterDensity.text().strip()
        turbine_graph=self.cmbxTurbineGraph.currentText()

        # Check if either field is empty
        if not water_tra or not water_density:
            self.log_message("<span style='color:red;'>Please fill in both Water Transverse Length in Meters and Water Density fields.</span>")
            return

        # Try to convert values to ensure they are numbers
        try:
            water_tra = float(water_tra)
            water_density = float(water_density)
        except ValueError:
            self.log_message("<span style='color:red;'>Invalid input. Please enter numeric values for both fields.</span>")
            return

        # Save the values (you can customize where these values should be saved)
        self.water_tra_value = water_tra
        self.water_density_value = water_density
        self.turbine_option=turbine_graph
        print("Water Transverse Length in Meters:", self.water_tra_value)
        print("Water Density in kg/m3:", self.water_density_value)
        print("Turbine Graph:", self.turbine_option)
        self.log_message("<span style='color:green;'>Water Transverse Length in Meters, Water Density in kg/m3 values and Turbine Graph are selected successfully.</span>")
        self.log_message_2("<br><b>Please click 'Resultados' to continue.</b><br>")

        # Enable the next button or trigger the next action if needed
        self.btnResult.setEnabled(True)
        self.btnProce.setDisabled(True)
        self.stackedWidget.setCurrentIndex(0)
        self.btnResult.setEnabled(True)
        self.btnProce.setDisabled(True)  # Disable btnProce to prevent re-clicking



    def confirm_years(self):
        """Check entered years for dry, wet, and normal, and validate against available years for both Caudal and Nivel, considering missing data."""  
        if self.caudal_data is None or self.nivel_data is None:
            self.log_message("Caudal or Nivel data is not available for validation.")
            return
        
        # Get usable years for Caudal dataset (less than 20% missing data)
        usable_years_caudal = self.get_usable_years(self.caudal_data)
        
        # Get usable years for Nivel dataset (less than 20% missing data)
        usable_years_nivel = self.get_usable_years(self.nivel_data)
        
        # Get the intersection of usable years from both datasets
        usable_years = usable_years_caudal | usable_years_nivel
        
        # Retrieve and validate entered years from line edits
        try:
            self.clear_logs()
            # Read years from text fields and convert them to sets of integers
            dry_years = set(map(int, self.txtDryYears.text().split(',')))
            wet_years = set(map(int, self.txtWetYears.text().split(',')))
            normal_years = set(map(int, self.txtNormalYears.text().split(',')))
            
            # Check if all entered years are in usable years
            invalid_years = (dry_years | wet_years | normal_years) - usable_years
            if invalid_years:
                self.log_message(f"Invalid years entered: {', '.join(map(str, invalid_years))}. Please only enter usable years.")
            else:
                self.log_message("Dry, Wet, and Normal Years confirmed successfully and saved.")
                
                self.log_message_2("<br><b>Please click 'Procesamiento' to continue.</b><br>")
                # After processing, enable the next button
                self.btnProce.setEnabled(True)
                self.btnTrata.setDisabled(True)  # Disable btnProce to prevent re-clicking
                self.stackedWidget.setCurrentIndex(0)
                # Save valid years or proceed with next logic as needed
                self.dry_years = dry_years
                self.wet_years = wet_years
                self.normal_years = normal_years
                print("Dry years:", self.dry_years)
                print("Wet years:", self.wet_years)
                print("Normal years:", self.normal_years)
        
        except ValueError:
            self.log_message("Error: Please enter only numeric years, separated by commas.")
            
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

    def process_data(self):
        # Implement the logic for processing (e.g., generating metrics)
        self.stackedWidget.setCurrentIndex(2)
        self.clear_logs()
        self.clear_logs_2()

        self.log_message_2("<span style='color:blue;'>\n Click on Confirm to Contine. \n</span>")


    



    def show_results(self):
        """Display each decade graph for caudal, nivel, and yearly graphs with 5-second intervals."""
        self.clear_logs()
        self.clear_logs_2()
        self.log_message_2("Showing Results (Each graph will change every 5 seconds)\n")
        self.btnResult.setDisabled(True)  # Disable btnProce to prevent re-clicking
        # self.showMaximized()

        # Define the graphs to cycle through
        # Define the graphs to cycle through
        





        # Initialize variables for cycling through graphs and decades
        self.graph_index = 0
        self.decade_index = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_graph_display)
        
        # Start displaying the first graph
        self.update_graph_display()
        self.timer.start(5000)  # Change graph every 5 seconds

    def update_graph_display(self):
        """Update the graph display in lblGraph for each decade and show stats in graphInformation."""
        # Stop the timer if all graphs and decades have been displayed
        if self.graph_index >= len(self.graphs):
            self.timer.stop()
            if self.turbine_option=='All':
                self.calculate_and_display_turbine_power()
            else:
                self.calculate_and_display_turbine_power(self.turbine_option)
            return

        # Retrieve the current graph info
        graph_info = self.graphs[self.graph_index]
        graph_function = graph_info["function"]
        title = graph_info["title"]
        decades = graph_info["decades"]

        # Plot the graph for the current decade or yearly data
        if decades:
            current_decade = decades[self.decade_index]
            graph_function(title, current_decade)
            self.decade_index += 1
            if self.decade_index >= len(decades):
                self.decade_index = 0
                self.graph_index += 1
        else:
            self.graph_index += 1
            graph_function(title)

    def plot_decade_scatter_caudal_graph(self, title='Decadal Caudal Series', decades=[],flag=False):
        """Plot scatter graph of caudal data for a specific decade in lblGraph."""
        try:
            if flag==True:
                decades=decades[0]
            caudal_process = self.caudal_process.sort_values(by='Fecha')
            decade_end = decades + 9
            decade_data = caudal_process[(caudal_process['Fecha'].dt.year >= decades) & (caudal_process['Fecha'].dt.year <= decade_end)]

            if decade_data.empty:
                self.log_message(f"No data available for the {decades}s decade.")
                return

            # Create scatter plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(decade_data['Fecha'], decade_data['Valor'], label=f'Década {decades}s', alpha=0.5)
            ax.set_title(f'{title} for {decades}s')
            ax.set_xlabel('Date')
            ax.set_ylabel('Mean Flow (m³/s)')
            ax.legend()

            # Format x-axis
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            plt.xticks(rotation=45)
            ax.grid(True)
            plt.tight_layout()
            print("Caudal")
            temp_filename = "temp_plot.png"
            fig.savefig(temp_filename)

            # Convert the saved image to a QPixmap and display it
            qpixmap = QPixmap(temp_filename)
            

            # Display statistics in graphInformation
            stats = decade_data['Valor'].describe()
            if flag==True:
                self.lblGraph_2.setPixmap(qpixmap)
                self.graphInformation_2.clear()
                self.graphInformation_2.append(f"{title} ({decades}s)\nStatistics:\n{stats.to_string()}")
            else:
                self.lblGraph.setPixmap(qpixmap)
                self.graphInformation.clear()
                self.graphInformation.append(f"{title} ({decades}s)\nStatistics:\n{stats.to_string()}")
                self.log_message(f"Displayed {title} for {decades}s\n")
            # Delete the temporary file
            os.remove(temp_filename) 


            plt.close(fig)
        except Exception as e:
            self.log_message(f"<span style='color:red;'>Error in plot_decade_scatter_caudal_graph: {e}</span>")

    def plot_decade_scatter_nivel_graph(self, title='Decadal Nivel Series', decades=[],flag=False):
        """Plot scatter graph of nivel data for a specific decade in lblGraph."""
        try:
            if flag==True:
                decades=decades[0]
            df_copycaudal = self.caudal_process
            df_copycaudal = df_copycaudal.rename(columns={'Valor': 'Caudal'})
            df_copynivel = self.nivel_process
            df_copynivel = self.nivel_process.rename(columns={'Valor': 'Nivel'})
            self.df_merge = pd.merge(df_copycaudal, df_copynivel, on='Fecha', how='outer')
            nivel_process = self.nivel_process.sort_values(by='Fecha')
            decade_end = decades + 9
            decade_data = nivel_process[(nivel_process['Fecha'].dt.year >= decades) & (nivel_process['Fecha'].dt.year <= decade_end)]

            if decade_data.empty:
                self.log_message(f"No data available for the {decades}s decade.")
                return

            # Create scatter plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(decade_data['Fecha'], decade_data['Valor'], label=f'Década {decades}s', alpha=0.5)
            ax.set_title(f'{title} for {decades}s')
            ax.set_xlabel('Date')
            ax.set_ylabel('Mean Level (m)')
            ax.legend()

            # Format x-axis
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            plt.xticks(rotation=45)
            ax.grid(True)
            plt.tight_layout()
            if flag==True:
                print("Flag True")
            temp_filename = "temp_plot.png"
            fig.savefig(temp_filename)

            # Convert the saved image to a QPixmap and display it
            qpixmap = QPixmap(temp_filename)
            
            # Display statistics in graphInformation
            stats = decade_data['Valor'].describe()
            if flag==True:
                self.lblGraph_2.setPixmap(qpixmap)
                self.graphInformation_2.clear()
                self.graphInformation_2.append(f"{title} ({decades}s)\nStatistics:\n{stats.to_string()}")
            else:
                self.lblGraph.setPixmap(qpixmap)
                self.graphInformation.clear()
                self.graphInformation.append(f"{title} ({decades}s)\nStatistics:\n{stats.to_string()}")
                self.log_message(f"Displayed {title} for {decades}s\n")
            # Delete the temporary file
            os.remove(temp_filename) 


            plt.close(fig)
        except Exception as e:
            self.log_message(f"<span style='color:red;'>Error in plot_decade_scatter_nivel_graph: {e}</span>")

    def plot_yearly_scatter_caudal_graph(self, title='',decades=[],flag=False):
        """Plot scatter graph of yearly caudal data for all years."""
        try:
            
            # Sort caudal data by date
            caudal_process = self.caudal_process.sort_values(by='Fecha')
            
            # Get unique years from the data
            years = caudal_process['Fecha'].dt.year.unique()
            
            # Create the plot
            fig, axs = plt.subplots(nrows=(len(years) + 1) // 2, ncols=2, figsize=(10, (len(years) // 2 + 1) * 3))
            
            # Flatten the axes for easy iteration if there's more than one row
            axs = axs.flatten()

            # Iterate over the years and create scatter plots in subplots
            for i, year in enumerate(years):
                year_data = caudal_process[caudal_process['Fecha'].dt.year == year]
                
                if not year_data.empty:
                    ax = axs[i]  # Get the axis corresponding to the current year
                    ax.scatter(year_data['Fecha'], year_data['Valor'], label=f'Año {year}', alpha=0.5)
                    ax.set_title(f'Año {year}')
                    ax.set_xlabel('Fecha')
                    ax.set_ylabel('Valor en m³/s')

                    # Format the x-axis to show months
                    ax.xaxis.set_major_locator(mdates.MonthLocator())
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # Show abbreviated month names
                    ax.tick_params(axis='x', rotation=45)
                    ax.grid(True)

            # Remove any empty subplots if the number of years is odd
            if len(years) % 2 != 0:
                fig.delaxes(axs[-1])

            # Adjust layout for better spacing
            plt.tight_layout()

            # Resize the plot to match QLabel dimensions
            label_width = self.lblGraph.width()
            label_height = self.lblGraph.height()
            fig.set_size_inches(label_width / 100, label_height / 100)  # Convert QLabel dimensions to inches
            canvas = FigureCanvas(fig)
            canvas.draw()

            # Convert the plot to QPixmap and display in QLabel
            temp_filename = "temp_plot.png"
            fig.savefig(temp_filename)

            # Convert the saved image to a QPixmap and display it
            qpixmap = QPixmap(temp_filename)
            
            # Display statistics in graphInformation
            stats = caudal_process['Valor'].describe()
            # Display statistics in graphInformation
            if flag==True:
                self.lblGraph_2.setPixmap(qpixmap)
                self.graphInformation_2.clear()
                self.graphInformation_2.append(f"{title} ({decades}s)\nStatistics:\n{stats.to_string()}")
            else:
                self.lblGraph.setPixmap(qpixmap)
                self.graphInformation.clear()
                self.graphInformation.append(f"{title} ({decades}s)\nStatistics:\n{stats.to_string()}")
                self.log_message(f"Displayed {title} for {decades}s\n")
            # Delete the temporary file
            os.remove(temp_filename) 


            plt.close(fig)
        except Exception as e:
            self.log_message(f"<span style='color:red;'>Error in plot_yearly_scatter_graph: {e}</span>")

    def plot_yearly_scatter_nivel_graph(self, title='',decades=[],flag=False):
        """Plot scatter graph of yearly nivel data for all years."""
        try:
          
            # Sort nivel data by date
            nivel_process = self.nivel_process.sort_values(by='Fecha')
            
            # Get unique years from the data
            years = nivel_process['Fecha'].dt.year.unique()
            
            # Create the plot
            fig, axs = plt.subplots(nrows=(len(years) + 1) // 2, ncols=2, figsize=(10, (len(years) // 2 + 1) * 3))
            
            # Flatten the axes for easy iteration if there's more than one row
            axs = axs.flatten()

            # Iterate over the years and create scatter plots in subplots
            for i, year in enumerate(years):
                year_data = nivel_process[nivel_process['Fecha'].dt.year == year]
                
                if not year_data.empty:
                    ax = axs[i]  # Get the axis corresponding to the current year
                    ax.scatter(year_data['Fecha'], year_data['Valor'], label=f'Año {year}', alpha=0.5)
                    ax.set_title(f'Año {year}')
                    ax.set_xlabel('Fecha')
                    ax.set_ylabel('Valor en cm')

                    # Format the x-axis to show months
                    ax.xaxis.set_major_locator(mdates.MonthLocator())
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # Show abbreviated month names
                    ax.tick_params(axis='x', rotation=45)
                    ax.grid(True)

            # Remove any empty subplots if the number of years is odd
            if len(years) % 2 != 0:
                fig.delaxes(axs[-1])

            # Adjust layout for better spacing
            plt.tight_layout()

           # Resize the plot to match QLabel dimensions
            label_width = self.lblGraph.width()
            label_height = self.lblGraph.height()
            fig.set_size_inches(label_width / 100, label_height / 100)  # Convert QLabel dimensions to inches
            canvas = FigureCanvas(fig)
            canvas.draw()

            # Convert the plot to QPixmap and display in QLabel
            temp_filename = "temp_plot.png"
            fig.savefig(temp_filename)

            # Convert the saved image to a QPixmap and display it
            qpixmap = QPixmap(temp_filename)
            

            # Display statistics in graphInformation
            stats = nivel_process['Valor'].describe()
            # Display statistics in graphInformation
            if flag==True:
                self.lblGraph_2.setPixmap(qpixmap)
                self.graphInformation_2.clear()
                self.graphInformation_2.append(f"{title} ({decades}s)\nStatistics:\n{stats.to_string()}")
            else:
                self.lblGraph.setPixmap(qpixmap)
                self.graphInformation.clear()
                self.graphInformation.append(f"{title} ({decades}s)\nStatistics:\n{stats.to_string()}")
                self.log_message(f"Displayed {title} for {decades}s\n")
            # Delete the temporary file
            os.remove(temp_filename) 


            plt.close(fig)
        except Exception as e:
            self.log_message(f"<span style='color:red;'>Error in plot_yearly_scatter_nivel_graph: {e}</span>")
    def show_statistics(self, title="Statistics",decades=[],flag=False):
        """
        Calculate and display statistics for caudal and nivel data in the provided QTextEdit widget.
        :param info_textedit: QTextEdit where statistics information will be displayed.
        """
        # self.showMaximized()
        # List of years to remove
        years_to_remove = [1969, 1985, 1987, 1991, 2001, 2002, 2003, 2004]

        # Filter data for caudal and nivel excluding specified years
        caudal_sel = self.caudal_process[~self.caudal_process['Fecha'].dt.year.isin(years_to_remove)]
        nivel_sel = self.nivel_process[~self.nivel_process['Fecha'].dt.year.isin(years_to_remove)]

        # Calculate statistics for caudal
        caudal_mean = caudal_sel['Valor'].mean()
        caudal_std = caudal_sel['Valor'].std()
        caudal_range = caudal_sel['Valor'].max() - caudal_sel['Valor'].min()
        caudal_cv = (caudal_std / caudal_mean) * 100

        # Calculate statistics for nivel
        nivel_mean = nivel_sel['Valor'].mean()
        nivel_std = nivel_sel['Valor'].std()
        nivel_range = nivel_sel['Valor'].max() - nivel_sel['Valor'].min()
        nivel_cv = (nivel_std / nivel_mean) * 100

        # Format the statistics for display
        caudal_stats = (
            f"Caudal Statistics:\n"
            f"- Mean: {caudal_mean:.2f}\n"
            f"- Standard Deviation: {caudal_std:.2f}\n"
            f"- Range: {caudal_range:.2f}\n"
            f"- Coefficient of Variation: {caudal_cv:.2f}%\n\n"
        )

        nivel_stats = (
            f"Nivel Statistics:\n"
            f"- Mean: {nivel_mean:.2f}\n"
            f"- Standard Deviation: {nivel_std:.2f}\n"
            f"- Range: {nivel_range:.2f}\n"
            f"- Coefficient of Variation: {nivel_cv:.2f}%\n"
        )

        # Display the statistics in the QTextEdit widget
        # Display statistics in graphInformation
        if flag==True:
            self.lblGraph.clear()
            self.graphInformation_2.clear()
            self.graphInformation_2.append(caudal_stats + nivel_stats)
        else:
            self.lblGraph.clear()
            self.graphInformation.clear()
            self.graphInformation.append(caudal_stats + nivel_stats)
            self.log_message(f"Displayed {title} for {decades}s\n")
            # Delete the temporary file


            # Log the action
            self.log_message("Displayed overall statistics for Caudal and Nivel.\n")

    def plot_distribution_caudal_graph(self, title='',decades=[],flag=False):
        """Plot histogram and KDE of caudal data with normal distribution overlay in lblGraph and show skewness & kurtosis in graphInformation."""
        try:
            if flag==True:
                print("Flag True")
            # Filter out NaN values
            caudal_data = self.caudal_process['Valor'].dropna()

            # Calculate skewness and kurtosis
            skewness = skew(caudal_data)
            kurtosis_value = kurtosis(caudal_data)

            # Create histogram and KDE plot
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(caudal_data, kde=True, bins=30, color='skyblue', stat="density", linewidth=0, ax=ax)

            # Calculate and plot the normal distribution overlay
            mu, std = caudal_data.mean(), caudal_data.std()
            xmin, xmax = ax.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            ax.plot(x, p, 'r', linewidth=2, label=f'Normal Dist.\nμ={mu:.2f}, σ={std:.2f}')

            # Set titles, labels, and legend
            ax.set_title(title)
            ax.set_xlabel('Flow Rate (Caudal)')
            ax.set_ylabel('Density')
            ax.legend()

            # Convert plot to QPixmap and display in lblGraph
            temp_filename = "temp_plot.png"
            fig.savefig(temp_filename)

            # Convert the saved image to a QPixmap and display it
            qpixmap = QPixmap(temp_filename)
            if flag==True:
                self.lblGraph_2.setPixmap(qpixmap)
                self.graphInformation_2.clear()
                self.graphInformation_2.append(
                f"{title}\n"
                f"Skewness (Asimetría): {skewness:.2f}\n"
                f"Kurtosis (Curtosis): {kurtosis_value:.2f}\n"
                f"Note: Skewness measures asymmetry; Kurtosis indicates tail heaviness."
            )
            else:
                self.lblGraph.setPixmap(qpixmap)
                self.graphInformation.clear()
                self.graphInformation.append(
                f"{title}\n"
                f"Skewness (Asimetría): {skewness:.2f}\n"
                f"Kurtosis (Curtosis): {kurtosis_value:.2f}\n"
                f"Note: Skewness measures asymmetry; Kurtosis indicates tail heaviness."
            )
            # Delete the temporary file            

                self.log_message(f"Displayed {title}\n")
            os.remove(temp_filename) 

        except Exception as e:
            # Log error
            self.log_message(f"<span style='color:red;'>Error in plot_distribution_caudal_graph: {e}</span>")
    def plot_distribution_nivel_graph(self, title='',decades=[],flag=False):
        """Plot histogram and KDE of nivel data with normal distribution overlay in lblGraph and show skewness & kurtosis in graphInformation."""
        try:
            # Filter out NaN values
            nivel_data = self.nivel_process['Valor'].dropna()

            # Calculate skewness and kurtosis
            skewness = skew(nivel_data)
            kurtosis_value = kurtosis(nivel_data)

            # Create histogram and KDE plot
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(nivel_data, kde=True, bins=30, color='skyblue', stat="density", linewidth=0, ax=ax)

            # Calculate and plot the normal distribution overlay
            mu, std = nivel_data.mean(), nivel_data.std()
            xmin, xmax = ax.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            ax.plot(x, p, 'r', linewidth=2, label=f'Normal Dist.\nμ={mu:.2f}, σ={std:.2f}')

            # Set titles, labels, and legend
            ax.set_title(title)
            ax.set_xlabel('Level (Nivel)')
            ax.set_ylabel('Density')
            ax.legend()

            # Convert plot to QPixmap and display in lblGraph
            temp_filename = "temp_plot.png"
            fig.savefig(temp_filename)

            # Convert the saved image to a QPixmap and display it
            qpixmap = QPixmap(temp_filename)
            if flag==True:
                self.lblGraph_2.setPixmap(qpixmap)
                self.graphInformation_2.clear()
                self.graphInformation_2.append(
                f"{title}\n"
                f"Skewness (Asimetría): {skewness:.2f}\n"
                f"Kurtosis (Curtosis): {kurtosis_value:.2f}\n"
                f"Note: Skewness measures asymmetry; Kurtosis indicates tail heaviness."
            )
            else:
                self.lblGraph.setPixmap(qpixmap)
                self.graphInformation.clear()
                self.graphInformation.append(
                f"{title}\n"
                f"Skewness (Asimetría): {skewness:.2f}\n"
                f"Kurtosis (Curtosis): {kurtosis_value:.2f}\n"
                f"Note: Skewness measures asymmetry; Kurtosis indicates tail heaviness."
            )
                self.log_message(f"Displayed {title}\n")

            # Delete the temporary file
            os.remove(temp_filename) 

            # Close the figure
            plt.close(fig)

            
            # Log success

        except Exception as e:
            # Log error
            self.log_message(f"<span style='color:red;'>Error in plot_distribution_nivel_graph: {e}</span>")
    def plot_decadal_density_caudal_graph(self, title='', decades=[],flag=False):
        """Plot probability density graphs for caudal data for each decade in lblGraph."""
        try:
            if flag==False:
                decades=[decades]
            # Loop through each decade in the list of decades
            for decade_start in decades:
                decade_end = decade_start + 9
                decade_data = self.caudal_process[
                    (self.caudal_process['Fecha'].dt.year >= decade_start) & 
                    (self.caudal_process['Fecha'].dt.year <= decade_end)
                ]

                # Check if there's data for the decade
                if decade_data.empty:
                    self.log_message(f"No data available for the {decade_start}s decade.")
                    continue

                # Create KDE plot for probability density
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.kdeplot(data=decade_data['Valor'], fill=True, label=f'{decade_start}s', alpha=0.5, ax=ax)
                ax.set_title(f'{title} - {decade_start}s')
                ax.set_xlabel("Average Flow (Caudal) m³/s")
                ax.set_ylabel("Probability Density - Caudal")
                ax.legend()

                # Convert plot to QPixmap and display in lblGraph
                temp_filename = "temp_plot.png"
                fig.savefig(temp_filename)

                # Convert the saved image to a QPixmap and display it
                qpixmap = QPixmap(temp_filename)
                stats = decade_data['Valor'].describe()
                if flag==True:
                    self.lblGraph_2.setPixmap(qpixmap)
                    self.graphInformation_2.clear()
                    self.graphInformation_2.append(f"{title} ({decade_start}s)\nStatistics:\n{stats.to_string()}")
                else:
                    self.lblGraph.setPixmap(qpixmap)
                    self.graphInformation.clear()
                    self.graphInformation.append(f"{title} ({decade_start}s)\nStatistics:\n{stats.to_string()}")
                    self.log_message(f"Displayed {title} for {decade_start}s\n")

                # Delete the temporary file
                os.remove(temp_filename) 

                # Close the plot to free memory
                plt.close(fig)
                    
        except Exception as e:
            self.log_message(f"<span style='color:red;'>Error in plot_decadal_density_caudal_graph: {e}</span>")

    def plot_decadal_density_nivel_graph(self, title, decades=[],flag=False):
        """Plot probability density graphs for nivel data for each decade in lblGraph."""
        try:
            if flag==False:
                decades=[decades]
            for decade_start in decades:
                decade_end = decade_start + 9
                decade_data = self.nivel_process[(self.nivel_process['Fecha'].dt.year >= decade_start) & 
                                                (self.nivel_process['Fecha'].dt.year <= decade_end)]

                if decade_data.empty:
                    self.log_message(f"No data available for the {decade_start}s decade.")
                    continue

                # Create KDE plot for probability density
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.kdeplot(data=decade_data['Valor'], fill=True, label=f'{decade_start}s', alpha=0.5, ax=ax)
                ax.set_title(f'{title} - {decade_start}s')
                ax.set_xlabel("Average Level (Nivel) m")
                ax.set_ylabel("Probability Density - Nivel")
                ax.legend()

                # Convert plot to QPixmap and display in lblGraph
                temp_filename = "temp_plot.png"
                fig.savefig(temp_filename)

                # Convert the saved image to a QPixmap and display it
                qpixmap = QPixmap(temp_filename)
                if flag==True:
                    self.lblGraph_2.setPixmap(qpixmap)
                    self.graphInformation_2.clear()
                    self.graphInformation_2.append(f"{title} ({decade_start}s)\nProbability Density by Decade")
                else:
                    self.lblGraph.setPixmap(qpixmap)
                    self.graphInformation.clear()
                    self.graphInformation.append(f"{title} ({decade_start}s)\nProbability Density by Decade")
                    self.log_message(f"Displayed {title} for {decade_start}s\n")
                   
                # Delete the temporary file
                os.remove(temp_filename) 

                
                plt.close(fig)
                
        except Exception as e:
            self.log_message(f"<span style='color:red;'>Error in plot_decadal_density_nivel_graph: {e}</span>")


    def plot_annual_behavior_decadal_caudal_graph(self, title, decades=[],flag=False):
        """Plot annual flow behavior for each decade in lblGraph."""
        try:
            if flag==False:
                decades=[decades]
            # Cycle through colors and line styles for visual differentiation
            colors = itertools.cycle(['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'brown'])
            linestyles = itertools.cycle(['-', '--', '-.', ':'])

            # Create a new column for "DiaMes" (Day and Month) for plotting purposes
            self.caudal_process['DiaMes'] = self.caudal_process['Fecha'].apply(lambda x: x.replace(year=2000))

            # Loop through each specified decade and plot the data for each year within that decade
            for decade_start in decades:
                decade_end = decade_start + 9
                decade_data = self.caudal_process[
                    (self.caudal_process['Fecha'].dt.year >= decade_start) & 
                    (self.caudal_process['Fecha'].dt.year <= decade_end)
                ]

                if decade_data.empty:
                    self.log_message(f"No data available for the {decade_start}s decade.")
                    continue

                # Create a figure for each decade
                fig, ax = plt.subplots(figsize=(10, 6))

                # Plot data for each year within the decade
                for year in range(decade_start, decade_end + 1):
                    year_data = decade_data[decade_data['Fecha'].dt.year == year]
                    if not year_data.empty:
                        ax.plot(
                            year_data['DiaMes'],
                            year_data['Valor'],
                            label=str(year),
                            color=next(colors),
                            linestyle=next(linestyles),
                            linewidth=2,
                            marker='o',
                            markersize=4,
                            alpha=0.7
                        )

                # Title, labels, and legend
                ax.set_title(f'{title} - {decade_start}s')
                ax.set_xlabel('Day and Month')
                ax.set_ylabel('Flow (m³/s)')
                ax.legend(title='Year', loc='upper left', bbox_to_anchor=(1, 1))
                ax.grid(True)

                # Format the X-axis to show only the day and month
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
                plt.xticks(rotation=45)
                plt.tight_layout()

                # Convert plot to QPixmap and display in lblGraph
                temp_filename = "temp_plot.png"
                fig.savefig(temp_filename)

                # Convert the saved image to a QPixmap and display it
                qpixmap = QPixmap(temp_filename)
                if flag==True:
                    self.lblGraph_2.setPixmap(qpixmap)
                    self.graphInformation_2.clear()
                    self.graphInformation_2.append(f"{title} ({decade_start}s)\nProbability Density by Decade")
                else:
                    self.lblGraph.setPixmap(qpixmap)
                    self.graphInformation.clear()
                    self.graphInformation.append(f"{title} ({decade_start}s)\nProbability Density by Decade")
                    self.log_message(f"Displayed {title} for {decade_start}s\n")
                # Delete the temporary file
                os.remove(temp_filename) 

                # Log the displayed graph title

                # Close the figure to free memory
                plt.close(fig)
                
        except Exception as e:
            self.log_message(f"<span style='color:red;'>Error in plot_annual_behavior_decadal_caudal_graph: {e}</span>")
    def plot_annual_behavior_decadal_nivel_graph(self, title, decades=[],flag=False):
        """Plot annual behavior of nivel data by decade in lblGraph."""
        try:
            if flag==True:
                decades=decades[0]
            # Filter data for the specified decade
            nivel_process = self.nivel_process.sort_values(by='Fecha')
            decade_end = decades + 9
            decade_data = nivel_process[(nivel_process['Fecha'].dt.year >= decades) & (nivel_process['Fecha'].dt.year <= decade_end)]

            if decade_data.empty:
                self.log_message(f"No data available for the {decades}s decade.\n")
                return

            # Create DiaMes column to represent day and month
            decade_data['DiaMes'] = decade_data['Fecha'].apply(lambda x: x.replace(year=2000))

            fig, ax = plt.subplots(figsize=(10, 6))
            colors = itertools.cycle(['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'brown'])
            linestyles = itertools.cycle(['-', '--', '-.', ':'])

            # Plot data for each year in the decade
            for year in range(decades, decade_end + 1):
                year_data = decade_data[decade_data['Fecha'].dt.year == year]
                if not year_data.empty:
                    ax.plot(
                        year_data['DiaMes'], 
                        year_data['Valor'], 
                        label=str(year),
                        color=next(colors),
                        linestyle=next(linestyles),
                        linewidth=2,
                        marker='o',
                        markersize=4,
                        alpha=0.7
                    )

            ax.set_title(f'{title} - {decades}s')
            ax.set_xlabel('Day and Month')
            ax.set_ylabel('Level (m)')
            ax.legend(title='Year', loc='upper left', bbox_to_anchor=(1, 1))
            ax.grid(True)
            plt.tight_layout()

            # Format the X-axis to show only day and month
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b-%d'))
            plt.xticks(rotation=45)

            # Convert plot to QPixmap and display in lblGraph
            temp_filename = "temp_plot.png"
            fig.savefig(temp_filename)

            # Convert the saved image to a QPixmap and display it
            qpixmap = QPixmap(temp_filename)
            if flag==True:
                self.lblGraph_2.setPixmap(qpixmap)
                self.graphInformation_2.clear()
                self.graphInformation_2.append(f"{title} ({decades}s)\nAnnual Level Behavior")
            else:
                self.lblGraph.setPixmap(qpixmap)
                self.graphInformation.clear()
                self.graphInformation.append(f"{title} ({decades}s)\nAnnual Level Behavior")
                self.log_message(f"Displayed {title} for {decades}s\n")
        # Delete the temporary file
            os.remove(temp_filename) 

           
            plt.close(fig)
        except Exception as e:
            self.log_message(f"<span style='color:red;'>Error in plot_annual_behavior_decadal_nivel_graph: {e}</span>")
    def plot_hydrological_profile_caudal(self, title="Perfil Hidrológico Anual - Caudal",decades=[],flag=False):
        """Plot annual hydrological profile by month for dry, wet, and normal caudal cases in lblGraph."""
        try:
            # Ensure 'Fecha' is in datetime format
            self.caudal_process['Fecha'] = pd.to_datetime(self.caudal_process['Fecha'], errors='coerce')
            
            # Add 'Mes' and 'Year' columns
            self.caudal_process['Mes'] = self.caudal_process['Fecha'].dt.month
            self.caudal_process['Year'] = self.caudal_process['Fecha'].dt.year

            # Define a function to calculate the monthly average
            def get_monthly_avg(year_list):
                df_filtered = self.caudal_process[self.caudal_process['Year'].isin(year_list)]
                return df_filtered.groupby('Mes')['Valor'].mean()

            # Calculate monthly averages for each category
            monthly_avg_wet = get_monthly_avg(self.wet_years)
            monthly_avg_normal = get_monthly_avg(self.normal_years)
            monthly_avg_dry = get_monthly_avg(self.dry_years)

            # Set up the figure and plot each category as a bar chart
            fig, axs = plt.subplots(3, 1, figsize=(10, 6))
            y_limits = (0, 400)
            y_ticks = range(0, 401, 50)
            month_labels = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']

            # Plot each hydrological profile
            for ax, data, color, title_text in zip(
                axs,
                [monthly_avg_wet, monthly_avg_normal, monthly_avg_dry],
                ['b', 'g', 'r'],
                ['Caso Húmedo', 'Caso Base', 'Caso Seco']
            ):
                bars = ax.bar(data.index, data.values, color=color)
                ax.set_title(f'{title} - {title_text}', fontsize=14)
                ax.set_ylabel('Caudal Promedio (m³/s)')
                ax.set_ylim(y_limits)
                ax.set_yticks(y_ticks)
                ax.set_xticks(range(1, 13))
                ax.set_xticklabels(month_labels)
                for bar in bars:
                    yval = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, round(yval, 2), ha='center', va='bottom')

            axs[-1].set_xlabel('Mes')
            plt.tight_layout()

            # Convert plot to QPixmap and display in lblGraph
            temp_filename = "temp_plot.png"
            fig.savefig(temp_filename)

            # Convert the saved image to a QPixmap and display it
            qpixmap = QPixmap(temp_filename)
            if flag==True:
                self.lblGraph_2.setPixmap(qpixmap)
                self.graphInformation_2.clear()
                self.graphInformation_2.append(f"{title}\nMonthly Hydrological Profile for Wet, Normal, and Dry Cases")
            else:
                self.lblGraph.setPixmap(qpixmap)
                self.graphInformation.clear()
                self.graphInformation.append(f"{title}\nMonthly Hydrological Profile for Wet, Normal, and Dry Cases")
                self.log_message(f"Displayed {title}s\n")
            # Delete the temporary file
            os.remove(temp_filename) 
            plt.close(fig)
        except Exception as e:
            self.log_message(f"<span style='color:red;'>Error in plot_hydrological_profile_caudal: {e}</span>")
    def plot_hydrological_profile_nivel(self, title='',decades=[],flag=False):
        """Plot the annual hydrological profile (Nivel) by month for different categories (Húmedo, Base, Seco)."""

        try:
            # Ensure 'Fecha' is in datetime format and add columns for 'Mes' and 'Year'
            self.nivel_process['Fecha'] = pd.to_datetime(self.nivel_process['Fecha'], errors='coerce')
            self.nivel_process['Mes'] = self.nivel_process['Fecha'].dt.month
            self.nivel_process['Year'] = self.nivel_process['Fecha'].dt.year

            # Define function to get monthly averages for a specific list of years
            def get_monthly_avg(nivel_sel, year_list):
                df_casosNV = nivel_sel[nivel_sel['Year'].isin(year_list)]
                return df_casosNV.groupby('Mes')['Valor'].mean()

            # Calculate monthly averages for each case
            monthly_avg_wet = get_monthly_avg(self.nivel_process, self.wet_years)
            monthly_avg_normal = get_monthly_avg(self.nivel_process, self.normal_years)
            monthly_avg_dry = get_monthly_avg(self.nivel_process, self.dry_years)

            # Plot settings
            fig, axes = plt.subplots(3, 1, figsize=(10, 6))
            y_limitsNV = (0, 4)
            y_ticksNV = [i * 0.5 for i in range(9)]
            categories = [
                {"avg": monthly_avg_wet, "color": 'b', "title": "Caso Húmedo", "ax": axes[0]},
                {"avg": monthly_avg_normal, "color": 'g', "title": "Caso Base", "ax": axes[1]},
                {"avg": monthly_avg_dry, "color": 'r', "title": "Caso Seco", "ax": axes[2]},
            ]

            # Plot bars for each category
            for category in categories:
                ax = category["ax"]
                bars = ax.bar(category["avg"].index, category["avg"].values, color=category["color"])
                ax.set_title(f'Perfil Hidrológico Anual - {category["title"]} NIVEL', fontsize=14)
                ax.set_ylabel('Nivel Promedio (m)')
                ax.set_ylim(y_limitsNV)
                ax.set_yticks(y_ticksNV)
                ax.set_xticks(range(1, 13))
                ax.set_xticklabels(['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'])
                for bar in bars:
                    yval = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, round(yval, 2), ha='center', va='bottom')

            axes[2].set_xlabel('Mes')
            plt.tight_layout()

            # Convert plot to QPixmap and display in lblGraph
            temp_filename = "temp_plot.png"
            fig.savefig(temp_filename)

            # Convert the saved image to a QPixmap and display it
            qpixmap = QPixmap(temp_filename)
            if flag==True:
                self.lblGraph_2.setPixmap(qpixmap)
                self.graphInformation_2.clear()
                self.graphInformation_2.append(f"Displayed {title}\n")
            else:
                self.lblGraph.setPixmap(qpixmap)
                self.graphInformation.clear()
                self.graphInformation.append(f"Displayed {title}\n")
                self.log_message(f"Displayed {title}s\n")
            # Delete the temporary file
            os.remove(temp_filename) 

  

            plt.close(fig)

        except Exception as e:
            self.log_message(f"<span style='color:red;'>Error in plot_hydrological_profile_nivel: {e}</span>")
    def plot_annual_profile_days_caudal(self, title,decades=[],flag=False):
        """Plot the annual hydrological profile by day of year (Caudal) for different categories (Húmedo, Base, Seco)."""

        try:
            # Ensure 'Fecha' is in datetime format and add columns for 'Year' and 'Day_of_Year'
            self.caudal_process['Fecha'] = pd.to_datetime(self.caudal_process['Fecha'], errors='coerce')
            self.caudal_process['Year'] = self.caudal_process['Fecha'].dt.year
            self.caudal_process['Day_of_Year'] = self.caudal_process['Fecha'].dt.dayofyear

            # Define function to get daily averages for a specific list of years
            def get_daily_avg(caudal_sel, year_list):
                df_casosQ = caudal_sel[caudal_sel['Year'].isin(year_list)]
                return df_casosQ.groupby('Day_of_Year')['Valor'].mean()

            # Calculate daily averages for each case
            daily_avg_wet = get_daily_avg(self.caudal_process, self.wet_years)
            daily_avg_normal = get_daily_avg(self.caudal_process, self.normal_years)
            daily_avg_dry = get_daily_avg(self.caudal_process, self.dry_years)

            # Plot settings
            fig, axes = plt.subplots(3, 1, figsize=(10, 6))
            y_limitsQ = (0, 500)
            y_ticksQ = range(0, 501, 50)
            categories = [
                {"avg": daily_avg_wet, "color": 'b', "title": "Caso Húmedo", "ax": axes[0]},
                {"avg": daily_avg_normal, "color": 'g', "title": "Caso Base", "ax": axes[1]},
                {"avg": daily_avg_dry, "color": 'r', "title": "Caso Seco", "ax": axes[2]},
            ]

            # Plot bars for each category
            for category in categories:
                ax = category["ax"]
                ax.bar(category["avg"].index, category["avg"].values, color=category["color"])
                ax.set_title(f'Perfil Hidrológico Anual CAUDAL - {category["title"]}', fontsize=14)
                ax.set_ylabel('Caudal Promedio (m³/s)')
                ax.set_xlabel('Día del Año')
                ax.set_ylim(y_limitsQ)
                ax.set_yticks(y_ticksQ)

            plt.tight_layout()

            # Convert plot to QPixmap and display in lblGraph
            temp_filename = "temp_plot.png"
            fig.savefig(temp_filename)

            # Convert the saved image to a QPixmap and display it
            qpixmap = QPixmap(temp_filename)
            if flag==True:
                self.lblGraph_2.setPixmap(qpixmap)
                self.graphInformation_2.clear()
                self.graphInformation_2.append(f"Displayed {title}\n")
            else:
                self.lblGraph.setPixmap(qpixmap)
                self.graphInformation.clear()
                self.graphInformation.append(f"Displayed {title}\n")
                self.log_message(f"Displayed {title}s\n")

            # Delete the temporary file
            os.remove(temp_filename) 


            plt.close(fig)

        except Exception as e:
            self.log_message(f"<span style='color:red;'>Error in plot_annual_profile_days_caudal: {e}</span>")
    def plot_annual_profile_days_nivel(self, title,decades=[],flag=False):
        """Plot the annual hydrological profile by day of year (Nivel) for different categories (Húmedo, Base, Seco)."""

        try:
            # Ensure 'Fecha' is in datetime format and add columns for 'Year' and 'Day_of_Year'
            self.nivel_process['Fecha'] = pd.to_datetime(self.nivel_process['Fecha'], errors='coerce')
            self.nivel_process['Year'] = self.nivel_process['Fecha'].dt.year
            self.nivel_process['Day_of_Year'] = self.nivel_process['Fecha'].dt.dayofyear

            # Define function to get daily averages for a specific list of years
            def get_daily_avg(nivel_sel, year_list):
                df_casosNV = nivel_sel[nivel_sel['Year'].isin(year_list)]
                return df_casosNV.groupby('Day_of_Year')['Valor'].mean()

            # Calculate daily averages for each case
            daily_avg_wet = get_daily_avg(self.nivel_process, self.wet_years)
            daily_avg_normal = get_daily_avg(self.nivel_process, self.normal_years)
            daily_avg_dry = get_daily_avg(self.nivel_process, self.dry_years)

            # Plot settings
            fig, axes = plt.subplots(3, 1, figsize=(10, 6))
            y_limitsNV = (0, 5)
            y_ticksNV = [i * 0.5 for i in range(11)]
            categories = [
                {"avg": daily_avg_wet, "color": 'b', "title": "Caso Húmedo", "ax": axes[0]},
                {"avg": daily_avg_normal, "color": 'g', "title": "Caso Base", "ax": axes[1]},
                {"avg": daily_avg_dry, "color": 'r', "title": "Caso Seco", "ax": axes[2]},
            ]

            # Plot bars for each category
            for category in categories:
                ax = category["ax"]
                ax.bar(category["avg"].index, category["avg"].values, color=category["color"])
                ax.set_title(f'Perfil Hidrológico Anual NIVEL - {category["title"]}', fontsize=14)
                ax.set_ylabel('Nivel Promedio (m)')
                ax.set_xlabel('Día del Año')
                ax.set_ylim(y_limitsNV)
                ax.set_yticks(y_ticksNV)

            plt.tight_layout()

            # Convert plot to QPixmap and display in lblGraph
            temp_filename = "temp_plot.png"
            fig.savefig(temp_filename)

            # Convert the saved image to a QPixmap and display it
            qpixmap = QPixmap(temp_filename)
            if flag==True:
                self.lblGraph_2.setPixmap(qpixmap)
                self.graphInformation_2.clear()
                self.graphInformation_2.append(f"Displayed {title}\n")
            else:
                self.lblGraph.setPixmap(qpixmap)
                self.graphInformation.clear()
                self.graphInformation.append(f"Displayed {title}\n")
                self.log_message(f"Displayed {title}s\n")
            # Delete the temporary file
            os.remove(temp_filename) 
            # Log the displayed graph title

            plt.close(fig)

        except Exception as e:
            self.log_message(f"<span style='color:red;'>Error in plot_annual_profile_days_nivel: {e}</span>")
    def show_nominal_stats(self,title,decades=[],flag=False):
        """Calculate and display nominal flow and level statistics (max, mean, min) for each hydrological category."""

        try:
            # Ensure 'Fecha' is in datetime format and add 'Year' column
            self.caudal_process['Fecha'] = pd.to_datetime(self.caudal_process['Fecha'], errors='coerce')
            self.caudal_process['Year'] = self.caudal_process['Fecha'].dt.year
            self.nivel_process['Fecha'] = pd.to_datetime(self.nivel_process['Fecha'], errors='coerce')
            self.nivel_process['Year'] = self.nivel_process['Fecha'].dt.year

            # Function to calculate max, mean, min for flow (caudal) or level (nivel)
            def calculate_stats(df, year_list):
                filtered_df = df[df['Year'].isin(year_list)]
                max_val = filtered_df['Valor'].max()
                mean_val = filtered_df['Valor'].mean()
                min_val = filtered_df['Valor'].min()
                return max_val, mean_val, min_val

            # Calculate flow (caudal) stats for each category
            caudal_stats_h2 = calculate_stats(self.caudal_process, self.wet_years)
            caudal_stats_b2 = calculate_stats(self.caudal_process, self.normal_years)
            caudal_stats_s2 = calculate_stats(self.caudal_process, self.dry_years)

            # Calculate level (nivel) stats for each category
            nivel_stats_h2 = calculate_stats(self.nivel_process, self.wet_years)
            nivel_stats_b2 = calculate_stats(self.nivel_process, self.normal_years)
            nivel_stats_s2 = calculate_stats(self.nivel_process, self.dry_years)

            # Format the stats for display
            stats_text = (
                f"**Caudal (Flow) Statistics**\n"
                f"Húmedo (Wet) - Max: {caudal_stats_h2[0]:.2f}, Mean: {caudal_stats_h2[1]:.2f}, Min: {caudal_stats_h2[2]:.2f}\n"
                f"Base (Normal) - Max: {caudal_stats_b2[0]:.2f}, Mean: {caudal_stats_b2[1]:.2f}, Min: {caudal_stats_b2[2]:.2f}\n"
                f"Seco (Dry) - Max: {caudal_stats_s2[0]:.2f}, Mean: {caudal_stats_s2[1]:.2f}, Min: {caudal_stats_s2[2]:.2f}\n\n"
                f"**Nivel (Level) Statistics**\n"
                f"Húmedo (Wet) - Max: {nivel_stats_h2[0]:.2f}, Mean: {nivel_stats_h2[1]:.2f}, Min: {nivel_stats_h2[2]:.2f}\n"
                f"Base (Normal) - Max: {nivel_stats_b2[0]:.2f}, Mean: {nivel_stats_b2[1]:.2f}, Min: {nivel_stats_b2[2]:.2f}\n"
                f"Seco (Dry) - Max: {nivel_stats_s2[0]:.2f}, Mean: {nivel_stats_s2[1]:.2f}, Min: {nivel_stats_s2[2]:.2f}"
            )
            if flag==True:
                self.lblGraph_2.clear()
                self.graphInformation_2.clear()
                self.graphInformation_2.append(stats_text)
            else:
                self.lblGraph.clear()
                self.graphInformation.clear()
                self.graphInformation.setText(stats_text)
                self.log_message("Displayed nominal flow and level statistics\n")
            
        except Exception as e:
            self.log_message(f"<span style='color:red;'>Error in show_nominal_stats: {e}</span>")


    def calculate_P95_and_display(self,title,decades=[],flag=False):
        """Calculate and display the P95 exceedance probability and its corresponding graph."""
        
        try:
            # List of years from the 'caudal_sel' DataFrame
            anos_listaQ = self.caudal_process['Fecha'].dt.year.tolist()

            # Initialize lists and dictionaries
            caudales_P95 = []  # List for P95 values
            Caudales = self.caudal_process['Valor'].tolist()
            Fechas_compQ = self.caudal_process['Fecha'].tolist()

            # Convert dates to datetime objects if they are not already
            Fechas_compQ = [pd.to_datetime(fecha) for fecha in Fechas_compQ]

            # Dictionary to group flows by year
            caudales_por_ano = {ano: [] for ano in anos_listaQ}

            # Group flows by year
            for fecha, caudal in zip(Fechas_compQ, Caudales):
                ano = fecha.year
                if ano in caudales_por_ano:
                    caudales_por_ano[ano].append(caudal)

            # Calculate P95 for each year
            for ano in anos_listaQ:
                if caudales_por_ano[ano]:  # If there is data for the year
                    # Sort the flows for the current year
                    caudales_ordenados = sorted(caudales_por_ano[ano])

                    # Calculate the 95th percentile (P95)
                    P95 = np.percentile(caudales_ordenados, 95)
                    caudales_P95.append(P95)
                else:
                    caudales_P95.append(None)  # No data for the year

            # Create a DataFrame for the results
            P95_dfQ = pd.DataFrame({
                'Año': anos_listaQ,
                'Caudal_P95': caudales_P95})

            # Print the DataFrame for verification
            print(P95_dfQ.head())

            # Plot the results
            plt.figure(figsize=(8, 6))
            plt.plot(P95_dfQ['Año'], P95_dfQ['Caudal_P95'], label='Caudal (m³/s)', color='blueviolet')
            plt.xlabel('Año')
            plt.ylabel('Caudal (m³/s)')
            plt.title('Caudal (Q95) vs Décadas')
            plt.legend()
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)

             # Convert plot to QPixmap and display in lblGraph
            canvas = FigureCanvas(plt.gcf())
            canvas.draw()

            # Convert canvas to QImage
            width, height = canvas.get_width_height()
            img = QImage(canvas.buffer_rgba(), width, height, QImage.Format_RGBA8888)

            # Display the relevant information in graphInformation
            max_P95 = np.max(caudales_P95) if caudales_P95 else 'No Data'
            min_P95 = np.min(caudales_P95) if caudales_P95 else 'No Data'
            mean_P95 = np.mean(caudales_P95) if caudales_P95 else 'No Data'

            # Set the statistics text in graphInformation
            graph_info_text = f"Max P95: {max_P95:.2f}\nMin P95: {min_P95:.2f}\nMean P95: {mean_P95:.2f}"
            if flag==True:
                self.lblGraph_2.setPixmap(QPixmap(img))
                self.graphInformation_2.clear()
                self.graphInformation_2.setText(graph_info_text)
            else:
                self.lblGraph.setPixmap(QPixmap(img))
                self.graphInformation.clear()
                self.graphInformation.setText(graph_info_text)
                self.log_message(f"Displayed {title}s\n")
            # Delete the temporary file
            # Log the displayed graph title

        except Exception as e:
            self.log_message(f"<span style='color:red;'>Error in calculate_P95_and_display: {e}</span>")


    def display_average_flow(self,title,decades=[],flag=False):
        """Calculate and display the average flow (caudal promedio) over the years."""
        
        try:
            # **CAUDALES PROMEDIOS**
            # Convert 'Fecha' to datetime and set it as the index
            caudal_sel = self.caudal_process  # Assuming self.caudal_process is your data
            caudal_sel['Fecha'] = pd.to_datetime(caudal_sel['Fecha'])
            caudal_sel = caudal_sel.set_index('Fecha')

            # Resample with annual frequency and calculate the mean of flow rate
            promedios_anuales = caudal_sel['Valor'].resample('A').mean()

            # Convert years and annual averages into lists
            lista_anosQ = promedios_anuales.index.year.tolist()  # List of years
            lista_promedios_anualesQ = promedios_anuales.tolist()  # List of annual averages

            # Debugging prints
            print("Años:", lista_anosQ)
            print("Promedios anuales:", lista_promedios_anualesQ)

            # Create the average flow graph
            plt.figure(figsize=(8, 6))
            plt.plot(lista_anosQ, lista_promedios_anualesQ, label='Caudal m3/s', color='blueviolet')
            plt.xlabel('Año')
            plt.ylabel('Caudal (m³/s)')
            plt.title('Caudal Promedio vs Años')
            plt.legend()
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)

            # Convert plot to QPixmap and display in lblGraph
            canvas = FigureCanvas(plt.gcf())
            canvas.draw()

            # Convert canvas to QImage
            width, height = canvas.get_width_height()
            img = QImage(canvas.buffer_rgba(), width, height, QImage.Format_RGBA8888)


            # Set text with flow statistics in graphInformation
            max_flow = np.max(lista_promedios_anualesQ)
            min_flow = np.min(lista_promedios_anualesQ)
            mean_flow = np.mean(lista_promedios_anualesQ)

            flow_info_text = f"Max Caudal: {max_flow:.2f} m³/s\nMin Caudal: {min_flow:.2f} m³/s\nMean Caudal: {mean_flow:.2f} m³/s"
            if flag==True:
                self.lblGraph_2.setPixmap(QPixmap(img))
                self.graphInformation_2.clear()
                self.graphInformation_2.setText(flow_info_text)
            else:
                self.lblGraph.setPixmap(QPixmap(img))
                self.graphInformation.clear()
                self.graphInformation.setText(flow_info_text)
                self.log_message(f"Displayed {title}s\n")
            # Delete the temporary file
            # Log message

        except Exception as e:
            self.log_message(f"<span style='color:red;'>Error in display_average_flow: {e}</span>")
    def display_level_p95(self,title,decades=[],flag=False):
        """Calculate and display the P95 levels for each year."""
        
        try:
            # **Nivel P95**
            # List of years from the different categories (H2, B2, S2)
            # Ensure both files and processed data are available
            if not self.dry_years or not self.wet_years or not self.normal_years :
                QMessageBox.warning(self, "Missing Years", "Please select Dry,Wet and Normal years Before.")
                return
            total_years = self.dry_years | self.wet_years | self.normal_years  # Using the union operator
            total_years = sorted(total_years)
            print("Total Years:", len(total_years))

            # Extract years from the 'Fecha' column
            anos_listaNV = self.nivel_process['Fecha'].dt.year.tolist()

            # Initialize lists for P95 values
            nivel_P95 = []
            niveles = self.nivel_process['Valor'].tolist()  # List of level values
            Fechas_compNV = self.nivel_process['Fecha'].tolist()

            # Convert dates to datetime if not already
            Fechas_compNV = [pd.to_datetime(fecha) for fecha in Fechas_compNV]

            # Group levels by year
            niveles_por_ano = {ano: [] for ano in anos_listaNV}
            for fecha, lvl in zip(Fechas_compNV, niveles):
                ano = fecha.year
                if ano in niveles_por_ano:
                    niveles_por_ano[ano].append(lvl)

            # Calculate P95 for each year
            for ano in anos_listaNV:
                if niveles_por_ano[ano]:  # If there are data for the year
                    niveles_ordenados = sorted(niveles_por_ano[ano])
                    P95 = np.percentile(niveles_ordenados, 95)
                    nivel_P95.append(P95)
                else:
                    nivel_P95.append(None)  # No data for the year

            # Create the DataFrame for P95 levels
            NivelP95_df = pd.DataFrame({
                'Año': anos_listaNV,
                'Nivel_P95': nivel_P95
            })

            # Filter DataFrame using the list of total years
            NivelP95_df['Año'] = pd.to_datetime(NivelP95_df['Año'], errors='coerce')
            NivelP95_df_fil = NivelP95_df[NivelP95_df['Año'].dt.year.isin(total_years)]

            # Show the DataFrame for debugging
            print(NivelP95_df.head())

            # Plotting the P95 level graph
            plt.figure(figsize=(8, 6))
            plt.plot(NivelP95_df['Año'], NivelP95_df['Nivel_P95'], label='Nivel cm', color='blueviolet')
            plt.xlabel('Año')
            plt.ylabel('Nivel (cm)')
            plt.title('Nivel P95 vs Décadas')
            plt.legend()
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)

            canvas = FigureCanvas(plt.gcf())
            canvas.draw()

            # Convert canvas to QImage
            width, height = canvas.get_width_height()
            img = QImage(canvas.buffer_rgba(), width, height, QImage.Format_RGBA8888)


            
            # Set text with level statistics in graphInformation
            max_level = np.nanmax(nivel_P95) if nivel_P95 else None
            min_level = np.nanmin(nivel_P95) if nivel_P95 else None
            mean_level = np.nanmean(nivel_P95) if nivel_P95 else None

            level_info_text = f"Max Nivel: {max_level:.2f} cm\nMin Nivel: {min_level:.2f} cm\nMean Nivel: {mean_level:.2f} cm"
            self.graphInformation.setText(level_info_text)
            if flag==True:
                self.lblGraph_2.setPixmap(QPixmap(img))
                self.graphInformation_2.clear()
                self.graphInformation_2.setText(level_info_text)
            else:
                self.lblGraph.setPixmap(QPixmap(img))
                self.graphInformation.clear()
                self.graphInformation.setText(level_info_text)
                self.log_message(f"Displayed {title}s\n")
            # Delete the temporary file
            # Log message
            self.log_message("Displayed Level P95 statistics and graph.\n")

        except Exception as e:
            self.log_message(f"<span style='color:red;'>Error in display_level_p95: {e}</span>")
    def display_flow_velocity(self,title,decades=[],flag=False):
        """Calculate and display the flow velocity (Velocidad) graph."""

        try:
            # **self.velocidad de flujo**
            
            nivel_cha_cha = self.df_merge['Nivel'].tolist()
            caudal_cha_cha = self.df_merge['Caudal'].tolist()
            fecha_cha_cha = self.df_merge['Fecha'].tolist()

            print('aja?')
            print(len(fecha_cha_cha))
            print(len(nivel_cha_cha))
            print(len(caudal_cha_cha))
            print('o no aja?')

            lvl = 0
            longitud = 142.73  # meters (for example)
            Lvlm = [lvl / 100 for lvl in nivel_cha_cha]

            # Calculate the area variable based on level
            Area_variable = [elemento * longitud if elemento >= 0.5 else 0 for elemento in Lvlm]

            # Calculate velocity
            self.velocidad = []
            ultimo_valor = None  # Variable to store the last valid velocity value

            for a, b in zip(caudal_cha_cha, Area_variable):
                try:
                    if pd.isna(b):
                        # Use the last valid velocity value if b is NaN
                        self.velocidad.append(ultimo_valor if ultimo_valor is not None else 0)
                    else:
                        # Calculate the velocity
                        valor_actual = a / b
                        self.velocidad.append(valor_actual)
                        ultimo_valor = valor_actual  # Update the last valid value
                except ZeroDivisionError:
                    # Assign zero in case of a division by zero
                    self.velocidad.append(0)

            # Add the calculated velocity as a new column to the DataFrame
            self.df_merge['Velocidad'] = self.velocidad

            # Plotting the velocity graph
            plt.figure(figsize=(8, 6))
            plt.plot(fecha_cha_cha, self.velocidad, label='Velocidad m/s', color='blueviolet')
            plt.xlabel('Año')
            plt.ylabel('Velocidad (m/s)')
            plt.title('Velocidad vs Décadas')
            plt.legend()
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)

            # Convert plot to QPixmap and display in lblGraph
            # Convert matplotlib figure to QPixmap for display in QLabel
            canvas = FigureCanvas(plt.gcf())
            canvas.draw()

            # Convert canvas to QImage
            width, height = canvas.get_width_height()
            img = QImage(canvas.buffer_rgba(), width, height, QImage.Format_RGBA8888)

         


            # Set text with velocity statistics in graphInformation
            max_velocity = np.nanmax(self.velocidad) if self.velocidad else None
            min_velocity = np.nanmin(self.velocidad) if self.velocidad else None
            mean_velocity = np.nanmean(self.velocidad) if self.velocidad else None

            velocity_info_text = f"Max Velocidad: {max_velocity:.2f} m/s\nMin Velocidad: {min_velocity:.2f} m/s\nMean Velocidad: {mean_velocity:.2f} m/s"

            if flag==True:
                self.lblGraph_2.setPixmap(QPixmap(img))
                self.graphInformation_2.clear()
                self.graphInformation_2.setText(velocity_info_text)
            else:
                self.lblGraph.setPixmap(QPixmap(img))
                self.graphInformation.clear()
                self.graphInformation.setText(velocity_info_text)
                self.log_message(f"Displayed {title}s\n")
            # Delete the temporary file

            # Log message

        except Exception as e:
            self.log_message(f"<span style='color:red;'>Error in display_flow_velocity: {e}</span>")
    def calculate_monthly_behavior(self,title,decades=[],flag=False):
        """Calculate and display the daily average velocity per month and category."""

        try:
            # Ensure that the 'Fecha' column is of datetime type
            self.df_merge['Fecha'] = pd.to_datetime(self.df_merge['Fecha'])

            # Prepare the data for 'Year', 'Month' and 'Day of Year'
            self.df_merge['Year'] = self.df_merge['Fecha'].dt.year
            self.df_merge['Day_of_Year'] = self.df_merge['Fecha'].dt.dayofyear
            self.df_merge['Mes'] = self.df_merge['Fecha'].dt.month

            # Use self.dry, self.wet, and self.normal for categories
            # These should be lists of years or boolean filters defining each category
            dry_years = self.dry_years  # Dry years (e.g., list of years or boolean condition)
            wet_years = self.wet_years   # Wet years
            normal_years = self.normal_years  # Normal years

            # Function to calculate the daily average velocity for each category
            def get_daily_avg_day(df, year_list):
                df_filtered = df[df['Year'].isin(year_list)]
                return df_filtered.groupby('Day_of_Year')['Velocidad'].mean()

            # Get the daily averages for each category using the class attributes (self.dry, self.wet, self.normal)
            df_merge_dry = get_daily_avg_day(self.df_merge, dry_years)   # Dry years
            df_merge_wet = get_daily_avg_day(self.df_merge, wet_years)   # Wet years
            df_merge_normal = get_daily_avg_day(self.df_merge, normal_years)  # Normal years

            # Plot the daily average velocity
            self.plot_monthly_velocity(df_merge_dry, df_merge_wet, df_merge_normal, 'Día del Año',flag)

        except Exception as e:
            self.log_message(f"<span style='color:red;'>Error in calculate_monthly_behavior: {e}</span>")

    def calculate_average_monthly_velocity(self,title,decades=[],flag=False):
        """Calculate and display the daily average velocity per month and category."""

        try:
            # Ensure that the 'Fecha' column is of datetime type
            self.df_merge['Fecha'] = pd.to_datetime(self.df_merge['Fecha'])

            # Prepare the data for 'Year', 'Month' and 'Day of Year'
            self.df_merge['Year'] = self.df_merge['Fecha'].dt.year
            self.df_merge['Day_of_Year'] = self.df_merge['Fecha'].dt.dayofyear
            self.df_merge['Mes'] = self.df_merge['Fecha'].dt.month

            # Use self.dry, self.wet, and self.normal for categories
            # These should be lists of years or boolean filters defining each category
            dry_years = self.dry_years  # Dry years (e.g., list of years or boolean condition)
            wet_years = self.wet_years   # Wet years
            normal_years = self.normal_years  # Normal years

            # Function to calculate the daily average velocity for each category
            def get_daily_avg_day(df, year_list):
                df_filtered = df[df['Year'].isin(year_list)]
                return df_filtered.groupby('Day_of_Year')['Velocidad'].mean()

            # Get the daily averages for each category using the class attributes (self.dry, self.wet, self.normal)
            df_merge_dry = get_daily_avg_day(self.df_merge, dry_years)   # Dry years
            df_merge_wet = get_daily_avg_day(self.df_merge, wet_years)   # Wet years
            df_merge_normal = get_daily_avg_day(self.df_merge, normal_years)  # Normal years

            # Plot the daily average velocity
            self.plot_monthly_velocity(df_merge_dry, df_merge_wet, df_merge_normal, 'Mes',flag)

        except Exception as e:
            self.log_message(f"<span style='color:red;'>Error in calculate_monthly_behavior: {e}</span>")
    def plot_monthly_velocity(self, df_merge_dry, df_merge_wet, df_merge_normal, caso,flag):
        """Plot the daily average velocity graph for each category."""

        try:
            # Create the figure for the bar graphs
            plt.figure(figsize=(10, 6))

            # Plot for dry years (self.dry)
            plt.subplot(3, 1, 1)
            bars_dry = plt.bar(df_merge_dry.index, df_merge_dry.values, color='b')
            plt.title('Perfil Hidrológico Anual Velocidad - Caso Seco', fontsize=12)
            plt.ylabel('Velocidad Promedio (m/s)')
            if caso == 'Mes':
                plt.xticks(range(1, 13), ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'])
            plt.xlabel(caso)

            # Add data labels for the bars
            if caso == 'Mes':
                for bar in bars_dry:
                    yval = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.3f}', 
                            ha='center', va='bottom', fontsize=8)

            # Plot for wet years (self.wet)
            plt.subplot(3, 1, 2)
            bars_wet = plt.bar(df_merge_wet.index, df_merge_wet.values, color='g')
            plt.title('Perfil Hidrológico Anual Velocidad - Caso Húmedo', fontsize=12)
            plt.ylabel('Velocidad Promedio (m/s)')
            if caso == 'Mes':
                plt.xticks(range(1, 13), ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'])
            plt.xlabel(caso)

            # Add data labels for the bars
            if caso == 'Mes':
                for bar in bars_wet:
                    yval = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.3f}', 
                            ha='center', va='bottom', fontsize=8)

            # Plot for normal years (self.normal)
            plt.subplot(3, 1, 3)
            bars_normal = plt.bar(df_merge_normal.index, df_merge_normal.values, color='r')
            plt.title('Perfil Hidrológico Anual Velocidad - Caso Normal', fontsize=12)
            plt.xlabel(caso)
            if caso == 'Mes':
                plt.xticks(range(1, 13), ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'])
            plt.ylabel('Velocidad Promedio (m/s)')

            # Add data labels for the bars
            if caso == 'Mes':
                for bar in bars_normal:
                    yval = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.3f}', 
                            ha='center', va='bottom', fontsize=8)

            # Adjust spacing between the plots
            plt.tight_layout(pad=2)

            # Convert matplotlib figure to QPixmap for display in QLabel
            canvas = FigureCanvas(plt.gcf())
            canvas.draw()

            # Convert canvas to QImage
            width, height = canvas.get_width_height()
            img = QImage(canvas.buffer_rgba(), width, height, QImage.Format_RGBA8888)

            # Display the image in the QLabel
            
            if flag==True:
                self.lblGraph_2.setPixmap(QPixmap(img))
                self.graphInformation_2.clear()
                self.graphInformation_2.setText("Displayed daily average velocity graph for categories.\n")
            else:
                self.lblGraph.setPixmap(QPixmap(img))
                self.graphInformation.clear()
                self.graphInformation.setText("Displayed daily average velocity graph for categories.\n")
                self.log_message("Displayed daily average velocity graph for categories.\n")
            # Delete the temporary file
            # Log success

        except Exception as e:
            self.log_message(f"<span style='color:red;'>Error in plot_monthly_velocity: {e}</span>")

    def calculate_and_display_turbine_power(self, turbine_options=None, title='', flag=False):
        """Calculate and display power output for the specified turbine models."""
        try:
            # Default turbine options if none are provided
            print(turbine_options, title, flag)
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

            # Print turbine names to ensure they are correctly passed
            print(f"Using turbine options: {turbine_options}")

            # Load turbine data
            file_name_datasheet = r'Datasheet_V2.csv'
            file_name_powercurve = r'PowerCurves.xlsx'
            df_datasheet = pd.read_csv(file_name_datasheet, delimiter=',')
            df_powercurve = pd.read_excel(file_name_powercurve)

            # Clean up the column names to remove leading/trailing spaces
            df_datasheet.columns = df_datasheet.columns.str.strip()

            # Debugging: Print out columns of the dataframe to verify they are correct
            print("Columns in the datasheet:", df_datasheet.columns)

            # Initialize turbines
            turbines = {}
            for turbine_name in turbine_options:
                # Debugging: Print turbine name being checked
                print(f"Checking turbine: {turbine_name}")

                # Check if the turbine name exists in the columns
                if turbine_name not in df_datasheet.columns:
                    error_message = f"Turbine '{turbine_name}' not found in data sheet columns."
                    print(f"Error: {error_message}")
                    if flag:
                        self.graphInformation_2.setText(
                            f"<span style='color:red;'>{error_message}</span>"
                        )
                    raise KeyError(error_message)

                # Extract the relevant turbine data from the datasheet
                data = df_datasheet[turbine_name].tolist()
                filteredpc = df_powercurve[df_powercurve['Type'] == turbine_name]

                turbines[turbine_name] = Turbina(
                    turbine_name, *data[:7], filteredpc
                )

            # Extract velocity data
            velocity = self.df_merge['Velocidad'].tolist()
            list_of_turb_power = []

            # Pre-compute power outputs for each turbine
            for turbine_name in turbine_options:
                turbine = turbines[turbine_name]
                power_output = turbine.PowerOut(velocity)
                list_of_turb_power.append(power_output)

            # If turbine_options is provided, display all turbine graphs immediately
            if turbine_options is not None:
                for t, turbine_name in enumerate(turbine_options):
                    self._update_turbine_plot(t, list_of_turb_power, turbine_options, flag)

            # If turbine_options is None, display turbine graphs with a delay of 5 seconds between each
            else:
                for t, turbine_name in enumerate(turbine_options):
                    QTimer.singleShot(
                        t * 5000,  # 5-second delay for each turbine
                        lambda t=t: self._update_turbine_plot(
                            t, list_of_turb_power, turbine_options
                        )
                    )
                self.log_message("Displaying all turbine power plots with delay.")

        except ValueError as ve:
            error_message = f"Value Error: {ve}"
            self.handle_error(error_message, flag)
        except KeyError as ke:
            error_message = f"Key Error: {ke}"
            self.handle_error(error_message, flag)
        except Exception as e:
            error_message = f"Unexpected Error: {e}"
            self.handle_error(error_message, flag)

    def handle_error(self, message, flag):
        """Handle errors by logging or showing in a specific component."""
        print(f"Error: {message}")
        if flag:
            self.graphInformation_2.setText(
                f"<span style='color:red;'>{message} Try to Process the data first please on other page</span>"
            )
        else:
            self.log_message(f"<span style='color:red;'>{message}</span>")


    def _update_turbine_plot(self, index, list_of_turb_power, turbine_options,flag):
        """Update and display plot for a specific turbine."""
        if index < len(turbine_options):
            turbine_name = turbine_options[index]
            plt.figure(figsize=(8, 6))
            plt.plot(
                self.df_merge['Fecha'], list_of_turb_power[index],
                label=turbine_name,
                color=['blue', 'green', 'red', 'lime', 'darkorange', 'aquamarine'][index]
            )
            plt.xlabel('Año')
            plt.ylabel('Potencia (kW)')
            plt.title(f'Modelo {turbine_name}')
            plt.legend()
            plt.grid(True)

            # Display the plot
            self.display_plot_on_canvas(flag)

            # Log the displayed plot
            self.clear_logs()
            if flag==True:
                self.graphInformation_2.clear()
                self.graphInformation_2.setText(f"Displayed {turbine_name} plot.\n")
            else:
                self.log_message(f"Displayed {turbine_name} plot.\n")


    def display_plot_on_canvas(self,flag):
        """Convert matplotlib figure to QPixmap and display on QLabel."""
        canvas = FigureCanvas(plt.gcf())
        canvas.draw()
        width, height = canvas.get_width_height()
        image = QImage(canvas.buffer_rgba(), width, height, QImage.Format_RGBA8888)
        if flag==True:
            self.lblGraph_2.setPixmap(QPixmap(image))
        else:
            self.lblGraph.setPixmap(QPixmap(image))




# Main application
app = QtWidgets.QApplication(sys.argv)

# Show the main screen
main_screen = MainScreen()
main_screen.show()

sys.exit(app.exec_())
