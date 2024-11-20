import tkinter as tk
from tkinter import ttk
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
import tkinter.filedialog as filedialog
from PIL import Image, ImageTk  # To work with images in Tkinter

class HydropowerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Hydropower Workflow Application")
        self.geometry("1060x680")
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
        # Register all pages
        for Page in (HomePage, ResultsPage):
            page_instance = Page(self)
            self.pages[Page] = page_instance
            page_instance.place(relwidth=1, relheight=1)

    def show_page(self, page_class):
        # Bring the desired page to the front
        page = self.pages[page_class]
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

        self.dry_years = None
        self.wet_years = None
        self.normal_years = None
        self.df_merge = None
        self.water_tra_value = None
        self.water_density_value = None
        self.turbine_option = None
        self.velocidad = None

        self.current_graph = 0  # Start with Caudal graph

        
        # Initialize the UI components
        self.initialize_ui(parent)
        self.log_message_2("Please select two data files to begin: First one must start with 'Q' and and second one must start with 'NV'.\n")
        # Setup your Tkinter components like label and graph information area
        # self.lblGraph = tk.Label(parent)
        # self.lblGraph.pack()
        
    def load_data(self):
        """Function that loads the data files when btnCarga is clicked."""
        # Clear logs for the current step
        self.clear_logs()
        self.log_message("Starting data loading...\n")

        # First dialog box for selecting a file
        first_file = self.select_file()

        if first_file:
            # Extract the filename to check the prefix
            first_filename = os.path.basename(first_file)

            # Check if the first file starts with 'Q'
            if first_filename.startswith("Q"):
                self.caudal_file = first_file
                self.log_message(f"First file (Q) selected: {first_filename}\n")

                # Prompt user for second file with prefix "NV"
                second_file = self.select_file("NV")
            else:
                self.log_message("Error: The first file must start with 'Q'. Please try again.")
                return  # Exit if the first file does not have 'Q' prefix
            
            # Check and assign the second file if selected correctly
            if second_file:
                second_filename = os.path.basename(second_file)
                
                # Check if second file starts with 'NV'
                if second_filename.startswith("NV"):
                    self.nivel_file = second_file
                    self.log_message(f"Second file (NV) selected: {second_filename}")
                    self.log_message_2("Please click 'Pretratamiento de datos' to continue")

                else:
                    self.log_message("Error: The second file must start with 'NV'. Please try again")
                    return  # Exit if the second file does not have 'NV' prefix
                
                # Load data from files
                try:
                    self.caudal_data = pd.read_csv(self.caudal_file, delimiter='|', decimal='.')
                    self.nivel_data = pd.read_csv(self.nivel_file, delimiter='|', decimal='.')
                    
                    # Convert 'Fecha' to datetime format for both caudal and nivel datasets
                    self.caudal_data["Fecha"] = pd.to_datetime(self.caudal_data["Fecha"], errors='coerce')
                    self.nivel_data["Fecha"] = pd.to_datetime(self.nivel_data["Fecha"], errors='coerce')
                    self.log_message("Date conversion successful.")
                    
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
                    self.log_message(f"<span style='color:red;'>Error loading data: {e}</span>")
            else:
                self.log_message("<span style='color:red;'>Error: Second file not selected properly. Please try again.</span>")
        else:
            self.log_message("<span style='color:red;'>Error: First file not selected properly. Please try again.</span>")
    def select_file(self, required_prefix=None):
        """Function for file selection using Tkinter's filedialog"""
        # Open file dialog to select a file
        file_path = filedialog.askopenfilename(title="Select File", filetypes=[("Data files", "*.data")])
        
        if file_path:  # Check if a file was selected
            file_name = os.path.basename(file_path)  # Get the file name without path
            
            # If required_prefix is given, check if the file has the correct prefix
            if required_prefix and not file_name.startswith(required_prefix):
                self.log_message(f"Error: The selected file must start with '{required_prefix}'. Please select a correct file")
                return None  # Return None if the file does not match the expected prefix
            
            # Ensure the file has not been selected already for both 'Q' and 'NV'
            if (required_prefix == "Q" and file_path == self.nivel_file) or (required_prefix == "NV" and file_path == self.caudal_file):
                self.log_message("Error: You cannot select the same file for both 'Q' and 'NV'. Please choose a different file.")
                return None  # Return None to indicate an error

            return file_path  # Return the selected file path if it matches the prefix
        else:
            self.log_message("No file selected. Please try again.")
            return None  # Return None if no file was selected
    def show_graph(self):
        """Update the graph sequentially every 5 seconds using Tkinter's after() method."""
        if self.current_graph == 0:
            # Display Caudal line graph
            self.plot_data_in_label(self.caudal_data, ylabel='Caudal (m3/s)', title='Gráfico de Línea de Caudal')
            self.current_graph = 1
        elif self.current_graph == 1:
            # Display Nivel line graph
            self.plot_data_in_label(self.nivel_data, ylabel='Nivel (m)', title='Gráfico de Línea de Nivel')
            self.current_graph = 2
        elif self.current_graph == 2:
            # Display the Caudal heatmap
            self.plot_heatmap(self.caudal_data, 'Caudal')
            self.current_graph = 3
        elif self.current_graph == 3:
            # Display the Nivel heatmap
            self.plot_heatmap(self.nivel_data, 'Nivel')

        # Schedule the next update in 5000 milliseconds (5 seconds)
        if self.current_graph < 4:  # Continue if we haven't finished
            self.parent.after(5000, self.show_graph)  # Update graph every 5 seconds
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

        # Load the image with PIL
        image = Image.open(temp_filename)

        # Get the size of the label (self.lblGraph)
        label_width = self.lblGraph.winfo_width()
        label_height = self.lblGraph.winfo_height()

        # Resize the image to fit within the label size
        image_resized = image.resize((label_width, label_height), Image.Resampling.LANCZOS)

        # Convert the resized image to a Tkinter-compatible format
        image_tk = ImageTk.PhotoImage(image_resized)
        
        # Set the image in the Tkinter label (assuming self.lblGraph is the label)
        self.lblGraph.config(image=image_tk)
        self.lblGraph.image = image_tk  # Keep a reference to avoid garbage collection
        
        # Delete the temporary file
        os.remove(temp_filename)
        
        # Display statistics in QTextEdit
        stats = data.describe()
        self.show_information(f"{ylabel} Data Statistics:\n{stats.to_string()}")

        
        # Close the figure to free memory
        plt.close(fig)

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

        # Load the image with PIL
        image = Image.open(temp_filename)

        # Get the size of the label (self.lblGraph)
        label_width = self.lblGraph.winfo_width()
        label_height = self.lblGraph.winfo_height()

        # Resize the image to fit within the label size
        image_resized = image.resize((label_width, label_height),  Image.Resampling.LANCZOS)

        # Convert the resized image to a Tkinter-compatible format
        image_tk = ImageTk.PhotoImage(image_resized)
        
        # Set the image in the Tkinter label (assuming self.lblGraph is the label)
        self.lblGraph.config(image=image_tk)
        self.lblGraph.image = image_tk  # Keep a reference to avoid garbage collection
        
        # Delete the temporary file
        os.remove(temp_filename)
        
        description = dataset_grouped['Valor'].describe()
        graph_informtaion = ''
        graph_informtaion += f"{label} Data - Heatmap of Missing Values:\n"
        graph_informtaion += f"Visualizes missing data for each day of the year across years.\n"
        graph_informtaion += f"Descriptive Statistics of {label} Data:\n"
        graph_informtaion += f"Count: {description['count']}, Mean: {description['mean']}, Std: {description['std']}, Min: {description['min']}, Max: {description['max']}"
        self.show_information(f"{graph_informtaion}")


    def log_message(self, message):
        """Helper function to add messages to the Text log area."""
        self.graphInformation.config(state=tk.NORMAL)  # Enable text box to update logs
        self.graphInformation.insert(tk.END, message + "\n")  # Append message with newline to the log
        self.graphInformation.config(state=tk.DISABLED)  # Disable text box to make it read-only

    def clear_logs(self):
        """Function to clear the log area."""
        self.graphInformation.config(state=tk.NORMAL)  # Enable text box to clear
        self.graphInformation.delete(1.0, tk.END)  # Delete all text
        self.graphInformation.config(state=tk.DISABLED)  # Disable text box to make it read-only
    def log_message_2(self, message):
    
        """Helper function to add messages to the Text log area."""
        self.textEditLogs.config(state=tk.NORMAL)  # Enable text box to update logs
        self.textEditLogs.insert(tk.END, message + "\n")   # Append message to the log
        self.textEditLogs.config(state=tk.DISABLED)  # Disable text box to make it read-only

    def clear_logs_2(self):
        """Function to clear the log area."""
        self.textEditLogs.config(state=tk.NORMAL)  # Enable text box to clear
        self.textEditLogs.delete(1.0, tk.END)  # Delete all text
        self.textEditLogs.config(state=tk.DISABLED)  # Disable text box to make it read-only
    def show_information(self, message):
        self.clear_information()
        """Display the given message in the graph information area (Text widget)."""
        self.graphInformation.config(state=tk.NORMAL)  # Enable text box to update
        self.graphInformation.delete(1.0, tk.END)  # Clear any previous content
        self.graphInformation.insert(tk.END, message)  # Insert new message
        self.graphInformation.config(state=tk.DISABLED)  # Disable text box to make it read-only

    def clear_information(self):
        """Clear the information displayed in the graph information area (Text widget)."""
        self.graphInformation.config(state=tk.NORMAL)
        self.graphInformation.delete(1.0, tk.END)  # Clear the content
        self.graphInformation.config(state=tk.DISABLED)  # Disable the text box again
    def initialize_ui(self, parent):
        """Function to initialize the UI components"""

        # Create the frame for the sidebar
        self.frame = tk.Frame(self.bg, bg="white", width=400)
        self.frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # Create a user section
        user_frame = tk.Frame(self.frame, bg="white", height=5)
        user_frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=10)
        user_label = tk.Label(user_frame, text="Usuario", bg="white", fg="black", font=("Helvetica", 14, "bold"), anchor="w")
        user_label.pack(fill=tk.BOTH, expand=False)

        # Create the button section
        self.button_frame = tk.Frame(self.frame, bg="white", padx=10)
        self.button_frame.pack(fill=tk.Y, expand=True, pady=(80, 0))  # Top padding: 20, Bottom padding: 0

        # Create the buttons (equivalent to btnCarga, btnPreta, etc.)
        self.btnCarga = tk.Button(self.button_frame, text="Carga de archivos", bg="blue", fg="white", font=("Helvetica", 12), width=20, command=self.load_data)
        self.btnCarga.pack(pady=5)
        self.apply_button_styles(self.btnCarga)

        self.btnPreta = tk.Button(self.button_frame, text="Pretratamiento de datos", bg="blue", fg="white", font=("Helvetica", 12), width=20, state=tk.DISABLED)
        self.btnPreta.pack(pady=5)
        self.apply_button_styles(self.btnPreta)

        self.btnTrata = tk.Button(self.button_frame, text="Tratamiento de datos", bg="blue", fg="white", font=("Helvetica", 12), width=20, state=tk.DISABLED, command=self.show_Tratamiento)
        self.btnTrata.pack(pady=5)
        self.apply_button_styles(self.btnTrata)

        self.btnProce = tk.Button(self.button_frame, text="Procesamiento", bg="blue", fg="white", font=("Helvetica", 12), width=20, state=tk.DISABLED, command=self.open_popup_window)
        self.btnProce.pack(pady=5)
        self.apply_button_styles(self.btnProce)

        self.btnResult = tk.Button(self.button_frame, text="Resultados", bg="blue", fg="white", font=("Helvetica", 12), width=20, state=tk.DISABLED)
        self.btnResult.pack(pady=5)
        self.apply_button_styles(self.btnResult)
        
        self.btnSimulator = tk.Button(self.button_frame, text="Simular",bg="#4d4eba", state=tk.DISABLED, fg="white", font=("Helvetica", 12), width=20, relief="flat", bd=2, highlightbackground="#4d4eba", highlightthickness=2, cursor="hand2")
        self.btnSimulator.pack(pady=5, anchor='e')

        # Add hover effects specific to btnSimulator
        self.btnSimulator.bind("<Enter>", lambda e: self.on_simulator_hover())
        self.btnSimulator.bind("<Leave>", lambda e: self.on_simulator_leave())
        
        # Create the log section (Text widget)
        text_frame = tk.Frame(self.frame, bg="white", width=50)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.textEditLogs = tk.Text(text_frame, height=10, width=50, bg="#e9ecef",font=("Helvetica", 11), wrap="word",padx=10, pady=5)
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
        self.btnViewAllGraphs = tk.Button(self.content_frame, text="View All Graphs", bg="#4d4eba", fg="white", font=("Helvetica", 12), width=20, relief="flat", bd=2, highlightbackground="#4d4eba", highlightthickness=2, cursor="hand2", command=lambda: parent.show_page(ResultsPage))
        self.btnViewAllGraphs.pack(side=tk.TOP, pady=10, padx=10, anchor="e")

        # Add hover effects specific to btnViewAllGraphs
        self.btnViewAllGraphs.bind("<Enter>", lambda e: self.on_view_all_graphs_hover())
        self.btnViewAllGraphs.bind("<Leave>", lambda e: self.on_view_all_graphs_leave())

        # Add a label for graph information
        self.lblGraph = tk.Label(self.content_frame, text="Graph will be displayed here", bg="#f5f5f5", font=("Helvetica", 14), borderwidth=2, relief="solid")
        self.lblGraph.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Add a Text widget to show graph information (like QTextEdit for graph information)
        self.graphInformation = tk.Text(self.content_frame, height=10, width=50, bg="#e9ecef",font=("Helvetica", 11),padx=10, pady=5)
        self.graphInformation.config(state=tk.DISABLED)  # Making it read-only
        self.graphInformation.pack(fill=tk.BOTH, padx=10, pady=10)
    
    def open_popup_window(self):
        """Creates and opens a pop-up window containing the combo box and button, centered on the screen."""
        # Create a new top-level window (pop-up)
        popup = tk.Toplevel(self.bg)
        popup.title("Select Turbine Graph")  # Set title for the pop-up window

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
        self.cmbxTurbineGraph = self.create_combo_box(center_frame, "Select Turbine Graphs", 
                                                    ["All", "SmartFreeStream", "SmartMonoFloat", "EnviroGen005series", 
                                                    "Hydroquest1.4", "EVG-050H", "EVG-050H"])

        # Confirm button
        self.btnConfirm_2 = tk.Button(
            center_frame,
            text="Confirm",
            bg="#4d4eba",  # Background color
            fg="white",  # Text color
            font=("Helvetica", 12),
            width=10,
            relief="flat",  # Flat border
            bd=2,  # Border width
            highlightbackground="#4d4eba",  # Border color
            highlightthickness=2,  # Border thickness
            cursor="hand2",  # Pointer cursor
            command=self.show_content_frame  # Button command to go back to main content
        )
        self.btnConfirm_2.pack(pady=20, padx=20, anchor='e')  # Padding for the button

    def apply_button_styles(self, button):
        # Apply normal style to the button
        button.config(
            bg="white", 
            fg="black", 
            bd=2,  # border width
            relief="solid",  # solid border to replicate Qt style
            font=("Helvetica", 12),  # Font size and style
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
            page, height=10, width=50, bg="#e9ecef", wrap="word", font=("Helvetica", 11),padx=10, pady=5
        )
        self.availableYears.insert("1.0", "Available years information will go here.")
        self.availableYears.config(state=tk.DISABLED)
        self.availableYears.pack(fill=tk.X, pady=50, padx=20)

        # Container for input fields, centered on the page
        input_container = tk.Frame(page, bg="white")
        input_container.pack(fill=tk.BOTH, expand=True, pady=30)  # Center the container vertically

        # Input fields
        self.txtDryYears = self.create_input_field(input_container, "Dry Years","Enter Dry Years (Comma separate)")
        self.txtWetYears = self.create_input_field(input_container, "Wet Years","Enter Wet Years (Comma separate)")
        self.txtNormalYears = self.create_input_field(input_container, "Normal Years","Enter Normal Years (Comma separate)")

        # Confirm button
        self.btnConfirm = tk.Button(
            input_container,
            text="Confirm",
            bg="#4d4eba",  # Background color
            fg="white",  # Text color
            font=("Helvetica", 12),
            width=15,
            
            relief="flat",  # Flat border
            bd=2,  # Border width
            highlightbackground="#4d4eba",  # Border color
            highlightthickness=2,  # Border thickness
            cursor="hand2",  # Pointer cursor
            command=self.show_content_frame
        )
        self.btnConfirm.pack(pady=20,padx=20,anchor='e')  # Padding for the button

        return page

    def create_input_field(self, parent, label_text, placeholder="Enter text..."):
        """Creates a styled input field with a label above it, aligned to the left, and with a placeholder."""
        frame = tk.Frame(parent, bg="white")
        frame.pack(pady=10)  # Add vertical spacing between input fields

        # Label (above the entry field)
        label = tk.Label(frame, text=label_text, bg="white", font=("Helvetica", 12), anchor="w", justify="left")
        label.pack(fill=tk.X, padx=20,pady=5, anchor="w")  # Align to the left with padding

        # Entry field with placeholder functionality
        entry_frame = tk.Frame(frame, bg="white")  # Wrapper for the entry and the border
        entry_frame.pack(fill=tk.X, padx=20)

        entry = tk.Entry(
            entry_frame,
            font=("Helvetica", 13),
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
        self.cmbxTurbineGraph = self.create_combo_box(center_frame, "Select Turbine Graphs", 
                                                    ["All", "SmartFreeStream", "SmartMonoFloat", "EnviroGen005series", 
                                                    "Hydroquest1.4", "EVG-050H", "EVG-050H"])

        # Confirm button
        self.btnConfirm_2 = tk.Button(
            center_frame,
            text="Confirm",
            bg="#4d4eba",  # Background color
            fg="white",  # Text color
            font=("Helvetica", 12),
            width=10,
            relief="flat",  # Flat border
            bd=2,  # Border width
            highlightbackground="#4d4eba",  # Border color
            highlightthickness=2,  # Border thickness
            cursor="hand2",  # Pointer cursor
            command=self.show_content_frame  # Button command to go back to main content
        )
        self.btnConfirm_2.pack(pady=20, padx=20, anchor='e')  # Padding for the button

        return page

    def create_combo_box(self, parent, label_text, options):
        """Creates a combo box with a label above it."""
        frame = tk.Frame(parent, bg="white")
        frame.pack(pady=10)  # Add vertical spacing between input fields

        # Label (above the combo box)
        label = tk.Label(frame, text=label_text, bg="white", font=("Helvetica", 12), anchor="w")
        label.pack(fill=tk.X, padx=20, pady=5, anchor="w")  # Align to the left with padding

        # Combo Box for selecting options
        combo_box = ttk.Combobox(frame, values=options, font=("Helvetica", 13), state="readonly")
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
    def __init__(self, parent):
        super().__init__(parent, bg="white")
        self.bg = tk.Frame(self, bg="white")
        self.bg.pack(fill=tk.BOTH, expand=True)
        
        # Create the frame for the sidebar, similar to QFrame
        self.frame = tk.Frame(self.bg, bg="white", width=400)
        self.frame.pack(side=tk.LEFT, fill=tk.Y)
        # Create a frame for the QTextEdit (Text widget) in the sidebar (light gray)
        user_frame = tk.Frame(self.frame, bg="white", height=5)
        user_frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=10)

        # Create the Label for user section and align it to the left
        user_label = tk.Label(user_frame, text="Graph List", bg="white", fg="black", font=("Helvetica", 14, "bold"))
        user_label.pack(fill=tk.BOTH, expand=False)
        # Create a frame below the label to contain the Treeview
        tree_frame = tk.Frame(self.frame, bg="white")
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Add a Treeview widget to the frame
        self.tree = ttk.Treeview(tree_frame, columns=("Graph Name", "Date"), show="headings", height=15)
        self.tree.pack(fill=tk.BOTH, expand=True)

        # Configure Treeview columns
        self.tree.heading("Graph Name", text="Graph Name")
        self.tree.heading("Date", text="Date")
        self.tree.column("Graph Name", width=200, anchor="center")
        self.tree.column("Date", width=100, anchor="center")

        # Example data to populate the Treeview
        sample_data = [
            ("Hydrograph A", "2024-01-01"),
            ("Flow Analysis B", "2024-02-15"),
            ("Performance C", "2024-03-10"),
        ]
        for graph_name, date in sample_data:
            self.tree.insert("", "end", values=(graph_name, date))

        # Add a scrollbar for the Treeview
        tree_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        
        # Create the content section on the right side (like stacked widget)
        self.content_frame = tk.Frame(self.bg, bg="white")
        self.content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)  # Expands to occupy the remaining space

        self.btnBack = tk.Button(
        self.content_frame,
        text="Back",
        bg="#4d4eba",  # Background color
        fg="white",  # Text color
        font=("Helvetica", 12),
        width=20,
        relief="flat",  # Flat border
        bd=2,  # Border width
        highlightbackground="#4d4eba",  # Border color
        highlightthickness=2,  # Border thickness
        cursor="hand2",  # Pointer cursor
        command=lambda: parent.show_page(HomePage)

    )
        self.btnBack.pack(side=tk.TOP, pady=10, anchor="e")

        # Add hover effects specific to btnViewAllGraphs
        self.btnBack.bind("<Enter>", lambda e: self.on_btnBack_hover())
        self.btnBack.bind("<Leave>", lambda e: self.on_btnBack_leave())


        # Add a label for graph information
        self.lblGraph = tk.Label(self.content_frame, text="Graph will be displayed here", bg="lightgray", font=("Helvetica", 14), borderwidth=2, relief="solid")
        self.lblGraph.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Add a Text widget to show graph information (like QTextEdit for graph information)
        self.graphInformation = tk.Text(self.content_frame,font=("Helvetica", 11), height=10, width=50, bg="#e9ecef")
        self.graphInformation.config(state=tk.DISABLED)  # Making it read-only
        self.graphInformation.pack(fill=tk.BOTH,  padx=10, pady=10)
    def apply_button_styles(self, button):
        # Apply normal style to the button
        button.config(
            bg="white", 
            fg="black", 
            bd=2,  # border width
            relief="solid",  # solid border to replicate Qt style
            font=("Helvetica", 12),  # Font size and style
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
    def on_btnBack_hover(self):
        """Handles hover effects specifically for btnViewAllGraphs."""
        self.btnBack.config(bg="#6c70ca", fg="#f6f8ff")

    def on_btnBack_leave(self):
        """Handles leave effects specifically for btnViewAllGraphs."""
        self.btnBack.config(bg="#4d4eba", fg="white")
    

    
if __name__ == "__main__":
    app = HydropowerApp()
    app.mainloop()