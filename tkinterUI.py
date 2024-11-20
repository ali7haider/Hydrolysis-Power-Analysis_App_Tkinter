import tkinter as tk
from tkinter import ttk

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
        # Create a horizontal layout (like QHBoxLayout)
        self.bg = tk.Frame(self, bg="white")
        self.bg.pack(fill=tk.BOTH, expand=True)
        
        # Create the frame for the sidebar, similar to QFrame
        self.frame = tk.Frame(self.bg, bg="white", width=400)
        self.frame.pack(side=tk.LEFT, fill=tk.Y)
        # Create a frame for the QTextEdit (Text widget) in the sidebar (light gray)
        user_frame = tk.Frame(self.frame, bg="white", height=5)
        user_frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=10)

        # Create the Label for user section and align it to the left
        user_label = tk.Label(user_frame, text="Usuario", bg="white", fg="black", font=("Helvetica", 14, "bold"), anchor="w")
        user_label.pack(fill=tk.BOTH, expand=False)
        # Create the button section
        self.button_frame = tk.Frame(self.frame, bg="white", padx=10)
        self.button_frame.pack(fill=tk.Y, expand=True, pady=(80, 0))  # Top padding: 20, Bottom padding: 0

        
        # Create the buttons (equivalent to btnCarga, btnPreta, etc.)
        self.btnCarga = tk.Button(self.button_frame, text="Carga de archivos", bg="blue", fg="white", font=("Helvetica", 12), width=20)
        self.btnCarga.pack(pady=5)
        self.apply_button_styles(self.btnCarga)

        self.btnPreta = tk.Button(self.button_frame, text="Pretratamiento de datos", bg="blue", fg="white", font=("Helvetica", 12), width=20)
        self.btnPreta.pack(pady=5)
        self.apply_button_styles(self.btnPreta)
        
        self.btnTrata = tk.Button(self.button_frame, text="Tratamiento de datos", bg="blue", fg="white", font=("Helvetica", 12), width=20,command=self.show_Tratamiento)
        self.btnTrata.pack(pady=5)
        self.apply_button_styles(self.btnTrata)
        
        self.btnProce = tk.Button(self.button_frame, text="Procesamiento", bg="blue", fg="white", font=("Helvetica", 12), width=20,command=self.open_popup_window)
        self.btnProce.pack(pady=5)
        self.apply_button_styles(self.btnProce)
       
        self.btnResult = tk.Button(self.button_frame, text="Resultados", bg="blue", fg="white", font=("Helvetica", 12), width=20)
        self.btnResult.pack(pady=5)
        self.apply_button_styles(self.btnResult)
        
        self.btnSimulator = tk.Button(self.button_frame, text="Simular",bg="#4d4eba",  # Background color
        fg="white",  # Text color
        font=("Helvetica", 12),
        width=20,
        relief="flat",  # Flat border
        bd=2,  # Border width
        highlightbackground="#4d4eba",  # Border color
        highlightthickness=2,  # Border thickness
        cursor="hand2",  # Pointer cursor

    )
        self.btnSimulator.pack(pady=5,anchor='e')
        # Add hover effects specific to btnViewAllGraphs
        self.btnSimulator.bind("<Enter>", lambda e: self.on_simulator_hover())
        self.btnSimulator.bind("<Leave>", lambda e: self.on_simulator_leave())
        
        # Create the log section (equivalent to QTextEditLogs)
        # Create a frame for the QTextEdit (Text widget) in the sidebar (light gray)
        text_frame = tk.Frame(self.frame, bg="white", width=50)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create the Text widget for logging or output
        self.textEditLogs = tk.Text(text_frame, height=10, width=50, bg="#f5f5f5",wrap="word")
        self.textEditLogs.config(state=tk.DISABLED)  # Making it read-only

        self.textEditLogs.pack(fill=tk.BOTH, pady=10,expand=True)
        # Create the content section on the right side (like stacked widget)
        self.content_frame = tk.Frame(self.bg, bg="white")
        self.content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)  # Expands to occupy the remaining space
        # Tratamiento page (hidden by default)
        self.page_procesamiento = self.create_procesamiento_page()
        self.page_procesamiento.pack_forget()  # Initially hidden

        self.page_Tratamiento = self.create_Tratamiento_page()
        self.page_Tratamiento.pack_forget()  # Initially hidden

        self.btnViewAllGraphs = tk.Button(
        self.content_frame,
        text="View All Graphs",
        bg="#4d4eba",  # Background color
        fg="white",  # Text color
        font=("Helvetica", 12),
        width=20,
        relief="flat",  # Flat border
        bd=2,  # Border width
        highlightbackground="#4d4eba",  # Border color
        highlightthickness=2,  # Border thickness
        cursor="hand2",  # Pointer cursor
        command=lambda: parent.show_page(ResultsPage)

    )
        self.btnViewAllGraphs.pack(side=tk.TOP, pady=10, padx=10,anchor="e")

        # Add hover effects specific to btnViewAllGraphs
        self.btnViewAllGraphs.bind("<Enter>", lambda e: self.on_view_all_graphs_hover())
        self.btnViewAllGraphs.bind("<Leave>", lambda e: self.on_view_all_graphs_leave())


        # Add a label for graph information
        self.lblGraph = tk.Label(self.content_frame, text="Graph will be displayed here", bg="#f5f5f5", font=("Helvetica", 14), borderwidth=2, relief="solid")
        self.lblGraph.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Add a Text widget to show graph information (like QTextEdit for graph information)
        self.graphInformation = tk.Text(self.content_frame, height=10, width=50, bg="#f5f5f5")
        self.graphInformation.config(state=tk.DISABLED)  # Making it read-only
        self.graphInformation.pack(fill=tk.BOTH,  padx=10, pady=10)
    
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
            page, height=10, width=50, bg="#f5f5f5", wrap="word", font=("Helvetica", 11),padx=10, pady=5
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
        self.graphInformation = tk.Text(self.content_frame, height=10, width=50, bg="#f5f5f5")
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
