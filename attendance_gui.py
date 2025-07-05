# FACIAL RECOGNITION ATTENDANCE SYSTEM WITH STUDENT REGISTRATION
import cv2 as cv
from cv2.data import haarcascades
from keras_facenet import FaceNet
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import numpy as np
import os
import pickle
import warnings
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
from PIL import Image, ImageTk
import pandas as pd
from datetime import datetime, date
import threading
import json
import time

# Suppress warnings and configure environment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore', category=UserWarning)


class ModernAttendanceSystem:
    def __init__(self):
        # Initialize main window with modern styling
        self.root = tk.Tk()
        self.root.title("FaceTrack Pro - AI Attendance System")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1e1e1e')
        self.root.state('zoomed') if os.name == 'nt' else self.root.attributes('-zoomed', True)
        
        # Modern color scheme
        self.colors = {
            'primary': '#0d7377',      # Teal
            'secondary': '#14a085',    # Light teal
            'accent': '#40e0d0',       # Turquoise
            'success': '#00ff87',      # Bright green
            'warning': '#ffb347',      # Orange
            'danger': '#ff6b6b',       # Red
            'dark': '#1e1e1e',         # Dark background
            'surface': '#2d2d2d',      # Card background
            'text': '#ffffff',         # White text
            'text_secondary': '#b0b0b0' # Gray text
        }
        
        # Configure modern styling
        self._configure_styles()
        
        # Initialize face recognition components
        self._initialize_face_recognition()
        
        # Initialize video capture variables
        self.cap = None
        self.is_running = False
        self.video_thread = None
        
        # Registration mode variables
        self.is_registering = False
        self.registration_name = ""
        self.captured_faces = []
        self.registration_count = 0
        self.max_registration_samples = 20
        
        # Initialize attendance data
        self.attendance_data = []
        self.recognized_today = set()
        self.load_attendance_data()
        
        # Initialize GUI variables
        self.confidence_var = tk.DoubleVar(value=0.5)  # Lower default threshold
        
        # Setup modern GUI
        self.setup_modern_gui()
        
    def _configure_styles(self):
        """Configure modern ttk styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('Modern.TFrame', background=self.colors['surface'])
        style.configure('Card.TFrame', background=self.colors['surface'], relief='flat', borderwidth=1)
        style.configure('Title.TLabel', background=self.colors['dark'], foreground=self.colors['accent'], 
                       font=('Segoe UI', 24, 'bold'))
        style.configure('Heading.TLabel', background=self.colors['surface'], foreground=self.colors['text'], 
                       font=('Segoe UI', 12, 'bold'))
        style.configure('Modern.TLabel', background=self.colors['surface'], foreground=self.colors['text'],
                       font=('Segoe UI', 10))
        style.configure('Status.TLabel', background=self.colors['dark'], foreground=self.colors['text_secondary'],
                       font=('Segoe UI', 9))
        
        # Button styles
        style.configure('Primary.TButton', background=self.colors['primary'], foreground='white',
                       font=('Segoe UI', 10, 'bold'), borderwidth=0, focuscolor='none')
        style.map('Primary.TButton', background=[('active', self.colors['secondary'])])
        
        style.configure('Success.TButton', background=self.colors['success'], foreground='black',
                       font=('Segoe UI', 10, 'bold'), borderwidth=0, focuscolor='none')
        style.map('Success.TButton', background=[('active', '#00e07a')])
        
        style.configure('Warning.TButton', background=self.colors['warning'], foreground='black',
                       font=('Segoe UI', 10, 'bold'), borderwidth=0, focuscolor='none')
        style.map('Warning.TButton', background=[('active', '#ff9f33')])
        
        style.configure('Danger.TButton', background=self.colors['danger'], foreground='white',
                       font=('Segoe UI', 10, 'bold'), borderwidth=0, focuscolor='none')
        style.map('Danger.TButton', background=[('active', '#ff5252')])
        
        # Registration button style
        style.configure('Register.TButton', background='#6c5ce7', foreground='white',
                       font=('Segoe UI', 10, 'bold'), borderwidth=0, focuscolor='none')
        style.map('Register.TButton', background=[('active', '#5a4fcf')])
        
        # Scale style
        style.configure('Modern.Horizontal.TScale', background=self.colors['surface'],
                       troughcolor=self.colors['dark'], borderwidth=0, lightcolor=self.colors['accent'],
                       darkcolor=self.colors['accent'])
        
        # Treeview style
        style.configure('Modern.Treeview', background=self.colors['surface'], foreground=self.colors['text'],
                       fieldbackground=self.colors['surface'], borderwidth=0, font=('Segoe UI', 10))
        style.configure('Modern.Treeview.Heading', background=self.colors['primary'], foreground='white',
                       borderwidth=1, relief='flat', font=('Segoe UI', 10, 'bold'))
        style.map('Modern.Treeview', background=[('selected', self.colors['primary'])])
        
        # Separator style
        style.configure('Modern.TSeparator', background=self.colors['primary'])
        
    def _initialize_face_recognition(self):
        """Initialize all face recognition components"""
        try:
            print("🚀 Loading AI models...")
            self.facenet = FaceNet()
            
            # Load existing data or create new
            if os.path.exists('faces_embeddings_done_35classes.npz'):
                self.faces_embeddings = np.load('faces_embeddings_done_35classes.npz')
                self.X = self.faces_embeddings['arr_0']
                self.Y = self.faces_embeddings['arr_1']
            else:
                self.X = np.array([])
                self.Y = np.array([])
                
            self.encoder = LabelEncoder()
            if len(self.Y) > 0:
                self.encoder.fit(self.Y)
                
            # Ensure encoder matches model classes if model exists
            if hasattr(self, 'model') and self.model is not None and hasattr(self.model, 'classes_'):
                # Re-fit encoder with model classes to ensure compatibility
                self.encoder.fit(self.model.classes_)
                
            # Try to load fixed encoder if available
            if os.path.exists('fixed_encoder.pkl'):
                try:
                    with open('fixed_encoder.pkl', 'rb') as f:
                        self.encoder = pickle.load(f)
                    print("✅ Loaded fixed encoder")
                except Exception as e:
                    print(f"⚠️ Could not load fixed encoder: {e}")
            
            self.haarcascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
            
            # Load or create model
            if os.path.exists('svm_model_160x160.pkl') and len(self.Y) > 0:
                try:
                    self.model = pickle.load(open('svm_model_160x160.pkl', 'rb'))
                    
                    # Check if model is compatible with current data
                    if hasattr(self.model, 'classes_'):
                        model_classes = set(self.model.classes_)
                        current_classes = set(self.encoder.classes_)
                        
                        if model_classes != current_classes:
                            print("⚠️ Model mismatch detected - retraining with current data...")
                            self._retrain_model_with_current_data()
                        else:
                            print("✅ Model is compatible with current data")
                    else:
                        print("⚠️ Model format issue - retraining...")
                        self._retrain_model_with_current_data()
                        
                except Exception as e:
                    print(f"⚠️ Model loading issue - {e}")
                    self._retrain_model_with_current_data()
            else:
                self.model = None
                
            print("✅ AI models loaded successfully!")
            
            # Check and display model status
            self._check_model_status()
            
            # Fix encoder synchronization if needed
            self._fix_encoder_sync()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load AI models: {str(e)}")
            raise
            
    def _check_model_status(self):
        """Check and display the current model status"""
        if self.model is None:
            status_text = "⚠️ No AI model available - Register students to train model"
        elif len(self.Y) == 0:
            status_text = "⚠️ No training data - Register students to begin"
        else:
            unique_students = len(set(self.Y))
            status_text = f"✅ Model ready - {unique_students} students registered"
            
            # Check if ISRAEL OLUWAYEMI is in the model
            if 'ISRAEL OLUWAYEMI' in self.model.classes_:
                status_text += " (ISRAEL registered)"
            else:
                status_text += " (ISRAEL not found)"
            
        # Update status label if available
        if hasattr(self, 'status_label'):
            self.root.after(0, lambda: self.status_label.config(text=status_text))
            
    def _retrain_model_with_current_data(self):
        """Retrain the model with current data to fix compatibility issues"""
        try:
            if len(self.X) > 0 and len(self.Y) > 0:
                print("🔄 Retraining model with current data...")
                
                # Update status if available
                if hasattr(self, 'status_label'):
                    self.root.after(0, lambda: self.status_label.config(
                        text="🔄 Retraining AI model..."))
                
                self.model = SVC(kernel='linear', probability=True)
                self.model.fit(self.X, self.Y)
                
                # Ensure encoder matches model classes
                self.encoder.fit(self.model.classes_)
                
                # Save the updated model
                with open('svm_model_160x160.pkl', 'wb') as f:
                    pickle.dump(self.model, f)
                    
                print("✅ Model retrained and saved successfully!")
                print(f"✅ Encoder synchronized with {len(self.model.classes_)} classes")
                
                # Update status if available
                if hasattr(self, 'status_label'):
                    self.root.after(0, lambda: self.status_label.config(
                        text="✅ Model ready for recognition"))
            else:
                print("⚠️ No training data available")
                self.model = None
        except Exception as e:
            print(f"❌ Error retraining model: {e}")
            self.model = None
            
    def _fix_encoder_sync(self):
        """Fix encoder synchronization with model"""
        try:
            if self.model is not None and hasattr(self.model, 'classes_'):
                print("🔄 Synchronizing encoder with model classes...")
                self.encoder.fit(self.model.classes_)
                print(f"✅ Encoder synchronized with {len(self.model.classes_)} classes")
                return True
            else:
                print("⚠️ No model available for synchronization")
                return False
        except Exception as e:
            print(f"❌ Error synchronizing encoder: {e}")
            return False
        
    def setup_modern_gui(self):
        """Setup the modern GUI interface"""
        # Main container
        main_container = tk.Frame(self.root, bg=self.colors['dark'])
        main_container.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Configure grid weights
        main_container.columnconfigure(1, weight=2)  # Video gets more space
        main_container.columnconfigure(2, weight=1)  # Attendance panel
        main_container.rowconfigure(1, weight=1)
        
        # Setup sections
        self._setup_modern_header(main_container)
        self._setup_modern_sidebar(main_container)
        self._setup_modern_video_section(main_container)
        self._setup_modern_attendance_section(main_container)
        self._setup_modern_footer(main_container)
        
        # Update initial display
        self.update_display()
        
    def _setup_modern_header(self, parent):
        """Setup modern header with title and stats"""
        header_frame = tk.Frame(parent, bg=self.colors['dark'], height=80)
        header_frame.grid(row=0, column=0, columnspan=3, sticky='ew', pady=(0, 20))
        header_frame.grid_propagate(False)
        
        # Title section
        title_frame = tk.Frame(header_frame, bg=self.colors['dark'])
        title_frame.pack(side='left', fill='y')
        
        title_label = ttk.Label(title_frame, text="🎯 FaceTrack Pro", style='Title.TLabel')
        title_label.pack(anchor='w')
        
        subtitle_label = ttk.Label(title_frame, text="AI-Powered Attendance Management System", 
                                  background=self.colors['dark'], foreground=self.colors['text_secondary'],
                                  font=('Segoe UI', 11))
        subtitle_label.pack(anchor='w', pady=(5, 0))
        
        # Stats section
        stats_frame = tk.Frame(header_frame, bg=self.colors['dark'])
        stats_frame.pack(side='right', fill='y', padx=(20, 0))
        
        # Today's date
        today_label = ttk.Label(stats_frame, text=f"📅 {date.today().strftime('%B %d, %Y')}", 
                               background=self.colors['dark'], foreground=self.colors['text'],
                               font=('Segoe UI', 12, 'bold'))
        today_label.pack(anchor='e')
        
        # Quick stats
        self.quick_stats = ttk.Label(stats_frame, text="👥 Present: 0", 
                                    background=self.colors['dark'], foreground=self.colors['accent'],
                                    font=('Segoe UI', 11))
        self.quick_stats.pack(anchor='e', pady=(5, 0))
        
    def _setup_modern_sidebar(self, parent):
        """Setup modern sidebar with controls"""
        sidebar = ttk.Frame(parent, style='Card.TFrame', padding=20)
        sidebar.grid(row=1, column=0, sticky='nsew', padx=(0, 15))
        
        # Camera Controls Section
        self._create_section_header(sidebar, "🎥 Camera Controls", 0)
        
        self.start_btn = ttk.Button(sidebar, text="▶ Start Camera", 
                                   command=self.start_camera, style='Success.TButton', width=18)
        self.start_btn.grid(row=1, column=0, pady=(10, 5), sticky='ew')
        
        self.stop_btn = ttk.Button(sidebar, text="⏹ Stop Camera", 
                                  command=self.stop_camera, style='Danger.TButton', 
                                  width=18, state="disabled")
        self.stop_btn.grid(row=2, column=0, pady=5, sticky='ew')
        
        # Student Registration Section
        ttk.Separator(sidebar, style='Modern.TSeparator').grid(row=3, column=0, sticky='ew', pady=20)
        self._create_section_header(sidebar, "👤 Student Registration", 4)
        
        self.register_btn = ttk.Button(sidebar, text="➕ Register New Student", 
                                      command=self.start_registration, style='Register.TButton', width=18)
        self.register_btn.grid(row=5, column=0, pady=(10, 5), sticky='ew')
        
        self.view_students_btn = ttk.Button(sidebar, text="👥 View Registered Students", 
                                           command=self.view_registered_students, style='Primary.TButton', width=18)
        self.view_students_btn.grid(row=6, column=0, pady=5, sticky='ew')
        
        # Registration status
        self.registration_status = ttk.Label(sidebar, text="Ready for registration", 
                                           style='Modern.TLabel', font=('Segoe UI', 9))
        self.registration_status.grid(row=7, column=0, pady=(5, 0), sticky='w')
        
        # AI Settings Section
        ttk.Separator(sidebar, style='Modern.TSeparator').grid(row=8, column=0, sticky='ew', pady=20)
        self._create_section_header(sidebar, "🤖 AI Settings", 9)
        
        confidence_frame = ttk.Frame(sidebar, style='Modern.TFrame')
        confidence_frame.grid(row=10, column=0, sticky='ew', pady=(10, 0))
        
        ttk.Label(confidence_frame, text="🎯 Recognition Confidence", style='Modern.TLabel').pack(anchor='w')
        
        self.confidence_scale = ttk.Scale(confidence_frame, from_=0.3, to=0.95, 
                                         variable=self.confidence_var, orient="horizontal",
                                         style='Modern.Horizontal.TScale')
        self.confidence_scale.pack(fill='x', pady=(5, 0))
        self.confidence_scale.configure(command=self.update_confidence_label)
        
        confidence_display_frame = tk.Frame(confidence_frame, bg=self.colors['surface'])
        confidence_display_frame.pack(fill='x', pady=(5, 0))
        
        self.confidence_label = tk.Label(confidence_display_frame, text="70%", 
                                        bg=self.colors['primary'], fg='white',
                                        font=('Segoe UI', 10, 'bold'), padx=10, pady=5)
        self.confidence_label.pack()
        
        # Data Management Section
        ttk.Separator(sidebar, style='Modern.TSeparator').grid(row=11, column=0, sticky='ew', pady=20)
        self._create_section_header(sidebar, "📊 Data Management", 12)
        
        self.view_btn = ttk.Button(sidebar, text="👁 View All Records", 
                                  command=self.view_attendance, style='Primary.TButton', width=18)
        self.view_btn.grid(row=13, column=0, pady=(10, 5), sticky='ew')
        
        self.export_btn = ttk.Button(sidebar, text="📁 Export to CSV", 
                                    command=self.export_csv, style='Primary.TButton', width=18)
        self.export_btn.grid(row=14, column=0, pady=5, sticky='ew')
        
        self.clear_btn = ttk.Button(sidebar, text="🗑 Clear Today", 
                                   command=self.clear_today, style='Warning.TButton', width=18)
        self.clear_btn.grid(row=15, column=0, pady=5, sticky='ew')
        
        # Fix encoder sync button
        self.fix_btn = ttk.Button(sidebar, text="🔧 Fix Recognition", 
                                  command=self._fix_encoder_sync, style='Primary.TButton', width=18)
        self.fix_btn.grid(row=16, column=0, pady=5, sticky='ew')
        
        # Configure sidebar column weight
        sidebar.columnconfigure(0, weight=1)
        
    def _create_section_header(self, parent, text, row):
        """Create a modern section header"""
        header_label = ttk.Label(parent, text=text, style='Heading.TLabel')
        header_label.grid(row=row, column=0, sticky='w', pady=(0, 5))
        
    def _setup_modern_video_section(self, parent):
        """Setup modern video section"""
        video_frame = ttk.Frame(parent, style='Card.TFrame', padding=20)
        video_frame.grid(row=1, column=1, sticky='nsew', padx=10)
        
        # Video header
        video_header = tk.Frame(video_frame, bg=self.colors['surface'])
        video_header.pack(fill='x', pady=(0, 15))
        
        ttk.Label(video_header, text="📹 Live Camera Feed", style='Heading.TLabel').pack(side='left')
        
        # Camera status indicator
        self.camera_status = tk.Label(video_header, text="🔴 Offline", 
                                     bg=self.colors['surface'], fg=self.colors['danger'],
                                     font=('Segoe UI', 10, 'bold'))
        self.camera_status.pack(side='right')
        
        # Registration mode indicator
        self.mode_indicator = tk.Label(video_header, text="📷 Recognition Mode", 
                                      bg=self.colors['primary'], fg='white',
                                      font=('Segoe UI', 9, 'bold'), padx=10, pady=3)
        self.mode_indicator.pack(side='right', padx=(0, 10))
        
        # Video display area with proper container
        video_container = tk.Frame(video_frame, bg=self.colors['dark'], relief='solid', borderwidth=2)
        video_container.pack(fill='both', expand=True)
        
        # Create a centered frame for the video
        self.video_display_frame = tk.Frame(video_container, bg=self.colors['dark'])
        self.video_display_frame.pack(expand=True, fill='both')
        
        self.video_label = tk.Label(self.video_display_frame, 
                                   text="🎥\n\nCamera Ready\n\nClick 'Start Camera' to begin\nface recognition", 
                                   bg=self.colors['dark'], fg=self.colors['text_secondary'],
                                   font=('Segoe UI', 16), justify='center')
        self.video_label.pack(expand=True, anchor='center')
        
    def _setup_modern_attendance_section(self, parent):
        """Setup modern attendance section"""
        attendance_frame = ttk.Frame(parent, style='Card.TFrame', padding=20)
        attendance_frame.grid(row=1, column=2, sticky='nsew', padx=(15, 0))
        
        # Attendance header
        attendance_header = tk.Frame(attendance_frame, bg=self.colors['surface'])
        attendance_header.pack(fill='x', pady=(0, 15))
        
        ttk.Label(attendance_header, text="📋 Today's Attendance", style='Heading.TLabel').pack(side='left')
        
        # Live counter
        self.live_counter = tk.Label(attendance_header, text="0", 
                                    bg=self.colors['success'], fg='black',
                                    font=('Segoe UI', 12, 'bold'), padx=8, pady=4, borderwidth=2, relief='solid')
        self.live_counter.pack(side='right')
        
        # Search box
        search_frame = tk.Frame(attendance_frame, bg=self.colors['surface'])
        search_frame.pack(fill='x', pady=(0, 10))
        
        tk.Label(search_frame, text="🔍", bg=self.colors['surface'], fg=self.colors['text'],
                font=('Segoe UI', 12)).pack(side='left', padx=(0, 5))
        
        self.search_var = tk.StringVar()
        search_entry = tk.Entry(search_frame, textvariable=self.search_var,
                               bg=self.colors['dark'], fg=self.colors['text'],
                               font=('Segoe UI', 10), borderwidth=1, relief='solid')
        search_entry.pack(side='left', fill='x', expand=True)
        search_entry.bind('<KeyRelease>', self.filter_attendance)
        
        # Attendance list
        list_frame = tk.Frame(attendance_frame, bg=self.colors['surface'])
        list_frame.pack(fill='both', expand=True)
        
        # Configure treeview
        self.tree = ttk.Treeview(list_frame, columns=("Name", "Time"), show="headings", 
                                style='Modern.Treeview', height=25)
        self.tree.heading("Name", text="👤 Student Name")
        self.tree.heading("Time", text="⏰ Check-in Time")
        self.tree.column("Name", width=150)
        self.tree.column("Time", width=120)
        
        # Modern scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack treeview and scrollbar
        self.tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Configure attendance frame
        attendance_frame.columnconfigure(0, weight=1)
        attendance_frame.rowconfigure(1, weight=1)
        
    def _setup_modern_footer(self, parent):
        """Setup modern footer with status"""
        footer_frame = tk.Frame(parent, bg=self.colors['surface'], height=40)
        footer_frame.grid(row=2, column=0, columnspan=3, sticky='ew', pady=(20, 0))
        footer_frame.grid_propagate(False)
        
        # Status indicator
        status_container = tk.Frame(footer_frame, bg=self.colors['surface'])
        status_container.pack(fill='both', expand=True, padx=15, pady=8)
        
        self.status_indicator = tk.Label(status_container, text="🟢", 
                                        bg=self.colors['surface'], font=('Segoe UI', 12))
        self.status_indicator.pack(side='left')
        
        self.status_label = ttk.Label(status_container, text="System Ready - AI models loaded successfully", 
                                     style='Status.TLabel')
        self.status_label.pack(side='left', padx=(10, 0))
        
        # System info
        registered_count = len(set(self.Y)) if len(self.Y) > 0 else 0
        info_label = ttk.Label(status_container, text=f"FaceTrack Pro v2.0 | Registered Students: {registered_count}", 
                              style='Status.TLabel')
        info_label.pack(side='right')
        
    # ===================== STUDENT REGISTRATION =====================
    
    def start_registration(self):
        """Start the student registration process"""
        if self.is_running:
            messagebox.showwarning("Camera Active", "Please stop the camera before starting registration.")
            return
            
        # Get student name
        name = simpledialog.askstring("Student Registration", 
                                     "Enter student's full name:\n\n(This will be used for attendance tracking)",
                                     parent=self.root)
        
        if not name or not name.strip():
            return
            
        name = name.strip()
        
        # Check if student already exists
        if len(self.Y) > 0 and name in self.Y:
            if not messagebox.askyesno("Student Exists", 
                                      f"Student '{name}' is already registered.\n\nDo you want to add more training samples?"):
                return
        
        # Initialize registration
        self.registration_name = name
        self.captured_faces = []
        self.registration_count = 0
        self.is_registering = True
        
        # Update UI
        self.register_btn.config(state="disabled")
        self.mode_indicator.config(text="📸 Registration Mode", bg='#6c5ce7')
        self.registration_status.config(text=f"Registering: {name}")
        
        # Start camera for registration
        self.cap = cv.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Could not access camera for registration.")
            self._reset_registration()
            return
            
        self.is_running = True
        self.start_btn.config(state="disabled")
        self.camera_status.config(text="🟢 Registering", fg=self.colors['warning'])
        self.status_label.config(text=f"📸 Registering {name} - Look at camera and move slightly")
        
        # Start video processing thread
        self.video_thread = threading.Thread(target=self.process_registration_video)
        self.video_thread.daemon = True
        self.video_thread.start()
        
    def process_registration_video(self):
        """Process video frames for student registration"""
        last_capture_time = 0
        capture_interval = 0.5  # Capture every 0.5 seconds
        
        while self.is_running and self.is_registering:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            current_time = time.time()
            
            # Perform face detection and capture
            processed_frame = self._process_registration_frame(frame, current_time, last_capture_time, capture_interval)
            
            # Update last capture time if we captured a face
            if len(self.captured_faces) > self.registration_count:
                last_capture_time = current_time
                self.registration_count = len(self.captured_faces)
                
                # Update status
                remaining = self.max_registration_samples - self.registration_count
                self.root.after(0, lambda: self.registration_status.config(
                    text=f"Captured: {self.registration_count}/{self.max_registration_samples} - {remaining} more needed"))
                
                # Check if we have enough samples
                if self.registration_count >= self.max_registration_samples:
                    self.root.after(0, self._complete_registration)
                    break
            
            # Display frame
            self._display_frame(processed_frame)
            cv.waitKey(1)
            
        if self.cap:
            self.cap.release()
            
    def _process_registration_frame(self, frame, current_time, last_capture_time, capture_interval):
        """Process frame for registration"""
        rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = self.haarcascade.detectMultiScale(gray_img, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Filter out small faces
            if w < 100 or h < 100:
                continue
                
            # Check if enough time has passed since last capture
            can_capture = current_time - last_capture_time >= capture_interval
            
            if can_capture and len(self.captured_faces) < self.max_registration_samples:
                # Extract and process face
                face_img = rgb_img[y:y+h, x:x+w]
                face_img = cv.resize(face_img, (160, 160))
                
                # Quality check - ensure face is clear enough
                if self._is_face_quality_good(face_img):
                    self.captured_faces.append(face_img)
                    
                    # Draw capture indicator
                    cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 5)
                    cv.rectangle(frame, (x, y-40), (x+w, y), (0, 255, 0), -1)
                    cv.putText(frame, f"CAPTURED! {len(self.captured_faces)}/{self.max_registration_samples}", 
                              (x+5, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv.LINE_AA)
                else:
                    # Draw quality warning
                    cv.rectangle(frame, (x, y), (x+w, y+h), (0, 165, 255), 3)
                    cv.rectangle(frame, (x, y-40), (x+w, y), (0, 165, 255), -1)
                    cv.putText(frame, "MOVE CLOSER/CLEARER", (x+5, y-10), 
                              cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
            else:
                # Draw normal detection rectangle
                cv.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 3)
                cv.rectangle(frame, (x, y-40), (x+w, y), (255, 255, 0), -1)
                
                if len(self.captured_faces) >= self.max_registration_samples:
                    status_text = "PROCESSING..."
                else:
                    remaining_time = max(0, capture_interval - (current_time - last_capture_time))
                    status_text = f"READY IN {remaining_time:.1f}s"
                
                cv.putText(frame, status_text, (x+5, y-10), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv.LINE_AA)
        
        # Draw progress bar
        progress = len(self.captured_faces) / self.max_registration_samples
        bar_width = int(frame.shape[1] * 0.8)
        bar_height = 20
        bar_x = int((frame.shape[1] - bar_width) / 2)
        bar_y = frame.shape[0] - 50
        
        # Background bar
        cv.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
        # Progress bar
        progress_width = int(bar_width * progress)
        cv.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), (0, 255, 0), -1)
        # Progress text
        progress_text = f"Registration Progress: {len(self.captured_faces)}/{self.max_registration_samples}"
        cv.putText(frame, progress_text, (bar_x, bar_y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)
        
        return frame
        
    def _is_face_quality_good(self, face_img):
        """Check if the captured face image quality is good enough"""
        # Convert to grayscale for quality checks
        gray = cv.cvtColor(face_img, cv.COLOR_RGB2GRAY)
        
        # Check if image is too dark or too bright
        mean_brightness = np.mean(gray)
        if mean_brightness < 50 or mean_brightness > 200:
            return False
            
        # Check sharpness using Laplacian variance
        laplacian_var = cv.Laplacian(gray, cv.CV_64F).var()
        if laplacian_var < 100:  # Too blurry
            return False
            
        return True
        
    def _complete_registration(self):
        """Complete the student registration process"""
        try:
            self.is_running = False
            self.is_registering = False
            
            if self.cap:
                self.cap.release()
                
            # Update UI
            self.status_label.config(text=f"🔄 Processing {len(self.captured_faces)} face samples for {self.registration_name}...")
            self.registration_status.config(text="Processing samples...")
            
            # Process captured faces
            self._process_captured_faces()
            
        except Exception as e:
            messagebox.showerror("Registration Error", f"Failed to complete registration: {str(e)}")
            self._reset_registration()
            
    def _process_captured_faces(self):
        """Process captured face samples and update the model"""
        try:
            # Generate embeddings for captured faces
            face_embeddings = []
            for face_img in self.captured_faces:
                face_img_expanded = np.expand_dims(face_img, axis=0)
                embedding = self.facenet.embeddings(face_img_expanded)
                face_embeddings.append(embedding[0])
                
            face_embeddings = np.array(face_embeddings)
            
            # Create labels for new faces
            new_labels = [self.registration_name] * len(face_embeddings)
            
            # Update existing data
            if len(self.X) == 0:
                # First student
                self.X = face_embeddings
                self.Y = np.array(new_labels)
            else:
                # Add to existing data
                self.X = np.vstack([self.X, face_embeddings])
                self.Y = np.hstack([self.Y, new_labels])
            
            # Update encoder
            self.encoder = LabelEncoder()
            self.encoder.fit(self.Y)
            
            # Train new model
            self.status_label.config(text="🤖 Training AI model with new data...")
            self.root.update()
            
            self.model = SVC(kernel='linear', probability=True)
            self.model.fit(self.X, self.Y)
            
            # Save updated data and model
            np.savez('faces_embeddings_done_35classes.npz', self.X, self.Y)
            with open('svm_model_160x160.pkl', 'wb') as f:
                pickle.dump(self.model, f)
                
            # Save student registry
            self._save_student_registry()
            
            # Show success message
            unique_students = len(set(self.Y))
            messagebox.showinfo("Registration Complete", 
                              f"✅ Student '{self.registration_name}' registered successfully!\n\n"
                              f"📊 Samples captured: {len(self.captured_faces)}\n"
                              f"👥 Total registered students: {unique_students}\n"
                              f"🤖 AI model updated and ready for recognition")
            
            # Reset registration state
            self._reset_registration()
            self.status_label.config(text=f"✅ Registration complete - {self.registration_name} added to system")
            
            # Update footer with new count
            self._update_footer_student_count()
            
            # Check model status
            self._check_model_status()
            
        except Exception as e:
            messagebox.showerror("Processing Error", f"Failed to process face samples: {str(e)}")
            self._reset_registration()
            
    def _reset_registration(self):
        """Reset registration state"""
        self.is_registering = False
        self.registration_name = ""
        self.captured_faces = []
        self.registration_count = 0
        
        # Reset UI
        self.register_btn.config(state="normal")
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.mode_indicator.config(text="📷 Recognition Mode", bg=self.colors['primary'])
        self.camera_status.config(text="🔴 Offline", fg=self.colors['danger'])
        self.registration_status.config(text="Ready for registration")
        
        # Clear video display
        self.video_label.config(image="", text="🎥\n\nCamera Ready\n\nClick 'Start Camera' to begin\nface recognition")
        if hasattr(self.video_label, 'image'):
            self.video_label.image = None
            
    def _save_student_registry(self):
        """Save student registry with registration details"""
        try:
            registry_file = "student_registry.json"
            
            # Load existing registry or create new
            if os.path.exists(registry_file):
                with open(registry_file, 'r') as f:
                    registry = json.load(f)
            else:
                registry = {"students": []}
            
            # Check if student already exists
            existing_student = None
            for student in registry["students"]:
                if student["name"] == self.registration_name:
                    existing_student = student
                    break
            
            if existing_student:
                # Update existing student
                existing_student["samples_count"] += len(self.captured_faces)
                existing_student["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            else:
                # Add new student
                new_student = {
                    "name": self.registration_name,
                    "registration_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "samples_count": len(self.captured_faces),
                    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                registry["students"].append(new_student)
            
            # Save registry
            with open(registry_file, 'w') as f:
                json.dump(registry, f, indent=2)
                
        except Exception as e:
            print(f"Error saving student registry: {e}")
            
    def view_registered_students(self):
        """View all registered students"""
        try:
            registry_file = "student_registry.json"
            
            if not os.path.exists(registry_file) or len(self.Y) == 0:
                messagebox.showinfo("No Students", "👥 No students registered yet\n\nUse 'Register New Student' to add students to the system.")
                return
            
            # Load registry
            with open(registry_file, 'r') as f:
                registry = json.load(f)
            
            # Create modern window
            view_window = tk.Toplevel(self.root)
            view_window.title("👥 Registered Students - FaceTrack Pro")
            view_window.geometry("800x600")
            view_window.configure(bg=self.colors['dark'])
            view_window.transient(self.root)
            
            # Modern header
            header = tk.Frame(view_window, bg=self.colors['primary'], height=60)
            header.pack(fill='x')
            header.pack_propagate(False)
            
            tk.Label(header, text="👥 Registered Students Database", 
                    bg=self.colors['primary'], fg='white',
                    font=('Segoe UI', 16, 'bold')).pack(pady=15)
            
            # Content frame
            content = tk.Frame(view_window, bg=self.colors['dark'])
            content.pack(fill='both', expand=True, padx=20, pady=20)
            
            # Stats frame
            stats_frame = tk.Frame(content, bg=self.colors['surface'], relief='solid', borderwidth=1)
            stats_frame.pack(fill='x', pady=(0, 15))
            
            total_students = len(registry["students"])
            total_samples = sum(student["samples_count"] for student in registry["students"])
            
            stats_text = f"📊 Total Students: {total_students} | 🎯 Total Samples: {total_samples} | 🤖 Model Status: {'Ready' if self.model else 'Not Trained'}"
            tk.Label(stats_frame, text=stats_text, bg=self.colors['surface'], fg=self.colors['text'],
                    font=('Segoe UI', 10, 'bold')).pack(pady=10)
            
            # Create modern treeview
            tree = ttk.Treeview(content, columns=("Name", "Registration Date", "Samples", "Last Updated"), 
                               show="headings", style='Modern.Treeview', height=18)
            tree.heading("Name", text="👤 Student Name")
            tree.heading("Registration Date", text="📅 Registration Date")
            tree.heading("Samples", text="🎯 Samples")
            tree.heading("Last Updated", text="⏰ Last Updated")
            
            # Configure column widths
            tree.column("Name", width=200)
            tree.column("Registration Date", width=150)
            tree.column("Samples", width=100)
            tree.column("Last Updated", width=150)
            
            # Modern scrollbar
            scrollbar = ttk.Scrollbar(content, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=scrollbar.set)
            
            # Pack widgets
            tree.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            # Add data
            for i, student in enumerate(sorted(registry["students"], key=lambda x: x["name"])):
                tags = ('evenrow',) if i % 2 == 0 else ('oddrow',)
                tree.insert("", "end", values=(
                    student["name"], 
                    student["registration_date"].split()[0],  # Date only
                    student["samples_count"],
                    student["last_updated"].split()[0]  # Date only
                ), tags=tags)
            
            # Configure row colors
            tree.tag_configure('evenrow', background=self.colors['surface'])
            tree.tag_configure('oddrow', background=self.colors['dark'])
            
            # Buttons frame
            button_frame = tk.Frame(content, bg=self.colors['dark'])
            button_frame.pack(fill='x', pady=(15, 0))
            
            # Add buttons for additional functionality
            ttk.Button(button_frame, text="🔄 Refresh", 
                      command=lambda: view_window.destroy() or self.view_registered_students(),
                      style='Primary.TButton').pack(side='left', padx=(0, 10))
            
            ttk.Button(button_frame, text="📁 Export Student List", 
                      command=lambda: self._export_student_registry(registry),
                      style='Primary.TButton').pack(side='left', padx=(0, 10))
            
            ttk.Button(button_frame, text="❌ Close", 
                      command=view_window.destroy,
                      style='Danger.TButton').pack(side='right')
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load student registry: {str(e)}")
            
    def _export_student_registry(self, registry):
        """Export student registry to CSV"""
        try:
            # Generate default filename
            today = date.today().strftime("%Y-%m-%d")
            default_filename = f"Student_Registry_{today}.csv"
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")],
                title="💾 Export Student Registry",
                initialvalue=default_filename
            )
            
            if filename:
                df = pd.DataFrame(registry["students"])
                
                if filename.endswith('.xlsx'):
                    df.to_excel(filename, index=False)
                else:
                    df.to_csv(filename, index=False)
                    
                messagebox.showinfo("Export Successful", 
                    f"✅ Student registry exported successfully!\n\n"
                    f"📁 File: {filename}\n"
                    f"👥 Students: {len(df)}")
        except Exception as e:
            messagebox.showerror("Export Error", f"❌ Failed to export registry:\n\n{str(e)}")
            
    def _update_footer_student_count(self):
        """Update the footer with current student count"""
        try:
            registered_count = len(set(self.Y)) if len(self.Y) > 0 else 0
            # Find the info label in footer and update it
            for widget in self.root.winfo_children():
                if isinstance(widget, tk.Frame):
                    for child in widget.winfo_children():
                        if isinstance(child, tk.Frame):
                            for grandchild in child.winfo_children():
                                if isinstance(grandchild, tk.Frame) and grandchild.winfo_children():
                                    for label in grandchild.winfo_children():
                                        if isinstance(label, ttk.Label) and "FaceTrack Pro v2.0" in str(label.cget('text')):
                                            label.config(text=f"FaceTrack Pro v2.0 | Registered Students: {registered_count}")
                                            return
        except:
            pass  # Silently fail if we can't update the footer
        
    # ===================== CAMERA AND VIDEO PROCESSING =====================
    
    def start_camera(self):
        """Start the camera and face recognition"""
        if self.is_registering:
            messagebox.showwarning("Registration Active", "Please complete or cancel registration before starting recognition.")
            return
            
        if not self.model:
            messagebox.showwarning("No Model", "🤖 No trained model available\n\nPlease register at least one student before starting recognition.")
            return
            
        self.cap = cv.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "❌ Could not access camera\n\nPlease check if camera is connected and not being used by another application.")
            return
            
        self.is_running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.register_btn.config(state="disabled")  # Disable registration during recognition
        self.camera_status.config(text="🟢 Live", fg=self.colors['success'])
        self.status_indicator.config(text="🔴")
        self.status_label.config(text="Camera active - Scanning for faces...")
        
        # Start video processing thread
        self.video_thread = threading.Thread(target=self.process_video)
        self.video_thread.daemon = True
        self.video_thread.start()
        
    def stop_camera(self):
        """Stop the camera"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            cv.destroyAllWindows()  # Clean up any opencv windows
        
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.register_btn.config(state="normal")  # Re-enable registration
        self.camera_status.config(text="🔴 Offline", fg=self.colors['danger'])
        self.status_indicator.config(text="🟢")
        self.status_label.config(text="Camera stopped - Ready to start")
        
        # Clear the video display and show ready message
        self.video_label.config(image="", text="🎥\n\nCamera Ready\n\nClick 'Start Camera' to begin\nface recognition")
        # Clear the image reference
        if hasattr(self.video_label, 'image'):
            self.video_label.image = None
        
    def process_video(self):
        """Process video frames for face recognition"""
        frame_count = 0
        while self.is_running and not self.is_registering:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame from camera")
                break
                
            # Process every frame but display every 2nd frame for smoother performance
            frame_count += 1
            
            # Perform face recognition
            processed_frame = self._perform_face_recognition(frame)
            
            # Display frame (reduce frequency if needed)
            if frame_count % 1 == 0:  # Display every frame
                self._display_frame(processed_frame)
            
            # Small delay to prevent overwhelming the GUI thread
            cv.waitKey(1)
            
        if self.cap:
            self.cap.release()
        print("Video processing stopped")
            
    def _perform_face_recognition(self, frame):
        """Perform face recognition on a frame"""
        # Check if model is ready
        if self.model is None:
            # Draw "Model Not Ready" message on frame
            cv.putText(frame, "Model Not Ready", (50, 50), cv.FONT_HERSHEY_SIMPLEX,
                      1, (0, 0, 255), 2, cv.LINE_AA)
            return frame
            
        rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = self.haarcascade.detectMultiScale(gray_img, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Filter out small faces
            if w < 80 or h < 80:
                continue
                
            # Extract and process face
            face_img = rgb_img[y:y+h, x:x+w]
            face_img = cv.resize(face_img, (160, 160))
            face_img = np.expand_dims(face_img, axis=0)
            
            try:
                # Get face embeddings and prediction
                embeddings = self.facenet.embeddings(face_img)
                prediction = self.model.predict(embeddings)
                confidence_scores = self.model.predict_proba(embeddings)
                max_confidence = np.max(confidence_scores)
                
                # Debug: Print prediction info
                predicted_class = prediction[0]
                print(f"🔍 Debug: Predicted class index: {predicted_class}, Max confidence: {max_confidence:.3f}")
                
            except Exception as e:
                print(f"⚠️ Face recognition error: {e}")
                # Draw error indicator
                cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
                cv.putText(frame, "Error", (x+5, y-10), cv.FONT_HERSHEY_SIMPLEX,
                          0.6, (0, 0, 255), 2, cv.LINE_AA)
                continue
            
            # Apply confidence threshold
            confidence_threshold = self.confidence_var.get()
            
            if max_confidence >= confidence_threshold:
                # Recognized face
                try:
                    name = self.encoder.inverse_transform(prediction)[0]
                    self.mark_attendance(name)
                except ValueError as e:
                    # Handle case where model predicts unseen label
                    print(f"⚠️ Model prediction error: {e}")
                    # Treat as unknown face
                    name = "Unknown"
                    # Draw unknown face styling instead
                    cv.rectangle(frame, (x, y), (x+w, y+h), (100, 100, 255), 3)
                    
                    # Unknown label background
                    label_bg_height = 35
                    cv.rectangle(frame, (x, y-label_bg_height), (x+w, y), (100, 100, 255), -1)
                    
                    label = f"Unknown ({max_confidence:.1%})"
                    cv.putText(frame, label, (x+5, y-10), cv.FONT_HERSHEY_SIMPLEX,
                              0.6, (255, 255, 255), 2, cv.LINE_AA)
                                  
                    # Unknown indicator
                    cv.circle(frame, (x+w-15, y+15), 8, (100, 100, 255), -1)
                    cv.putText(frame, "?", (x+w-20, y+20), cv.FONT_HERSHEY_SIMPLEX,
                              0.5, (255, 255, 255), 2, cv.LINE_AA)
                    continue
                
                # Draw modern green rectangle and label
                cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 100), 3)
                
                # Modern label background
                label_bg_height = 35
                cv.rectangle(frame, (x, y-label_bg_height), (x+w, y), (0, 255, 100), -1)
                
                # Text label
                label = f"{name} ({max_confidence:.1%})"
                cv.putText(frame, label, (x+5, y-10), cv.FONT_HERSHEY_SIMPLEX,
                          0.6, (0, 0, 0), 2, cv.LINE_AA)
                          
                # Recognition indicator
                cv.circle(frame, (x+w-15, y+15), 8, (0, 255, 100), -1)
                cv.putText(frame, "✓", (x+w-20, y+20), cv.FONT_HERSHEY_SIMPLEX,
                          0.5, (0, 0, 0), 2, cv.LINE_AA)
            else:
                # Unrecognized face
                cv.rectangle(frame, (x, y), (x+w, y+h), (100, 100, 255), 3)
                
                # Unknown label background
                label_bg_height = 35
                cv.rectangle(frame, (x, y-label_bg_height), (x+w, y), (100, 100, 255), -1)
                
                label = f"Unknown ({max_confidence:.1%})"
                cv.putText(frame, label, (x+5, y-10), cv.FONT_HERSHEY_SIMPLEX,
                          0.6, (255, 255, 255), 2, cv.LINE_AA)
                          
                # Unknown indicator
                cv.circle(frame, (x+w-15, y+15), 8, (100, 100, 255), -1)
                cv.putText(frame, "?", (x+w-20, y+20), cv.FONT_HERSHEY_SIMPLEX,
                          0.5, (255, 255, 255), 2, cv.LINE_AA)
                
        return frame
        
    def _display_frame(self, frame):
        """Convert and display frame in GUI with proper aspect ratio"""
        try:
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            # Get current video label dimensions (with fallback values)
            self.video_label.update_idletasks()  # Ensure geometry is updated
            label_width = max(self.video_label.winfo_width(), 640)
            label_height = max(self.video_label.winfo_height(), 480)
            
            # Calculate display area (with padding)
            display_width = max(label_width - 40, 320)
            display_height = max(label_height - 40, 240)
            
            # Get original frame dimensions
            original_width, original_height = frame_pil.size
            if original_height == 0:  # Prevent division by zero
                return
                
            original_aspect = original_width / original_height
            
            # Calculate new dimensions maintaining aspect ratio
            if display_width / display_height > original_aspect:
                # Display is wider than frame aspect ratio
                new_height = display_height
                new_width = int(new_height * original_aspect)
            else:
                # Display is taller than frame aspect ratio
                new_width = display_width
                new_height = int(new_width / original_aspect)
            
            # Ensure minimum size
            new_width = max(new_width, 160)
            new_height = max(new_height, 120)
            
            # Resize maintaining aspect ratio
            frame_pil = frame_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
            frame_tk = ImageTk.PhotoImage(frame_pil)
            
            # Update in main thread
            self.root.after_idle(self._update_video_label, frame_tk)
            
        except Exception as e:
            print(f"Error displaying frame: {e}")
            # Continue running even if frame display fails
        
    def _update_video_label(self, frame_tk):
        """Update video label in main thread"""
        if self.is_running:
            self.video_label.config(image=frame_tk, text="", compound='center')
            self.video_label.image = frame_tk  # Keep a reference to prevent garbage collection
            
    # ===================== ATTENDANCE MANAGEMENT =====================
    
    def mark_attendance(self, name):
        """Mark attendance for a recognized person"""
        today = date.today().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M:%S")
        person_key = f"{name}_{today}"
        
        # Check if already marked today
        if person_key not in self.recognized_today:
            self.recognized_today.add(person_key)
            
            # Add attendance record
            attendance_record = {
                "Name": name,
                "Date": today,
                "Time": current_time,
                "Status": "Present"
            }
            self.attendance_data.append(attendance_record)
            
            # Update display and save data
            self.root.after(0, self.update_display)
            self.save_attendance_data()
            
            # Update status with modern styling
            self.root.after(0, lambda: self.status_label.config(
                text=f"✅ {name} marked present at {current_time}"))
            
    def update_display(self):
        """Update the attendance display"""
        # Clear current items
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        # Get today's attendance
        today = date.today().strftime("%Y-%m-%d")
        today_attendance = [record for record in self.attendance_data 
                          if record["Date"] == today]
        
        # Sort by time (newest first)
        today_attendance.sort(key=lambda x: x["Time"], reverse=True)
        
        # Filter by search if needed
        search_term = self.search_var.get().lower() if hasattr(self, 'search_var') else ""
        if search_term:
            today_attendance = [record for record in today_attendance 
                              if search_term in record["Name"].lower()]
        
        # Add to tree with alternating colors
        for i, record in enumerate(today_attendance):
            tags = ('evenrow',) if i % 2 == 0 else ('oddrow',)
            self.tree.insert("", "end", values=(record["Name"], record["Time"]), tags=tags)
            
        # Configure row colors
        self.tree.tag_configure('evenrow', background=self.colors['surface'])
        self.tree.tag_configure('oddrow', background=self.colors['dark'])
        
        # Update counters
        count = len(today_attendance)
        self.live_counter.config(text=str(count))
        self.quick_stats.config(text=f"👥 Present: {count}")
        
    def filter_attendance(self, event=None):
        """Filter attendance list based on search"""
        self.update_display()
        
    # ===================== GUI EVENT HANDLERS =====================
    
    def update_confidence_label(self, value):
        """Update confidence threshold label with modern styling"""
        confidence_percent = int(float(value) * 100)
        self.confidence_label.config(text=f"{confidence_percent}%")
        
        # Color coding for confidence levels
        if confidence_percent >= 80:
            self.confidence_label.config(bg=self.colors['success'], fg='black')
        elif confidence_percent >= 60:
            self.confidence_label.config(bg=self.colors['warning'], fg='black')
        else:
            self.confidence_label.config(bg=self.colors['danger'], fg='white')
        
    def view_attendance(self):
        """View all attendance records in modern window"""
        if not self.attendance_data:
            messagebox.showinfo("No Data", "📊 No attendance records found\n\nStart the camera to begin tracking attendance.")
            return
            
        # Create modern window
        view_window = tk.Toplevel(self.root)
        view_window.title("📊 All Attendance Records - FaceTrack Pro")
        view_window.geometry("900x600")
        view_window.configure(bg=self.colors['dark'])
        view_window.transient(self.root)
        
        # Modern header
        header = tk.Frame(view_window, bg=self.colors['primary'], height=60)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        tk.Label(header, text="📊 Complete Attendance History", 
                bg=self.colors['primary'], fg='white',
                font=('Segoe UI', 16, 'bold')).pack(pady=15)
        
        # Content frame
        content = tk.Frame(view_window, bg=self.colors['dark'])
        content.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Create modern treeview
        tree = ttk.Treeview(content, columns=("Name", "Date", "Time", "Status"), 
                           show="headings", style='Modern.Treeview', height=20)
        tree.heading("Name", text="👤 Student Name")
        tree.heading("Date", text="📅 Date")
        tree.heading("Time", text="⏰ Time")
        tree.heading("Status", text="✅ Status")
        
        # Modern scrollbar
        scrollbar = ttk.Scrollbar(content, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack widgets
        tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Add data with modern styling
        sorted_data = sorted(self.attendance_data, 
                           key=lambda x: (x["Date"], x["Time"]), reverse=True)
        for i, record in enumerate(sorted_data):
            tags = ('evenrow',) if i % 2 == 0 else ('oddrow',)
            tree.insert("", "end", values=(record["Name"], record["Date"], 
                                         record["Time"], record["Status"]), tags=tags)
        
        # Configure row colors
        tree.tag_configure('evenrow', background=self.colors['surface'])
        tree.tag_configure('oddrow', background=self.colors['dark'])
            
    def export_csv(self):
        """Export attendance data to CSV with modern dialog"""
        if not self.attendance_data:
            messagebox.showinfo("No Data", "📊 No attendance data to export\n\nStart tracking attendance first.")
            return
            
        # Generate default filename with timestamp
        today = date.today().strftime("%Y-%m-%d")
        timestamp = datetime.now().strftime("%H%M")
        default_filename = f"FaceTrack_Attendance_{today}_{timestamp}.csv"
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")],
            title="💾 Export Attendance Data",
            initialvalue=default_filename
        )
        
        if filename:
            try:
                df = pd.DataFrame(self.attendance_data)
                df = df.sort_values(['Date', 'Time'], ascending=[False, False])
                
                if filename.endswith('.xlsx'):
                    df.to_excel(filename, index=False)
                else:
                    df.to_csv(filename, index=False)
                    
                messagebox.showinfo("Export Successful", 
                    f"✅ Attendance data exported successfully!\n\n"
                    f"📁 File: {filename}\n"
                    f"📊 Records: {len(df)}\n"
                    f"📅 Date Range: {df['Date'].min()} to {df['Date'].max()}")
            except Exception as e:
                messagebox.showerror("Export Error", f"❌ Failed to export data:\n\n{str(e)}")
                
    def clear_today(self):
        """Clear today's attendance data with modern confirmation"""
        today = date.today().strftime("%Y-%m-%d")
        today_count = len([r for r in self.attendance_data if r["Date"] == today])
        
        if today_count == 0:
            messagebox.showinfo("No Data", "📊 No attendance records for today\n\nNothing to clear.")
            return
            
        # Custom modern confirmation dialog
        if messagebox.askyesno("⚠️ Confirm Clear", 
                              f"Are you sure you want to clear today's attendance?\n\n"
                              f"📊 Records to delete: {today_count}\n"
                              f"📅 Date: {today}\n\n"
                              f"⚠️ This action cannot be undone!",
                              icon='warning'):
            # Remove today's data
            self.attendance_data = [record for record in self.attendance_data 
                                  if record["Date"] != today]
            self.recognized_today = {key for key in self.recognized_today 
                                   if not key.endswith(today)}
            
            # Update display and save
            self.update_display()
            self.save_attendance_data()
            self.status_label.config(text=f"🗑️ Cleared {today_count} records for today")
            
    # ===================== DATA PERSISTENCE =====================
    
    def save_attendance_data(self):
        """Save attendance data to file"""
        try:
            with open("attendance_data.json", "w") as f:
                json.dump(self.attendance_data, f, indent=2)
        except Exception as e:
            print(f"💾 Error saving data: {e}")
            
    def load_attendance_data(self):
        """Load attendance data from file"""
        try:
            with open("attendance_data.json", "r") as f:
                self.attendance_data = json.load(f)
                
            # Rebuild today's recognition set
            today = date.today().strftime("%Y-%m-%d")
            self.recognized_today = {f"{record['Name']}_{record['Date']}" 
                                   for record in self.attendance_data 
                                   if record['Date'] == today}
        except FileNotFoundError:
            self.attendance_data = []
            self.recognized_today = set()
        except Exception as e:
            print(f"💾 Error loading data: {e}")
            self.attendance_data = []
            self.recognized_today = set()
            
    # ===================== APPLICATION LIFECYCLE =====================
    
    def on_closing(self):
        """Handle application closing with modern confirmation"""
        if self.is_running:
            if messagebox.askyesno("🎥 Camera Active", 
                                  "Camera is currently active.\n\nDo you want to stop the camera and exit?"):
                self.stop_camera()
                self.root.after(100, self.root.destroy)  # Small delay to ensure clean shutdown
        else:
            self.root.destroy()
        
    def run(self):
        """Start the modern application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Center window on screen
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (self.root.winfo_width() // 2)
        y = (self.root.winfo_screenheight() // 2) - (self.root.winfo_height() // 2)
        self.root.geometry(f"+{x}+{y}")
        
        # Show splash effect
        self.root.attributes('-alpha', 0.0)
        self.root.deiconify()
        self._fade_in()
        
        self.root.mainloop()
        
    def _fade_in(self):
        """Fade in effect for modern app launch"""
        alpha = self.root.attributes('-alpha')
        if alpha < 1.0:
            self.root.attributes('-alpha', alpha + 0.1)
            self.root.after(50, self._fade_in)


# ===================== MAIN EXECUTION =====================

if __name__ == "__main__":
    try:
        print("🚀 Starting FaceTrack Pro...")
        app = ModernAttendanceSystem()
        app.run()
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        input("Press Enter to exit...")
    except KeyboardInterrupt:
        print("\n👋 Application interrupted by user")
        print("Goodbye!")