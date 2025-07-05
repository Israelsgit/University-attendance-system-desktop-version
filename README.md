# FaceTrack Pro - AI-Powered Facial Recognition Attendance System

A modern, intelligent attendance management system that uses facial recognition technology to automatically track student attendance in real-time.

## 🚀 Features

- **Real-time Face Recognition**: Advanced AI-powered facial recognition using FaceNet and SVM
- **Student Registration**: Easy student enrollment with multiple face samples
- **Modern GUI**: Beautiful, intuitive interface built with Tkinter
- **Attendance Tracking**: Automatic attendance marking with timestamps
- **Data Export**: Export attendance records to CSV/Excel
- **Student Management**: View and manage registered students
- **Configurable Recognition**: Adjustable confidence thresholds
- **Cross-platform**: Works on Windows, macOS, and Linux

## 📋 Requirements

### Python Dependencies

```
tensorflow>=2.0.0
opencv-python>=4.5.0
keras-facenet>=0.3.1
scikit-learn>=1.0.0
numpy>=1.21.0
pandas>=1.3.0
Pillow>=8.0.0
```

### System Requirements

- Python 3.7 or higher
- Webcam for face capture
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space

## 🛠️ Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Israelsgit/facial-recognition-attendance-system.git
   cd facial-recognition-attendance-system
   ```

2. **Create virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python attendance_gui.py
   ```

## 🎯 Quick Start

1. **Launch the Application**

   - Run `python attendance_gui.py`
   - The modern GUI will open with all controls

2. **Register Students**

   - Click "➕ Register New Student"
   - Enter student's full name
   - Look at the camera and move slightly
   - System captures 20 face samples automatically

3. **Start Attendance Tracking**

   - Click "▶ Start Camera"
   - Students are automatically recognized and marked present
   - View real-time attendance in the right panel

4. **Export Data**
   - Click "📁 Export to CSV" to download attendance records
   - Use "👁 View All Records" to see complete history

## 📁 Project Structure

```
facial-recognition-attendance-system/
├── attendance_gui.py          # Main application GUI
├── main.py                    # Entry point
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── .gitignore               # Git ignore rules
├── haarcascade_frontalface_default.xml  # Face detection model
├── faces_embeddings_done_35classes.npz  # Face embeddings data
├── svm_model_160x160.pkl    # Trained SVM model
├── student_registry.json     # Student database
└── attendance_data.json      # Attendance records
```

## 🔧 Configuration

### Recognition Settings

- **Confidence Threshold**: Adjust from 30% to 95% for recognition sensitivity
- **Face Quality**: System automatically checks image quality during registration
- **Model Training**: Automatic retraining when new students are added

### Data Management

- **Automatic Backup**: Attendance data saved automatically
- **Export Options**: CSV, Excel formats supported
- **Student Registry**: Complete student database with registration details

## 🎨 Features in Detail

### Modern Interface

- **Dark Theme**: Professional dark mode interface
- **Real-time Updates**: Live attendance counter and status indicators
- **Responsive Design**: Adapts to different screen sizes
- **Intuitive Controls**: Easy-to-use buttons and sliders

### AI Recognition

- **FaceNet Embeddings**: State-of-the-art face feature extraction
- **SVM Classification**: Robust machine learning for face matching
- **Quality Control**: Automatic face quality assessment
- **Multi-sample Training**: 20 face samples per student for accuracy

### Attendance System

- **Duplicate Prevention**: Prevents multiple entries per day
- **Timestamp Recording**: Exact check-in times
- **Search Functionality**: Filter attendance records
- **Data Persistence**: Automatic saving and loading

## 📊 Usage Examples

### Registering a New Student

```python
# The GUI handles this automatically
# 1. Click "Register New Student"
# 2. Enter name: "John Doe"
# 3. Look at camera for 20 samples
# 4. System trains model automatically
```

### Starting Attendance Tracking

```python
# 1. Click "Start Camera"
# 2. Students are recognized automatically
# 3. Attendance marked with timestamps
# 4. View real-time updates
```

### Exporting Data

```python
# Export to CSV
# File: FaceTrack_Attendance_2024-01-15_1430.csv
# Format: Name, Date, Time, Status
```

## 🔍 Troubleshooting

### Common Issues

1. **Camera Not Working**

   - Check if camera is connected
   - Ensure no other application is using the camera
   - Try restarting the application

2. **Recognition Not Working**

   - Ensure good lighting conditions
   - Check if students are properly registered
   - Adjust confidence threshold if needed

3. **Model Training Issues**
   - Delete model files and re-register students
   - Ensure sufficient face samples during registration
   - Check Python dependencies are correctly installed

### Performance Tips

- Use good lighting for better recognition
- Keep faces clearly visible to camera
- Register multiple angles during student enrollment
- Close other applications to free up system resources

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **FaceNet**: Google's deep learning face recognition system
- **OpenCV**: Computer vision library for face detection
- **scikit-learn**: Machine learning library for SVM implementation
- **Tkinter**: GUI framework for the modern interface

## 📞 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Review the code comments for technical details
3. Open an issue on GitHub with detailed description
4. Include system information and error messages

## 🔄 Version History

- **v2.0** - Modern GUI, improved recognition, better data management
- **v1.0** - Initial release with basic functionality

---

**Made with ❤️ for efficient attendance management**
