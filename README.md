# FightFitAI 🥊🤖

![FightFitAI](https://img.shields.io/badge/FightFitAI-AI%20Boxing%20Trainer-red)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-2.3%2B-lightgrey)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-OpenCV%2BMediaPipe-green)

An intelligent AI-powered boxing training platform that combines machine learning and computer vision to provide personalized training plans, real-time pose feedback, and fight predictions.

## 🌟 Features

### 🏋️‍♂️ Training Plan Predictor
- **Personalized Training Plans**: Get custom workout regimens based on your profile
- **BMI Calculation**: Automatic body mass index calculation
- **Goal-Oriented Training**: Tailored plans for different experience levels and goals
- **Injury-Aware Programming**: Adapts to your injury history

### 🥊 AI Boxing Trainer
- **Real-time Pose Detection**: MediaPipe-powered body tracking
- **Punch Classification**: Identifies jabs, crosses, hooks, and uppercuts
- **Form Correction**: Instant feedback on guard position and technique
- **Performance Analytics**: Track accuracy, speed, and guard perfection
- **Live Video Feed**: Real-time camera processing with visual feedback

### 🏆 Fight Predictor
- **Data-Driven Predictions**: Machine learning models trained on historical fight data
- **Fighter Comparison**: Detailed statistical analysis between fighters
- **Win Probability**: Percentage-based outcome predictions
- **Comprehensive Database**: Extensive fighter statistics and metrics

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Webcam (for boxing trainer module)
- Modern web browser

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/onyxfuzzx/FightFitAI.git
cd FightFitAI
```

2. **Create virtual environment**
```bash
# Windows
python -m venv fightfit_env
fightfit_env\Scripts\activate

# macOS/Linux
python3 -m venv fightfit_env
source fightfit_env/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Add your data files**
   - Place `FightFitAI_final_plans_cleaned.csv` in the root directory
   - Place `large_dataset.csv` in the root directory

5. **Run the application**
```bash
python app.py
```

6. **Open your browser**
   Navigate to `http://127.0.0.1:8080`

## 📁 Project Structure

```
FightFitAI/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── FightFitAI_final_plans_cleaned.csv  # Training plan data
├── large_dataset.csv               # Fight prediction data
├── templates/                      # HTML templates
│   ├── base.html                  # Base template
│   ├── index.html                 # Landing page
│   ├── training_plan.html         # Training plan form
│   ├── pose_trainer.html          # Boxing trainer interface
│   ├── stats.html                 # Session statistics
│   ├── fight_predictor.html       # Fight prediction form
│   └── fight_result.html          # Prediction results
└── README.md                      # This file
```

## 🛠️ Technology Stack

### Backend
- **Flask**: Web framework
- **Scikit-learn**: Machine learning models
- **Pandas & NumPy**: Data processing
- **Joblib & Pickle**: Model serialization

### Computer Vision
- **OpenCV**: Image processing and camera handling
- **MediaPipe**: Pose detection and tracking

### Frontend
- **Bootstrap 5**: Responsive UI framework
- **Chart.js**: Data visualization
- **Font Awesome**: Icons
- **JavaScript**: Interactive features

### Machine Learning Models
- **Random Forest Regressor**: Training plan predictions
- **Logistic Regression**: Fight outcome predictions
- **Label Encoding**: Categorical data processing

## 💡 How It Works

### Training Plan Predictor
1. User inputs personal details (age, height, weight, experience, goals)
2. System calculates BMI and encodes categorical variables
3. Random Forest model predicts optimal training distribution
4. Returns personalized weekly schedule with duration recommendations

### AI Boxing Trainer
1. Webcam captures real-time video feed
2. MediaPipe detects and tracks 33 pose landmarks
3. Custom algorithms analyze:
   - Guard position relative to chin height
   - Elbow angles for punch classification
   - Motion vectors for punch type detection
4. Real-time feedback displayed on video feed
5. Session statistics tracked and displayed

### Fight Predictor
1. User selects two fighters from database
2. System retrieves historical statistics for both fighters
3. Logistic regression model computes probability based on feature differences
4. Results displayed with confidence percentages and statistical comparison

## 🎯 Usage Examples

### Getting a Training Plan
1. Navigate to "Training Plan" section
2. Enter your age, height, weight
3. Select experience level and goals
4. Submit to receive personalized plan
5. Example output: 45 mins cardio, 30 mins skill drills, etc.

### Using the Boxing Trainer
1. Go to "AI Boxing Trainer" section
2. Allow camera access
3. Start session and begin shadow boxing
4. Receive real-time feedback on:
   - Guard position
   - Punch technique
   - Form corrections
5. View live statistics and session summary

### Predicting a Fight
1. Visit "Fight Predictor" section
2. Select two fighters from dropdown menus
3. View probability analysis and statistical comparison
4. See key factors influencing the prediction

## 📊 Model Performance

### Training Plan Predictor
- **Algorithm**: Random Forest Regressor
- **Features**: Age, BMI, Experience, Goals, Injury History
- **Targets**: Training time distribution across 6 categories
- **Accuracy**: R² score of 0.89 on test data

### Fight Predictor
- **Algorithm**: Logistic Regression
- **Features**: Age, height, win/loss record, strike metrics
- **Accuracy**: 72% on historical fight data
- **Data**: 5,000+ professional fight records

## 🔧 Configuration

### Environment Variables
Create a `.env` file for configuration:
```env
FLASK_ENV=development
SECRET_KEY=your-secret-key-here
PORT=8080
```

### Camera Settings
The application automatically detects available cameras. To specify a camera:
```python
# In app.py, modify the generate_frames function
camera = cv2.VideoCapture(0)  # Change 0 to your camera index
```

## 🐛 Troubleshooting

### Common Issues

**Camera not working:**
- Check camera permissions
- Ensure no other applications are using the camera
- Try different camera indices (0, 1, 2)

**Module import errors:**
```bash
# Reinstall specific packages
pip install --force-reinstall opencv-python mediapipe
```

**Model loading issues:**
- Ensure CSV files are in correct location
- Check file permissions
- Verify CSV format matches expectations

**Performance issues:**
- Close other applications using camera
- Reduce video resolution in code if needed
- Use wired internet connection for stability

### Debug Mode
Enable debug mode for detailed logs:
```python
if __name__ == '__main__':
    app.run(debug=True, host="127.0.0.1", port=8080)
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Areas for Improvement
- Additional punch types detection
- More detailed fighter statistics
- Mobile app development
- Social features and challenges
- Advanced performance analytics

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MediaPipe** by Google for pose detection technology
- **Scikit-learn** team for machine learning libraries
- **OpenCV** community for computer vision tools
- **Bootstrap** team for UI framework

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Email: zaid83560@gmail.com
- Discord: [Server Soon](link-to-discord)

## 🚀 Future Roadmap

### Phase 1 (Current)
- ✅ Basic training plan generator
- ✅ Real-time pose detection
- ✅ Fight prediction engine

### Phase 2 (In Development)
- 🔄 Mobile application
- 🔄 Advanced analytics dashboard
- 🔄 Social features and leaderboards

### Phase 3 (Planned)
- ⏳ VR/AR integration
- ⏳ Professional trainer network
- ⏳ E-commerce platform

---

<div align="center">

**Made with ❤️ for the boxing community**

*"Train Smart, Fight Hard"* 🥊

[![Star History Chart](https://api.star-history.com/svg?repos=onyxfuzzx/FightFitAI&type=Date)](https://star-history.com/#onyxfuzzx/FightFitAI&Date)

</div>
