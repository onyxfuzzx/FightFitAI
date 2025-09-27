from flask import Flask, render_template, request, jsonify, Response, session
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import joblib
import os
import warnings
import cv2
import math
import time
import threading
from collections import deque
import numpy as np
from mediapipe import solutions as mp_solutions

warnings.filterwarnings("ignore")

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'smartspar-secret-key-2023'

# =============================================================================
# MODULE 1: TRAINING PLAN PREDICTOR
# =============================================================================

# Load and preprocess the data for training plan predictor
def load_training_data():
    data = pd.read_csv("FightFitAI_final_plans_cleaned.csv")
    df = pd.DataFrame(data)
    return df

# Calculate BMI function
def calculate_bmi(height, weight):
    return weight / ((height / 100) ** 2)

# Train model function for training plan predictor
def train_training_model(df):
    # Encode categorical variables
    le_gender = LabelEncoder()
    le_experience = LabelEncoder()
    le_goal = LabelEncoder()
    le_injury = LabelEncoder()
    
    df['Gender_encoded'] = le_gender.fit_transform(df['Gender'])
    df['Experience_encoded'] = le_experience.fit_transform(df['Experience'])
    df['Goal_encoded'] = le_goal.fit_transform(df['Goal'])
    df['Injury_encoded'] = le_injury.fit_transform(df['Injury_History'])
    
    # Features and targets
    X = df[['Age', 'Gender_encoded', 'BMI', 'Experience_encoded', 'Goal_encoded', 'Injury_encoded']]
    y = df[['Cardio_Endurance', 'Skill_Drills', 'Strength_Conditioning', 'Agility_Mobility', 'Recovery', 'Goal_Duration_Months']]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, le_gender, le_experience, le_goal, le_injury

# Load or train model for training plan predictor
def get_training_model():
    model_path = 'fightfit_model.pkl'
    encoders_path = 'fightfit_encoders.pkl'
    
    if os.path.exists(model_path) and os.path.exists(encoders_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(encoders_path, 'rb') as f:
            encoders = pickle.load(f)
        le_gender, le_experience, le_goal, le_injury = encoders
    else:
        df = load_training_data()
        model, le_gender, le_experience, le_goal, le_injury = train_training_model(df)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        with open(encoders_path, 'wb') as f:
            pickle.dump((le_gender, le_experience, le_goal, le_injury), f)
    
    return model, le_gender, le_experience, le_goal, le_injury

# =============================================================================
# MODULE 2: POSE PROCESSOR (BOXING TRAINER)
# =============================================================================

mp_pose = mp_solutions.pose
mp_drawing = mp_solutions.drawing_utils

# Helper functions for pose processor
def angle(a, b, c):
    ang = math.degrees(
        math.atan2(c.y - b.y, c.x - b.x) -
        math.atan2(a.y - b.y, a.x - b.x)
    )
    ang = abs(ang)
    return 360 - ang if ang > 180 else ang

def avg_motion(buf):
    if len(buf) < 2:
        return 0, 0
    dx, dy = 0, 0
    for i in range(1, len(buf)):
        px, py = buf[i - 1]
        cx, cy = buf[i]
        dx += cx - px
        dy += cy - py
    return dx / (len(buf) - 1), dy / (len(buf) - 1)

# Rules for pose processor
def check_guard_up(landmarks):
    lw_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
    rw_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
    chin_y = (landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].y +
              landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].y) / 2
    guard_threshold = chin_y + 0.15
    if lw_y < guard_threshold and rw_y < guard_threshold:
        return True, ("[OK] Guard", (0, 200, 0))
    else:
        return False, ("[!] Guard Down", (0, 0, 255))

def detect_punch_type(landmarks, lw_buf, rw_buf, last_time, cooldown=0.4):
    lw = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    rw = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    le = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    re = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    now = time.time()
    left_elb_angle = angle(ls, le, lw)
    right_elb_angle = angle(rs, re, rw)
    ldx, ldy = avg_motion(lw_buf)
    rdx, rdy = avg_motion(rw_buf)

    # JAB
    if lw.z < le.z and left_elb_angle > 145 and lw.z - le.z < -0.02 and (now - last_time["jab"]) > cooldown:
        last_time["jab"] = now
        return "[OK] Jab", (0, 200, 0)

    # CROSS
    if rw.z < re.z and right_elb_angle > 145 and rw.z - re.z < -0.02 and (now - last_time["cross"]) > cooldown:
        last_time["cross"] = now
        return "[OK] Cross", (0, 200, 0)

    # HOOK
    if ((60 < left_elb_angle < 120 and abs(ldx) > abs(ldy) * 1.5 and abs(ldx) > 0.02) or
        (60 < right_elb_angle < 120 and abs(rdx) > abs(rdy) * 1.5 and abs(rdx) > 0.02)) \
        and (now - last_time["hook"]) > cooldown:
        last_time["hook"] = now
        return "[OK] Hook", (0, 200, 0)

    # UPPERCUT
    if ((left_elb_angle < 110 and lw.y > landmarks[mp_pose.PoseLandmark.NOSE.value].y
         and -ldy > abs(ldx) * 2 and -ldy > 0.05) or
        (right_elb_angle < 110 and rw.y > landmarks[mp_pose.PoseLandmark.NOSE.value].y
         and -rdy > abs(rdx) * 2 and -rdy > 0.05)) \
        and (now - last_time["upper"]) > cooldown:
        last_time["upper"] = now
        return "[OK] Uppercut", (0, 200, 0)

    return None

def draw_motion_vectors(img, lw_buf, rw_buf):
    h, w, _ = img.shape
    if len(lw_buf) >= 2:
        (x1, y1), (x2, y2) = lw_buf[-2], lw_buf[-1]
        cv2.arrowedLine(img, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (0, 255, 0), 2)
    if len(rw_buf) >= 2:
        (x1, y1), (x2, y2) = rw_buf[-2], rw_buf[-1]
        cv2.arrowedLine(img, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (255, 0, 0), 2)

# Pose Processor Class
class PoseProcessor:
    def __init__(self, session_id=None):
        self.session_id = session_id or str(time.time())
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.lw_buf = deque(maxlen=5)
        self.rw_buf = deque(maxlen=5)
        self.last_punch = None
        self.last_time = 0
        self.display_time = 1.0
        self.cooldowns = {"jab": 0, "cross": 0, "hook": 0, "upper": 0}

        # Stats
        self.total_punches = 0
        self.valid_punches = 0
        self.guard_warnings = 0
        self.punch_counts = {"Jab": 0, "Cross": 0, "Hook": 0, "Uppercut": 0}
        self.session_start = time.time()

        # State
        self.guard_ok_prev = True
        self.last_counted_punch = None
        self.last_count_time = 0.0
        self.count_cooldown = 0.5
        self.feedback_text = "Starting up..."
        self.feedback_color = (255, 255, 255)
        self.guard_up_time = 0
        self.total_tracking_time = 0
        self.last_update_time = time.time()
        
        # Thread safety
        self.lock = threading.Lock()

    def _extract_name(self, msg_text):
        return msg_text.partition('] ')[2] if '] ' in msg_text else msg_text
    
    def update_guard_time(self, guard_ok):
        now = time.time()
        time_elapsed = now - self.last_update_time
        self.total_tracking_time += time_elapsed
        
        if guard_ok:
            self.guard_up_time += time_elapsed
        
        self.last_update_time = now

    def process_frame(self, frame):
        img = frame
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        feedback = []

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            lw = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
            rw = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            self.lw_buf.append((lw.x, lw.y))
            self.rw_buf.append((rw.x, rw.y))

            guard_ok, guard_msg = check_guard_up(results.pose_landmarks.landmark)
            feedback.append(guard_msg)
            
            self.update_guard_time(guard_ok)

            if not guard_ok and self.guard_ok_prev:
                with self.lock:
                    self.guard_warnings += 1
            self.guard_ok_prev = guard_ok

            if guard_ok:
                punch = detect_punch_type(results.pose_landmarks.landmark,
                                          self.lw_buf, self.rw_buf,
                                          self.cooldowns, cooldown=0.4)
                now = time.time()
                if punch:
                    msg_text, color = punch
                    name = self._extract_name(msg_text)

                    self.last_punch = punch
                    self.last_time = now

                    if (now - self.last_count_time) > self.count_cooldown or name != self.last_counted_punch:
                        with self.lock:
                            self.total_punches += 1
                            self.valid_punches += 1
                            if name in self.punch_counts:
                                self.punch_counts[name] += 1
                            self.last_counted_punch = name
                            self.last_count_time = now

                if self.last_punch and (time.time() - self.last_time < self.display_time):
                    feedback.append(self.last_punch)

            draw_motion_vectors(img, self.lw_buf, self.rw_buf)
            
            # Update feedback text for display
            if feedback:
                self.feedback_text, self.feedback_color = feedback[-1]
            else:
                self.feedback_text = "Ready for training"
                self.feedback_color = (255, 255, 255)
                
            # Draw feedback on frame
            y_offset = 30
            for msg, color in feedback:
                cv2.putText(img, msg, (10, y_offset), cv2.FONT_HERSHEY_DUPLEX,
                            0.7, color, 2, cv2.LINE_AA)
                y_offset += 30

        return img

    def get_stats(self):
        with self.lock:
            acc = (self.valid_punches / self.total_punches * 100) if self.total_punches else 0
            session_duration = time.time() - self.session_start
            guard_perfection = (self.guard_up_time / self.total_tracking_time * 100) if self.total_tracking_time > 0 else 0
            return {
                "total_punches": self.total_punches,
                "valid_punches": self.valid_punches,
                "accuracy": round(acc, 1),
                "guard_warnings": self.guard_warnings,
                "guard_perfection": round(guard_perfection, 1),
                "punch_counts": self.punch_counts,
                "session_duration": round(session_duration, 1),
                "punches_per_minute": round(self.total_punches / (session_duration / 60), 1) if session_duration > 0 else 0
            }
    
    def reset_stats(self):
        with self.lock:
            self.total_punches = 0
            self.valid_punches = 0
            self.guard_warnings = 0
            self.punch_counts = {"Jab": 0, "Cross": 0, "Hook": 0, "Uppercut": 0}
            self.session_start = time.time()
            return {"status": "success", "message": "Stats reset successfully"}

# =============================================================================
# MODULE 3: FIGHT PREDICTOR
# =============================================================================

# Load and preprocess data for fight predictor
def load_fight_data():
    data = pd.read_csv("large_dataset.csv")
    return pd.DataFrame(data)

# Initialize data and model for fight predictor
fight_df = load_fight_data()
fight_model = None
fighter_db = None

def train_fight_model():
    global fight_model, fighter_db
    
    # List of pre-fight stats to use for differences
    features_to_diff = [
        'age', 'height', 'wins_total', 'losses_total',
        'SLpM_total', 'SApM_total'
    ]

    # Create a new DataFrame for training with difference features
    df_diff = pd.DataFrame()

    # Compute differences for each feature
    for feat in features_to_diff:
        df_diff['diff_' + feat] = fight_df['r_' + feat] - fight_df['b_' + feat]

    # Target variable: 1 if red wins, 0 if blue wins
    df_diff['target'] = (fight_df['winner'] == 'Red').astype(int)

    # Drop rows with missing values if any
    df_diff.dropna(inplace=True)

    # Split the data into features and target
    X = df_diff.drop('target', axis=1)
    y = df_diff['target']

    # Train a logistic regression model
    fight_model = LogisticRegression(random_state=42)
    fight_model.fit(X, y)

    # Build a database of fighter average stats
    red_fighters = fight_df[['r_fighter'] + ['r_' + feat for feat in features_to_diff]]
    blue_fighters = fight_df[['b_fighter'] + ['b_' + feat for feat in features_to_diff]]

    # Rename columns to remove prefix
    red_fighters.columns = ['fighter'] + features_to_diff
    blue_fighters.columns = ['fighter'] + features_to_diff

    # Combine red and blue data
    all_fighters = pd.concat([red_fighters, blue_fighters], ignore_index=True)

    # Group by fighter name and compute mean stats
    fighter_db = all_fighters.groupby('fighter').mean().reset_index()
    
    # Save model and fighter database
    joblib.dump(fight_model, 'fight_model.pkl')
    fighter_db.to_csv('fighter_database.csv', index=False)

# =============================================================================
# GLOBAL INITIALIZATION
# =============================================================================

# Initialize models
training_model, le_gender, le_experience, le_goal, le_injury = get_training_model()
train_fight_model()

# Session management for pose processor
pose_processors = {}

def get_pose_processor(session_id):
    if session_id not in pose_processors:
        pose_processors[session_id] = PoseProcessor(session_id)
    return pose_processors[session_id]

# =============================================================================
# ROUTES
# =============================================================================

@app.route('/')
def index():
    return render_template('index.html')

# =============================================================================
# TRAINING PLAN PREDICTOR ROUTES
# =============================================================================

@app.route('/training-plan')
def training_plan_index():
    return render_template('training_plan.html', 
                          experience_levels=list(le_experience.classes_),
                          goals=list(le_goal.classes_),
                          injury_history=list(le_injury.classes_))

@app.route('/training-plan/predict', methods=['POST'])
def training_plan_predict():
    data = request.get_json()
    age = int(data['age'])
    height = int(data['height'])
    weight = int(data['weight'])
    experience = data['experience']
    goal = data['goal']
    injury_history = data['injury_history']
    
    # Calculate BMI
    bmi = calculate_bmi(height, weight)
    
    # Encode inputs
    gender_encoded = le_gender.transform(['Male'])[0]  # Default to Male
    experience_encoded = le_experience.transform([experience])[0]
    goal_encoded = le_goal.transform([goal])[0]
    injury_encoded = le_injury.transform([injury_history])[0]
    
    # Prepare input data
    input_data = np.array([[age, gender_encoded, bmi, experience_encoded, goal_encoded, injury_encoded]])
    
    # Make prediction
    prediction = training_model.predict(input_data)[0]
    
    # Format results
    results = {
        'bmi': round(bmi, 1),
        'cardio': round(prediction[0]),
        'skill': round(prediction[1]),
        'strength': round(prediction[2]),
        'agility': round(prediction[3]),
        'recovery': round(prediction[4]),
        'duration': round(prediction[5])
    }
    
    return jsonify(results)

# =============================================================================
# POSE PROCESSOR ROUTES
# =============================================================================

def generate_frames(session_id):
    processor = get_pose_processor(session_id)
    
    # Try different camera indices
    camera_indices = [0, 1, 2]
    camera = None
    
    for camera_index in camera_indices:
        try:
            camera = cv2.VideoCapture(camera_index)
            if camera.isOpened():
                print(f"Camera found at index {camera_index}")
                break
            else:
                camera.release()
        except:
            pass
    
    if camera is None or not camera.isOpened():
        print("No camera found. Using test pattern.")
        while True:
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(img, "No camera detected", (100, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, "Please check your camera connection", (50, 250), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)
    
    try:
        while True:
            success, frame = camera.read()
            if not success:
                break
            else:
                processed_frame = processor.process_frame(frame)
                ret, buffer = cv2.imencode('.jpg', processed_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        if camera:
            camera.release()

@app.route('/pose-trainer')
def pose_trainer_index():
    session_id = session.get('session_id')
    if not session_id:
        session_id = str(time.time())
        session['session_id'] = session_id
    return render_template('pose_trainer.html', session_id=session_id)

@app.route('/pose-trainer/video_feed')
def pose_trainer_video_feed():
    session_id = session.get('session_id', 'default')
    return Response(generate_frames(session_id), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/pose-trainer/stats')
def pose_trainer_stats():
    session_id = session.get('session_id', 'default')
    processor = get_pose_processor(session_id)
    stats_data = processor.get_stats()
    return jsonify(stats_data)

@app.route('/pose-trainer/end_session')
def pose_trainer_end_session():
    session_id = session.get('session_id', 'default')
    processor = get_pose_processor(session_id)
    stats_data = processor.get_stats()
    processor.reset_stats()
    return render_template('stats.html', stats=stats_data)

@app.route('/pose-trainer/reset_stats')
def pose_trainer_reset_stats():
    session_id = session.get('session_id', 'default')
    processor = get_pose_processor(session_id)
    result = processor.reset_stats()
    return jsonify(result)

# =============================================================================
# FIGHT PREDICTOR ROUTES
# =============================================================================

@app.route('/fight-predictor')
def fight_predictor_index():
    fighters = sorted(pd.concat([fight_df['r_fighter'], fight_df['b_fighter']]).unique())
    return render_template('fight_predictor.html', fighters=fighters)

@app.route('/fight-predictor/predict', methods=['POST'])
def fight_predictor_predict():
    fighter_a = request.form['fighter_a']
    fighter_b = request.form['fighter_b']
    
    # Get stats for both fighters
    stats_a = fighter_db[fighter_db['fighter'] == fighter_a].iloc[0][1:]
    stats_b = fighter_db[fighter_db['fighter'] == fighter_b].iloc[0][1:]
    
    # Compute differences: fighter_a as red, fighter_b as blue
    diff = stats_a.values - stats_b.values
    diff_df = pd.DataFrame([diff], columns=['diff_age', 'diff_height', 'diff_wins_total', 
                                          'diff_losses_total', 'diff_SLpM_total', 'diff_SApM_total'])
    
    # Predict probability
    prob = fight_model.predict_proba(diff_df)[0][1]  # Probability that fighter_a wins
    
    # Prepare data for visualization
    comparison_data = []
    features = ['Age', 'Height', 'Total Wins', 'Total Losses', 'Strikes Landed per Min', 'Strikes Absorbed per Min']
    
    for i, feature in enumerate(features):
        comparison_data.append({
            'feature': feature,
            'fighter_a': round(stats_a.iloc[i], 2),
            'fighter_b': round(stats_b.iloc[i], 2)
        })
    
    if prob > 0.5:
        winner = fighter_a
        winner_prob = round(prob * 100, 2)
        loser_prob = round((1 - prob) * 100, 2)
    else:
        winner = fighter_b
        winner_prob = round((1 - prob) * 100, 2)
        loser_prob = round(prob * 100, 2)
    
    return render_template('fight_result.html', 
                         fighter_a=fighter_a, 
                         fighter_b=fighter_b,
                         winner=winner,
                         winner_prob=winner_prob,
                         loser_prob=loser_prob,
                         comparison_data=comparison_data)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)