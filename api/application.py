from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO
from ultralytics import YOLO
import cv2
import numpy as np
from pymongo import MongoClient
import threading
import time
import base64
import uuid
import bcrypt
import tempfile
import jwt
import os
from dotenv import load_dotenv
import signal
import sys
from fpdf import FPDF
from io import BytesIO
from datetime import datetime, timedelta
from shapely.geometry import Point, Polygon
from bson import ObjectId
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000", "https://perimeterbreachdetection.vercel.app"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["Authorization"],
        "supports_credentials": True
    }
})
socketio = SocketIO(app, cors_allowed_origins="*")
os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'] = '10000'

# Load environment variables from .env file
load_dotenv()

# MongoDB configuration
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["test"]
users_collection = db["users"]
stream_collection = db["streams"]

# JWT Secret for token generation
JWT_SECRET = os.getenv("JWT_SECRET")

# Email configuration
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
# Load YOLOv8 model for human detection
model = YOLO("yolov8n.pt")

# Store active streams
active_streams = {}

def send_email(to_email, subject, body):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = to_email
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

def verify_token(auth_header):
    if not auth_header:
        return None
    try:
        token = auth_header.replace('Bearer ', '') if auth_header.startswith('Bearer ') else auth_header
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


class StreamReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Stream Activity Report', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')

def generate_stream_report(stream_data):
    pdf = StreamReport()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # Stream Information
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Stream Information', 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 10, f"Stream ID: {stream_data.get('stream_id', 'Unknown')}", 0, 1)
    pdf.cell(0, 10, f"Username: {stream_data.get('username', 'Unknown')}", 0, 1)

    created_at = stream_data.get('created_at')
    if created_at:
        if isinstance(created_at, datetime):
            created_at_str = created_at.strftime('%Y-%m-%d %H:%M:%S')
        else:
            created_at_str = created_at
        pdf.cell(0, 10, f"Created: {created_at_str}", 0, 1)
    else:
        pdf.cell(0, 10, "Created: Unknown", 0, 1)
    pdf.ln(10)

    # Breach Statistics
    breaches = stream_data.get('breaches', [])
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Breach Statistics', 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 10, f"Total Breaches: {len(breaches)}", 0, 1)
    
    if breaches:
        first_breach = breaches[0].get('timestamp')
        last_breach = breaches[-1].get('timestamp')
        if first_breach:
            first_breach_str = first_breach.strftime('%Y-%m-%d %H:%M:%S') if isinstance(first_breach, datetime) else first_breach
            pdf.cell(0, 10, f"First Breach: {first_breach_str}", 0, 1)
        if last_breach:
            last_breach_str = last_breach.strftime('%Y-%m-%d %H:%M:%S') if isinstance(last_breach, datetime) else last_breach
            pdf.cell(0, 10, f"Last Breach: {last_breach_str}", 0, 1)
    pdf.ln(10)

    # Detailed Breach List
    if breaches:
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Breach Details', 0, 1)
        pdf.set_font('Arial', '', 10)
        
        for i, breach in enumerate(breaches, 1):
            pdf.cell(0, 10, f"Breach #{i}", 0, 1)
            timestamp = breach.get('timestamp')
            confidence = breach.get('confidence', 'N/A')

            if timestamp:
                timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S') if isinstance(timestamp, datetime) else timestamp
                pdf.cell(0, 10, f"Timestamp: {timestamp_str}", 0, 1)
            pdf.cell(0, 10, f"Confidence: {confidence:.2f}" if isinstance(confidence, float) else f"Confidence: {confidence}", 0, 1)

            if 'image' in breach:
                try:
                    image_data = base64.b64decode(breach['image'])
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_image:
                        temp_image.write(image_data)
                        temp_image_path = temp_image.name
                    pdf.image(temp_image_path, x=10, y=pdf.get_y(), w=100)
                    os.remove(temp_image_path)  # Clean up temporary file
                    pdf.ln(80)  # Space for image
                except Exception as e:
                    print(f"Error adding image to PDF: {e}")
            
            pdf.ln(5)

            if pdf.get_y() > 250:
                pdf.add_page()

    return pdf
class StreamProcessor:
    def __init__(self, stream_id, rtsp_url, user_id, username):
        self.stream_id = stream_id
        self.rtsp_url = rtsp_url
        self.user_id = user_id
        self.username = username
        self.running = False
        self.perimeter = None
        self.frame_counter = 0
        self.SKIP_FRAMES = 25
        self.last_breach_time = None
        self.MIN_BREACH_INTERVAL = 2  # Minimum seconds between breach recordings
        self.frame_width = None  # Store actual frame width
        self.frame_height = None  # Store actual frame height
        self.container_width = None  # Store container width from frontend
        self.container_height = None  # Store container height from frontend

        # Create initial stream document
        stream_doc = {
            "stream_id": stream_id,
            "user_id": user_id,
            "username": username,
            "rtsp_url": rtsp_url,
            "created_at": datetime.utcnow(),
            "breaches": []  # Initialize empty breaches array
        }
        
        try:
            # First check if stream exists
            existing_stream = stream_collection.find_one({"stream_id": stream_id})
            if existing_stream:
                print(f"Stream already exists with ID: {stream_id}")
            else:
                # Insert new stream document
                result = stream_collection.insert_one(stream_doc)
                print(f"New stream created with ID: {stream_id}, MongoDB _id: {result.inserted_id}")
        except Exception as e:
            print(f"Error initializing stream document: {e}")
            print(traceback.format_exc())

        # Create initial stream document
        stream_doc = {
            "stream_id": stream_id,
            "user_id": user_id,
            "username": username,
            "rtsp_url": rtsp_url,
            "created_at": datetime.utcnow(),
            "breaches": []
        }
        
        try:
            # Use upsert to handle both insert and update cases
            stream_collection.update_one(
                {"stream_id": stream_id},
                {"$setOnInsert": stream_doc},
                upsert=True
            )
            print(f"StreamProcessor initialized for stream ID: {stream_id}")
        except Exception as e:
            print(f"Error initializing stream document: {e}")

    def set_container_dimensions(self, width, height):
        """Set the frontend container dimensions for scaling"""
        self.container_width = width
        self.container_height = height
        print(f"Container dimensions set: {width}x{height}")

    def scale_perimeter_points(self, points):
        """Scale perimeter points from container coordinates to frame coordinates"""
        if not self.frame_width or not self.frame_height or not self.container_width or not self.container_height:
            return points
        
        scaled_points = []
        for x, y in points:
            # Scale x and y coordinates from container dimensions to frame dimensions
            scaled_x = (x / self.container_width) * self.frame_width
            scaled_y = (y / self.container_height) * self.frame_height
            scaled_points.append([scaled_x, scaled_y])
        
        print(f"Scaled points from {points} to {scaled_points}")
        return scaled_points

        

    def save_breach_event(self, frame, box, confidence):
        current_time = datetime.utcnow()
        
        # Check if enough time has passed since last breach
        if (self.last_breach_time and 
            (current_time - self.last_breach_time).total_seconds() < self.MIN_BREACH_INTERVAL):
            return

        try:
            # Print debug information
            print(f"Attempting to save breach for stream {self.stream_id}")
            print(f"Current time: {current_time}")
            print(f"Box coordinates: {box}")
            print(f"Confidence: {confidence}")

            # Encode frame with reduced quality
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            image_base64 = base64.b64encode(buffer).decode('utf-8')

            # Create breach event
            breach_event = {
                "timestamp": current_time,
                "bounding_box": [float(x) for x in box],
                "confidence": float(confidence),
                "image": image_base64
            }

            # Update stream document with new breach using ObjectId
            result = stream_collection.update_one(
                {"stream_id": self.stream_id},
                {
                    "$push": {
                        "breaches": breach_event
                    }
                }
            )

            # Log the update result
            print(f"MongoDB update result - matched: {result.matched_count}, modified: {result.modified_count}")
            print(f"Stream ID being used: {self.stream_id}")

            if result.modified_count > 0:
                self.last_breach_time = current_time
                
                # Emit breach event via Socket.IO
                socketio.emit(f'breach-{self.stream_id}', {
                    "timestamp": current_time.isoformat(),
                    "confidence": float(confidence),
                    "image": image_base64
                })
                print(f"Breach event saved and emitted for stream {self.stream_id}")
            else:
                # If the update failed, let's check if the stream exists
                stream_check = stream_collection.find_one({"stream_id": self.stream_id})
                if stream_check:
                    print(f"Stream exists but update failed. Stream document: {stream_check}")
                else:
                    print(f"Stream not found with ID: {self.stream_id}")

        except Exception as e:
            print(f"Error saving breach event: {e}")
            print(f"Error type: {type(e)}")
            print(f"Error details: {str(e)}")
            # Print the full traceback for better debugging
            import traceback
            print(traceback.format_exc())



    def detect_humans(self, frame):
        try:
            height, width = frame.shape[:2]
            new_width = 640
            new_height = int(height * (new_width / width))
            resized_frame = cv2.resize(frame, (new_width, new_height))

            # Add debug print
            print("Running YOLO detection...")
            results = model(resized_frame)
            
            # Print detection results
            print(f"Number of detections: {len(results[0].boxes)}")
            for result in results[0].boxes:
                print(f"Class: {result.cls}, Confidence: {result.conf}")

            detections = []
            scale_x = width / new_width
            scale_y = height / new_height

            for result in results[0].boxes:
                if result.cls == 0 and result.conf > 0.5:
                    x1, y1, x2, y2 = map(float, result.xyxy[0])
                    x1, x2 = x1 * scale_x, x2 * scale_x
                    y1, y2 = y1 * scale_y, y2 * scale_y
                    detections.append((
                        (int(x1), int(y1), int(x2-x1), int(y2-y1)), 
                        float(result.conf)
                    ))
                    print(f"Human detected with confidence: {result.conf}")

            return detections
        except Exception as e:
            print(f"Error in detect_humans: {e}")
            return []

    def process_stream(self):
        try:
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            
            if not cap.isOpened():
                raise Exception(f"Failed to open video stream: {self.rtsp_url}")
            
            # Get frame dimensions
            self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Frame dimensions: {self.frame_width}x{self.frame_height}")

            self.running = True
            retry_count = 0
            max_retries = 5

            while self.running:
                ret, frame = cap.read()
                if not ret:
                    retry_count += 1
                    if retry_count >= max_retries:
                        break
                    time.sleep(0.1)
                    continue

                retry_count = 0
                self.frame_counter += 1
                
                # Skip frames for performance
                if self.frame_counter % (self.SKIP_FRAMES + 1) != 0:
                    continue

                try:
                    # Only process if perimeter is defined
                    if self.perimeter and len(self.perimeter) > 2:
                        human_detections = self.detect_humans(frame)
                        
                        if human_detections:
                            print("Human detected without perimeter check!")
                            # Create perimeter polygon once
                            print(f"Perimeter points: {self.perimeter}")
                            perimeter_points = np.array(self.perimeter, np.int32)
                            cv2.polylines(frame, [perimeter_points], True, (0, 255, 0), 2)
                            perimeter_polygon = Polygon(self.perimeter)

                            for (box, confidence) in human_detections:
                                x, y, w, h = box
                                # Check if person's center is inside perimeter
                                center_point = Point(x + w/2, y + h/2)
                                
                                # Print center point and containment check
                                print(f"Person center point: ({x + w/2}, {y + h/2})")
                                print(f"Inside perimeter: {perimeter_polygon.contains(center_point)}")

                                if perimeter_polygon.contains(center_point):
                                    # Draw bounding box with different colors based on confidence
                                    color = (0, 165, 255) if confidence < 0.5 else (0, 0, 255)
                                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                                    cv2.putText(frame, f"Person: {confidence:.2f}", 
                                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                            0.5, color, 2)
                                    
                                    # Always save breach for confidence > 0.5
                                    self.save_breach_event(frame, box, confidence)
                                    print(f"Human detected inside perimeter with confidence: {confidence:.2f}")

                    # Send frame to client
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    socketio.emit(f'stream-{self.stream_id}', {'image': frame_base64})

                except Exception as e:
                    print(f"Error processing frame: {e}")
                    continue

                time.sleep(0.01)

        except Exception as e:
            print(f"Error in process_stream: {e}")
            socketio.emit(f'stream-error-{self.stream_id}', {'error': str(e)})
        finally:
            if cap and cap.isOpened():
                cap.release()
            self.running = False
            
# Signal handler to gracefully shut down threads and Socket.IO
def cleanup_on_exit(signal, frame):
    print("Shutting down gracefully...")
    for stream_id, processor in active_streams.items():
        processor.running = False  # Stop all stream processors
    socketio.stop()  # Stop the Socket.IO server
    print("Socket.IO server stopped.")
    sys.exit(0)

# Attach signal handler
signal.signal(signal.SIGINT, cleanup_on_exit)
signal.signal(signal.SIGTERM, cleanup_on_exit)


# Authentication Routes
@app.route("/api/login", methods=["POST", "OPTIONS"])
def login():
    if request.method == "OPTIONS":
        return jsonify({"success": True}), 200

    try:
        data = request.json
        email = data.get("email")
        password = data.get("password")

        if not email or not password:
            return jsonify({"error": "Email and password are required"}), 400

        db_user = users_collection.find_one({"email": email})
        if not db_user:
            return jsonify({"error": "Invalid email or password"}), 401

        if not bcrypt.checkpw(password.encode(), db_user["password"].encode()):
            return jsonify({"error": "Invalid email or password"}), 401

        token = jwt.encode(
            {"id": str(db_user["_id"]), "exp": datetime.utcnow() + timedelta(hours=720)},
            JWT_SECRET,
            algorithm="HS256",
        )
        
        user_info = {
            "id": str(db_user["_id"]),
            "email": db_user["email"],
            "name": db_user.get("name", ""),
        }
        
        return jsonify({
            "success": True,
            "token": token,
            "user": user_info
        }), 200

    except Exception as e:
        print(f"Error in login: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/signup", methods=["POST"])
def signup():
    try:
        data = request.json
        email = data.get("email")
        password = data.get("password")
        name = data.get("name")
        
        if not email or not password or not name:
            return jsonify({"error": "All fields are required"}), 400
            
        if users_collection.find_one({"email": email}):
            return jsonify({"error": "Email already registered"}), 400
            
        hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
        
        user = {
            "email": email,
            "password": hashed_password.decode(),
            "name": name,
            "created_at": datetime.utcnow()
        }
        
        result = users_collection.insert_one(user)
        
        token = jwt.encode(
            {"id": str(result.inserted_id), "exp": datetime.utcnow() + timedelta(hours=1)},
            JWT_SECRET,
            algorithm="HS256"
        )
        
        user_info = {
            "id": str(result.inserted_id),
            "email": email,
            "name": name
        }
        
        return jsonify({
            "success": True,
            "token": token,
            "user": user_info
        }), 201
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/forgot-password", methods=["POST"])
def forgot_password():
    try:
        data = request.json
        email = data.get("email")
        
        if not email:
            return jsonify({"error": "Email is required"}), 400
            
        user = users_collection.find_one({"email": email})
        if not user:
            return jsonify({"error": "Email not found"}), 404
            
        reset_token = jwt.encode(
            {"email": email, "exp": datetime.utcnow() + timedelta(hours=1)},
            JWT_SECRET,
            algorithm="HS256"
        )
        
        users_collection.update_one(
            {"email": email},
            {"$set": {"resetPasswordToken": reset_token}}
        )
        
        reset_link = f"https://perimeterbreachdetection.vercel.app/reset-password?token={reset_token}"
        # reset_link = f"http://localhost:3000/reset-password?token={reset_token}"
        email_body = f"Click the following link to reset your password: {reset_link}"
        
        if send_email(email, "Password Reset Request", email_body):
            return jsonify({"success": True, "message": "Password reset email sent"}), 200
        else:
            return jsonify({"error": "Failed to send reset email"}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/reset-password", methods=["POST"])
def reset_password():
    try:
        data = request.json
        token = data.get("token")
        new_password = data.get("new_password")
        
        if not token or not new_password:
            return jsonify({"error": "Token and new password are required"}), 400
            
        try:
            # Verify the token
            payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token has expired"}), 400
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 400
        
        email = payload.get("email")
        
        user = users_collection.find_one({"email": email})
        if not user or user.get("resetPasswordToken") != token:
            return jsonify({"error": "Invalid or expired token"}), 400
        
        # Hash the new password
        hashed_password = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt())
        
        # Update the user's password and clear the resetPasswordToken field
        users_collection.update_one(
            {"email": email},
            {
                "$set": {"password": hashed_password.decode()},
                "$unset": {"resetPasswordToken": ""}
            }
        )
        
        return jsonify({"success": True, "message": "Password reset successful"}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    


@app.route("/api/user-info", methods=["GET", "OPTIONS"])
def get_user_info():
    if request.method == "OPTIONS":
        return jsonify({"success": True}), 200

    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({"error": "No token provided"}), 401
            
        payload = verify_token(auth_header)
        if not payload:
            return jsonify({"error": "Invalid or expired token"}), 401
            
        user = users_collection.find_one({"_id": ObjectId(payload["id"])})
        if not user:
            return jsonify({"error": "User not found"}), 404
            
        user_info = {
            "id": str(user["_id"]),
            "email": user["email"],
            "name": user.get("name", ""),
            "created_at": user.get("created_at", datetime.utcnow())
        }
        
        return jsonify({"success": True, "user": user_info}), 200
        
    except Exception as e:
        print(f"Error in get_user_info: {e}")
        return jsonify({"error": str(e)}), 500

# Stream Routes
@app.route('/api/start-stream', methods=['POST', 'OPTIONS'])
def start_stream():
    if request.method == "OPTIONS":
        return jsonify({"success": True}), 200
        
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({"error": "No token provided"}), 401
            
        payload = verify_token(auth_header)
        if not payload:
            return jsonify({"error": "Invalid or expired token"}), 401
        
        data = request.json
        rtsp_url = data.get('rtsp_url')
        
        if not rtsp_url:
            return jsonify({"error": "RTSP URL is required"}), 400
            
        # Get user info from database
        user = users_collection.find_one({"_id": ObjectId(payload["id"])})
        if not user:
            return jsonify({"error": "User not found"}), 404
            
        stream_id = str(uuid.uuid4())
        
        try:
            # Create new stream processor with user info
            processor = StreamProcessor(
                stream_id=stream_id,
                rtsp_url=rtsp_url,
                user_id=str(user["_id"]),
                username=user.get("name", "Unknown")
            )
            
            # Start stream processing in a separate thread
            active_streams[stream_id] = processor
            thread = threading.Thread(target=processor.process_stream)
            thread.daemon = True
            thread.start()
            
            return jsonify({
                "streamId": stream_id,
                "userId": str(user["_id"]),
                "username": user.get("name", "Unknown")
            }), 200
            
        except Exception as e:
            print(f"Error creating stream processor: {e}")
            return jsonify({"error": f"Failed to initialize stream: {str(e)}"}), 500
        
    except Exception as e:
        print(f"Error in start_stream: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/stop-stream/<stream_id>', methods=['POST', 'OPTIONS'])
def stop_stream(stream_id):
    if request.method == "OPTIONS":
        return jsonify({"success": True}), 200
        
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({"error": "No token provided"}), 401
            
        payload = verify_token(auth_header)
        if not payload:
            return jsonify({"error": "Invalid or expired token"}), 401
            
        if stream_id in active_streams:
            processor = active_streams[stream_id]
            if processor.user_id == payload["id"]:
                processor.running = False
                del active_streams[stream_id]
                return jsonify({"message": "Stream stopped successfully"}), 200
            else:
                return jsonify({"error": "Unauthorized to stop this stream"}), 403
        return jsonify({"error": "Stream not found"}), 404
        
    except Exception as e:
        print(f"Error in stop_stream: {e}")
        return jsonify({"error": str(e)}), 500
    

@app.route('/api/add-perimeter', methods=['POST', 'OPTIONS'])
def add_perimeter():
    if request.method == "OPTIONS":
        return jsonify({"success": True}), 200

    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({"error": "No token provided"}), 401

        payload = verify_token(auth_header)
        if not payload:
            return jsonify({"error": "Invalid or expired token"}), 401

        data = request.json
        stream_id = data.get('stream_id')
        points = data.get('points')

        container_width = data.get('containerWidth')  # Get container dimensions from frontend
        container_height = data.get('containerHeight')

        if not stream_id or not points or not container_width or not container_height:
            return jsonify({"error": "Stream ID, points, and container dimensions are required"}), 400

        if not isinstance(points, list) or not all(isinstance(p, list) and len(p) == 2 for p in points):
            return jsonify({"error": "Points must be a list of [x, y] coordinates"}), 400

        # Verify the stream belongs to the user
        stream = stream_collection.find_one({"stream_id": stream_id})
        if not stream:
            return jsonify({"error": "Stream not found in database"}), 404

        if stream["user_id"] != payload["id"]:
            return jsonify({"error": "Unauthorized to modify this stream"}), 403

        if stream_id in active_streams:
            processor = active_streams[stream_id]
            # Set container dimensions
            processor.set_container_dimensions(container_width, container_height)
            # Scale points before setting perimeter
            scaled_points = processor.scale_perimeter_points(points)
            processor.perimeter = scaled_points
            return jsonify({"message": "Perimeter added successfully"}), 200

        return jsonify({"error": "Stream is not active"}), 404

    except Exception as e:
        print(f"Error in add_perimeter: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/get-stream-breaches/<stream_id>', methods=['GET'])
def get_stream_breaches(stream_id):
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({"error": "No token provided"}), 401
            
        payload = verify_token(auth_header)
        if not payload:
            return jsonify({"error": "Invalid or expired token"}), 401
            
        stream = stream_collection.find_one({"stream_id": stream_id})
        if not stream:
            return jsonify({"error": "Stream not found"}), 404
            
        # Verify the stream belongs to the authenticated user
        if stream["user_id"] != payload["id"]:
            return jsonify({"error": "Unauthorized to access this stream"}), 403
            
        breaches = stream.get('breaches', [])
        # Convert ObjectId and datetime objects to strings
        for breach in breaches:
            breach['timestamp'] = breach['timestamp'].isoformat()
        
        return jsonify({"breaches": breaches}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/get-user-streams', methods=['GET'])
def get_user_streams():
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({"error": "No token provided"}), 401
            
        payload = verify_token(auth_header)
        if not payload:
            return jsonify({"error": "Invalid or expired token"}), 401
            
        # Get all streams for the authenticated user
        streams = list(stream_collection.find({"user_id": payload["id"]}))
        
        # Convert ObjectId and datetime objects to strings
        for stream in streams:
            stream['_id'] = str(stream['_id'])
            stream['created_at'] = stream['created_at'].isoformat()
            # Remove breach images from the response to reduce payload size
            if 'breaches' in stream:
                for breach in stream['breaches']:
                    breach['timestamp'] = breach['timestamp'].isoformat()
                    breach.pop('image', None)
        
        return jsonify({
            "success": True,
            "streams": streams
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/generate-report/<stream_id>', methods=['GET', 'POST', 'OPTIONS'])
def generate_report(stream_id):
    if request.method == "OPTIONS":
        return jsonify({"success": True}), 200

    try:
        # Check Authorization Header
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({"error": "No token provided"}), 401

        # Validate token
        payload = verify_token(auth_header)
        if not payload:
            return jsonify({"error": "Invalid or expired token"}), 401

        # Fetch stream data
        stream = stream_collection.find_one({"stream_id": stream_id})
        if not stream:
            return jsonify({"error": "Stream not found"}), 404

        print(f"Fetched stream data: {stream}")  # Debug log

        # Verify user ownership
        if stream.get("user_id") != payload.get("id"):
            return jsonify({"error": "Unauthorized to access this stream"}), 403

        # Generate PDF
        pdf = generate_stream_report(stream)

        # Save PDF to BytesIO
        pdf_buffer = BytesIO()
        pdf.output(pdf_buffer)
        pdf_buffer.seek(0)

        # Encode to base64
        pdf_base64 = base64.b64encode(pdf_buffer.read()).decode()

        # Generate filename
        filename = f"stream_report_{stream_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        return jsonify({
            "success": True,
            "pdf": pdf_base64,
            "filename": filename
        }), 200

    except Exception as e:
        # Log the error for debugging purposes
        print(f"Error generating report: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500

if __name__ == "__main__":
    try:
        socketio.run(app, host='0.0.0.0', port=8000, debug=False)
    except Exception as e:
        print(f"Error running application: {e}")
    finally:
        cleanup_on_exit(None, None)