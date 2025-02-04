from flask import Flask, jsonify, request
from flask import Flask, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO
from mangum import Mangum
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
from io import BytesIO
from fpdf import FPDF
from datetime import datetime, timedelta
from shapely.geometry import Point, Polygon
from bson import ObjectId
import smtplib
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import boto3
from botocore.exceptions import ClientError
from urllib.parse import quote
from sort import *
import traceback


app = Flask(__name__, static_folder='../build', static_url_path='/')
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

email_scheduler = None  # Global reference for one active scheduler
class EmailScheduler:
    def __init__(self, stream_id, user_email):
        self.stream_id = stream_id
        self.user_email = user_email
        self.running = False
        self.thread = None

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._send_emails, daemon=True)
            self.thread.start()
            print(f"[EMAIL SCHEDULER] Started for stream {self.stream_id}")

    def stop(self):
        self.running = False
        print(f"[EMAIL SCHEDULER] Stopped for stream {self.stream_id}")

    def _send_emails(self):
        while self.running:
            try:
                self.send_email_with_pdf()
                for _ in range(60):  # Check every second for 1 minute
                    if not self.running:
                        break
                    time.sleep(1)
            except Exception as e:
                print(f"[EMAIL SCHEDULER ERROR] {e}")
                time.sleep(10)

    def send_email_with_pdf(self):
        try:
            # Fetch stream data from MongoDB
            stream = stream_collection.find_one({"stream_id": self.stream_id})
            if not stream:
                print(f"[EMAIL ERROR] No stream found for ID {self.stream_id}")
                return

            # Generate the PDF report
            pdf_buffer = BytesIO()
            generate_stream_report(stream, pdf_buffer)
            pdf_buffer.seek(0)

            # Prepare email
            msg = MIMEMultipart()
            msg['From'] = EMAIL_ADDRESS
            msg['To'] = self.user_email
            msg['Subject'] = "Perimeter Breach Detection Report"

            body = f"""
            Alert: Your stream {self.stream_id} is currently active.

            Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}

            Attached is the latest perimeter breach detection report.

            Best regards,
            Perimeter Breach Detection System
            """
            msg.attach(MIMEText(body, 'plain'))

            # Attach PDF
            pdf_attachment = MIMEBase('application', 'octet-stream')
            pdf_attachment.set_payload(pdf_buffer.read())
            encoders.encode_base64(pdf_attachment)
            pdf_attachment.add_header('Content-Disposition', f'attachment; filename=report_{self.stream_id}.pdf')
            msg.attach(pdf_attachment)

            # Send email
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                server.send_message(msg)

            print(f"[EMAIL SENT] Report sent for stream {self.stream_id}")

        except Exception as email_error:
            print(f"[EMAIL ERROR] Failed to send email for stream {self.stream_id}: {email_error}")


def monitor_stream():
    global email_scheduler

    while True:
        try:
            active_stream = stream_collection.find_one({}, {"stream_id": 1, "user_id": 1})  # Get first active stream

            if active_stream:
                stream_id = active_stream["stream_id"]

                if not email_scheduler or email_scheduler.stream_id != stream_id:
                    user = users_collection.find_one({"_id": ObjectId(active_stream["user_id"])})

                    if not user:
                        continue

                    user_email = user["email"]

                    # Stop previous scheduler if a new stream starts
                    if email_scheduler:
                        email_scheduler.stop()

                    # Start a new scheduler
                    email_scheduler = EmailScheduler(stream_id, user_email)
                    email_scheduler.start()

            else:
                # No active streams, stop any existing scheduler
                if email_scheduler:
                    email_scheduler.stop()
                    email_scheduler = None

            time.sleep(5)  # Check for active streams every 5 seconds

        except Exception as e:
            print(f"[MONITOR ERROR] {e}")
            time.sleep(5)


s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)
BUCKET_NAME = os.getenv('AWS_BUCKET_NAME')
def upload_to_s3(image_bytes, stream_id, filename):
    """
    Upload an image to S3 and return its URL
    """
    try:
        # Create the key with stream_id folder structure
        s3_key = f"{stream_id}/{filename}"
        
        # Upload the file
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=s3_key,
            Body=image_bytes,
            ContentType='image/jpeg'
        )
        
        # Generate the URL
        url = f"https://{BUCKET_NAME}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{quote(s3_key)}"
        return url
    except ClientError as e:
        print(f"Error uploading to S3: {e}")
        return None
    

    
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
        # self.cell(0, 10, 'Stream Activity Report', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')


def generate_stream_report(stream_data, output_buffer, company_logo_path="https://i.ibb.co/7xvjfwt9/logo.png", company_name="LOGICLENS"):
    if not isinstance(output_buffer, BytesIO):
        output_buffer = BytesIO()

    pdf = StreamReport()
    pdf.alias_nb_pages()
    pdf.add_page()

     # Add logo and company name in row at the top center
    if company_logo_path:
        # Define logo dimensions
        logo_width = 20
        logo_height = 20
        
        # Calculate total width of logo + spacing + text
        pdf.set_font('Arial', 'B', 20)
        text_width = pdf.get_string_width(company_name)
        total_width = logo_width + text_width 
        
        # Calculate starting x position to center the whole group
        page_width = pdf.w - 2 * pdf.l_margin
        start_x = pdf.l_margin + (page_width - total_width) / 2
        
        # Draw logo
        pdf.image(company_logo_path, x=start_x, y=10, w=logo_width, h=logo_height)
        
        # Draw company name next to logo
        pdf.set_xy(start_x + logo_width + 2, 10 + (logo_height/2) - 4)  # Adjust Y to vertically center with logo
        pdf.cell(text_width, 8, company_name, 0, 1)
    
    pdf.ln(20)  # Space after header

    # Main title with modern styling
    pdf.set_fill_color(245, 245, 245)  # Light gray background
    pdf.set_font('Arial', 'B', 20)
    pdf.cell(0, 20, 'Perimeter Breach Detection Report', 0, 1, 'C', fill=True)
    pdf.ln(10)

    # Stream Information Section with modern styling
    pdf.set_font('Arial', 'B', 14)
    pdf.set_fill_color(230, 230, 230)
    pdf.cell(0, 10, 'Stream Information', 0, 1, fill=True)
    pdf.set_font('Arial', '', 12)
    pdf.ln(5)
    pdf.cell(0, 8, f"Username: {stream_data.get('username', 'N/A')}", 0, 1)
    pdf.cell(0, 8, f"Stream URL: {stream_data.get('rtsp_url', 'N/A')}", 0, 1)

    created_at = stream_data.get('created_at')
    if created_at:
        if isinstance(created_at, datetime):
            created_at_str = created_at.strftime('%Y-%m-%d %H:%M:%S')
        else:
            created_at_str = str(created_at)
        pdf.cell(0, 8, f"Created: {created_at_str}", 0, 1)
    pdf.ln(10)

    # Breach Summary Section
    breaches = stream_data.get('breaches', [])
    pdf.set_font('Arial', 'B', 14)
    pdf.set_fill_color(230, 230, 230)
    pdf.cell(0, 10, 'Breach Summary', 0, 1, fill=True)
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 8, f"Total Breaches Detected: {len(breaches)}", 0, 1)
    pdf.ln(10)

    # Detailed Breach List with modern layout
    if breaches:
        pdf.set_font('Arial', 'B', 14)
        pdf.set_fill_color(230, 230, 230)
        pdf.cell(0, 10, 'Detailed Breach List', 0, 1, fill=True)
        pdf.ln(5)

        for i, breach in enumerate(breaches, 1):
            if pdf.get_y() > 250:
                pdf.add_page()

            # Breach header with number
            pdf.set_font('Arial', 'B', 12)
            pdf.set_fill_color(240, 240, 240)
            pdf.cell(0, 10, f"Breach #{i}", 0, 1, fill=True)
            pdf.ln(5)

            timestamp = breach.get('timestamp')
            if timestamp:
                if isinstance(timestamp, datetime):
                    timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    timestamp_str = str(timestamp)

            # Center timestamp
            pdf.set_font('Arial', '', 12)
            pdf.cell(0, 8, f"Time: {timestamp_str}", 0, 1, 'C')

            # Center image
            if 'image' in breach:
                try:
                    import requests
                    response = requests.get(breach['image'])
                    if response.status_code == 200:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image:
                            temp_image.write(response.content)
                            temp_image_path = temp_image.name

                        # Calculate center position for image
                        page_width = pdf.w - 2 * pdf.l_margin
                        image_width = 100  # Fixed width for consistency
                        image_height = 60   # Fixed height for consistency
                        x_position = pdf.l_margin + (page_width - image_width) / 2
                        
                        pdf.image(temp_image_path, x=x_position, y=pdf.get_y(), w=image_width, h=image_height)
                        pdf.ln(image_height + 10)  # Space after image
                        os.remove(temp_image_path)

                except Exception as e:
                    pdf.cell(0, 30, "Error loading image", 0, 1, "C")
                    pdf.ln(10)

            pdf.ln(10)  # Space between breaches

    pdf_data = pdf.output(dest='S').encode('latin1')
    output_buffer.write(pdf_data)

    return output_buffer


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
        self.MIN_BREACH_INTERVAL = 2
        self.frame_width = None
        self.frame_height = None
        self.container_width = None
        self.container_height = None
        
        # Initialize SORT tracker with parameters
        self.tracker = Sort(max_age=30,  # Maximum frames to keep a track alive
                          min_hits=3,    # Minimum hits to start tracking
                          iou_threshold=0.3)  # IOU threshold for matching
        
        # Set to store tracked person IDs that have already breached
        self.detected_persons = set()

        stream_doc = {
            "stream_id": stream_id,
            "user_id": user_id,
            "username": username,
            "rtsp_url": rtsp_url,
            "created_at": datetime.utcnow(),
            "breaches": []
        }
        
        try:
            existing_stream = stream_collection.find_one({"stream_id": stream_id})
            if existing_stream:
                print(f"Stream already exists with ID: {stream_id}")
            else:
                result = stream_collection.insert_one(stream_doc)
                print(f"New stream created with ID: {stream_id}, MongoDB _id: {result.inserted_id}")
        except Exception as e:
            print(f"Error initializing stream document: {e}")

    def set_container_dimensions(self, width, height):
        self.container_width = width
        self.container_height = height
        print(f"Container dimensions set: {width}x{height}")

    def scale_perimeter_points(self, points):
        if not self.frame_width or not self.frame_height or not self.container_width or not self.container_height:
            return points
        
        scaled_points = []
        for x, y in points:
            scaled_x = (x / self.container_width) * self.frame_width
            scaled_y = (y / self.container_height) * self.frame_height
            scaled_points.append([scaled_x, scaled_y])
        
        # print(f"Scaled points from {points} to {scaled_points}")
        return scaled_points

    def detect_humans(self, frame):
        try:
            height, width = frame.shape[:2]
            new_width = 640
            new_height = int(height * (new_width / width))
            resized_frame = cv2.resize(frame, (new_width, new_height))

            print("Running YOLO detection...")
            results = model(resized_frame)
            
            detections = []
            scale_x = width / new_width
            scale_y = height / new_height

            # Convert YOLO detections to SORT format [x1,y1,x2,y2,confidence]
            for result in results[0].boxes:
                if result.cls == 0 and result.conf > 0.5:  # Only humans with confidence > 0.5
                    x1, y1, x2, y2 = map(float, result.xyxy[0])
                    # Scale coordinates back to original frame size
                    x1, x2 = x1 * scale_x, x2 * scale_x
                    y1, y2 = y1 * scale_y, y2 * scale_y
                    confidence = float(result.conf)
                    detections.append([x1, y1, x2, y2, confidence])

            return np.array(detections) if detections else np.empty((0, 5))

        except Exception as e:
            print(f"Error in detect_humans: {e}")
            return np.empty((0, 5))

    def save_breach_event(self, frame, box, track_id, confidence):
        current_time = datetime.utcnow()
        
        try:
            print(f"Attempting to save breach for stream {self.stream_id}")
            
            # Encode frame with reduced quality
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            
            # Generate unique filename for the image
            filename = f"breach_{current_time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg"
            
            # Upload to S3
            image_url = upload_to_s3(buffer.tobytes(), self.stream_id, filename)
            
            if not image_url:
                raise Exception("Failed to upload image to S3")

            # Create breach event with track ID
            breach_event = {
                "timestamp": current_time,
                "bounding_box": [float(x) for x in box],
                "track_id": int(track_id),
                "confidence": float(confidence),
                "image": image_url
            }

            # Update stream document with new breach
            result = stream_collection.update_one(
                {"stream_id": self.stream_id},
                {
                    "$push": {
                        "breaches": breach_event
                    }
                }
            )

            if result.modified_count > 0:
                image_base64 = base64.b64encode(buffer).decode('utf-8')
                socketio.emit(f'breach-{self.stream_id}', {
                    "timestamp": current_time.isoformat(),
                    "track_id": int(track_id),
                    "confidence": float(confidence),
                    "image": image_base64
                })
                print(f"Breach event saved and emitted for stream {self.stream_id} - Track ID: {track_id}")
            else:
                print(f"Failed to update stream document for stream ID: {self.stream_id}")

        except Exception as e:
            print(f"Error saving breach event: {e}")
            print(traceback.format_exc())

    def process_stream(self):
        try:
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            
            if not cap.isOpened():
                raise Exception(f"Failed to open video stream: {self.rtsp_url}")
            
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
                
                if self.frame_counter % (self.SKIP_FRAMES + 1) != 0:
                    continue

                try:
                    if self.perimeter and len(self.perimeter) > 2:
                        # Get detections in SORT format
                        detections = self.detect_humans(frame)
                        
                        if len(detections) > 0:
                            # Update SORT tracker
                            tracked_objects = self.tracker.update(detections)
                            
                            # Draw perimeter
                            perimeter_points = np.array(self.perimeter, np.int32)
                            cv2.polylines(frame, [perimeter_points], True, (0, 255, 0), 2)
                            perimeter_polygon = Polygon(self.perimeter)

                            # Process each tracked object
                            for track in tracked_objects:
                                x1, y1, x2, y2, track_id = track
                                
                                # Calculate center point
                                center_x = (x1 + x2) / 2
                                center_y = (y1 + y2) / 2
                                center_point = Point(center_x, center_y)
                                
                                # Check if person is inside perimeter
                                if perimeter_polygon.contains(center_point):
                                    # Only process if this person hasn't been detected before
                                    if track_id not in self.detected_persons:
                                        self.detected_persons.add(track_id)
                                        
                                        # Find confidence from original detections
                                        confidence = 0.0
                                        if len(detections) > 0:
                                            # Find matching detection based on IOU
                                            track_bbox = np.array([x1, y1, x2, y2])
                                            ious = [self.calculate_iou(track_bbox, det[:4]) for det in detections]
                                            if len(ious) > 0:
                                                max_iou_idx = np.argmax(ious)
                                                confidence = detections[max_iou_idx][4]
                                        
                                        # Draw bounding box and ID
                                        color = (0, 0, 255)  # Red for new detection
                                        cv2.rectangle(frame, 
                                                    (int(x1), int(y1)), 
                                                    (int(x2), int(y2)), 
                                                    color, 2)
                                        
                                        # Add ID and confidence label
                                        label = f"ID: {int(track_id)} Conf: {confidence:.2f}"
                                        cv2.putText(frame, 
                                                label,
                                                (int(x1), int(y1) - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.5,
                                                color,
                                                2)
                                        
                                        # Save breach event
                                        box = (int(x1), int(y1), int(x2-x1), int(y2-y1))
                                        self.save_breach_event(frame, box, track_id, confidence)
                                        print(f"New person detected inside perimeter - ID: {track_id}")
                                    else:
                                        # Draw previously detected person in different color
                                        cv2.rectangle(frame, 
                                                    (int(x1), int(y1)), 
                                                    (int(x2), int(y2)), 
                                                    (255, 165, 0), 2)  # Orange for tracked
                                        cv2.putText(frame, 
                                                f"ID: {int(track_id)} (Tracked)",
                                                (int(x1), int(y1) - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.5,
                                                (255, 165, 0),
                                                2)

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

    @staticmethod
    def calculate_iou(box1, box2):
        """Calculate Intersection over Union (IOU) between two bounding boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection

        return intersection / union if union > 0 else 0



def save_breach_event(self, frame, box, confidence):
    current_time = datetime.utcnow()
    
    if (self.last_breach_time and 
        (current_time - self.last_breach_time).total_seconds() < self.MIN_BREACH_INTERVAL):
        return

    try:
        print(f"Attempting to save breach for stream {self.stream_id}")
        
        # Encode frame with reduced quality
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        
        # Generate unique filename for the image
        filename = f"breach_{current_time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg"
        
        # Upload to S3
        image_url = upload_to_s3(buffer.tobytes(), self.stream_id, filename)
        
        if not image_url:
            raise Exception("Failed to upload image to S3")

        # Create breach event with S3 URL instead of base64 image
        breach_event = {
            "timestamp": current_time,
            "bounding_box": [float(x) for x in box],
            "confidence": float(confidence),
            "image": image_url
        }

        # Update stream document with new breach
        result = stream_collection.update_one(
            {"stream_id": self.stream_id},
            {
                "$push": {
                    "breaches": breach_event
                }
            }
        )

        if result.modified_count > 0:
            self.last_breach_time = current_time
            
            # Convert image to base64 just for real-time Socket.IO emission
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            socketio.emit(f'breach-{self.stream_id}', {
                "timestamp": current_time.isoformat(),
                "confidence": float(confidence),
                "image": image_base64  # Keep base64 for real-time display
            })
            print(f"Breach event saved and emitted for stream {self.stream_id}")
        else:
            print(f"Failed to update stream document for stream ID: {self.stream_id}")

    except Exception as e:
        print(f"Error saving breach event: {e}")
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
    global email_scheduler  # Declare global variable

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

            print(f"[STREAM STARTED] ID: {stream_id}")

            # **Start Email Scheduler for this Stream**
            if email_scheduler:
                email_scheduler.stop()  # Stop any existing scheduler before starting a new one

            email_scheduler = EmailScheduler(stream_id, user["email"])  # Assign to global variable
            email_scheduler.start()

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
                 
                # **Stop the Email Scheduler if the stopped stream matches**
                if email_scheduler and email_scheduler.stream_id == stream_id:
                    email_scheduler.stop()
                    email_scheduler = None  # Reset email scheduler reference
                    print(f"[EMAIL SCHEDULER] Stopped for stream {stream_id}")
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
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({"error": "No token provided"}), 401

        payload = verify_token(auth_header)
        if not payload:
            return jsonify({"error": "Invalid or expired token"}), 401

        stream = stream_collection.find_one({"stream_id": stream_id})
        if not stream:
            return jsonify({"error": "Stream not found"}), 404

        if stream.get("user_id") != payload.get("id"):
            return jsonify({"error": "Unauthorized to access this stream"}), 403

        pdf_buffer = BytesIO()
        generate_stream_report(stream, pdf_buffer)
        
        if not pdf_buffer:
            return jsonify({"error": "Failed to generate PDF"}), 500
            
        pdf_buffer.seek(0)
        pdf_content = pdf_buffer.read()
        pdf_base64 = base64.b64encode(pdf_content).decode()
        
        filename = f"stream_report_{stream_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        return jsonify({
            "success": True,
            "pdf": pdf_base64,
            "filename": filename
        }), 200

    except Exception as e:
        print(f"Error generating report: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500


handler = Mangum(app)

if __name__ == "__main__":
    try:
        socketio.run(app, host='0.0.0.0', port=8000, debug=False)
    except Exception as e:
        print(f"Error running application: {e}")
    finally:
        cleanup_on_exit(None, None)