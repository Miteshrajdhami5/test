#!/usr/bin/env python3

import requests
import serial
from gpiozero import OutputDevice, InputDevice
from time import sleep
from flask import Flask, render_template_string, Response, jsonify, redirect, request, send_file
import threading
from datetime import datetime
import json
import cv2
import face_recognition
import numpy as np
import os
import certifi
import glob

# Pin Definitions
IN1 = OutputDevice(14)  # Connect to motor IN1
IN2 = OutputDevice(15)  # Connect to motor IN2
IN3 = OutputDevice(18)  # Connect to motor IN3
IN4 = OutputDevice(23)  # Connect to motor IN4
IR_SENSOR = InputDevice(17)  # IR sensor on GPIO 17

step_sequence = [
    [1, 0, 0, 0],  # Step 1
    [1, 0, 0, 0],  # Step 2
    [1, 0, 0, 0],  # Step 3
    [1, 0, 0, 0],  # Step 4
    [1, 0, 0, 0],  # Step 5
    [1, 0, 0, 0],  # Step 6
    [1, 0, 0, 0],  # Step 7
    [1, 0, 0, 0],  # Step 8
]

# Initialize Flask web server
app = Flask(__name__)

# Global variables
motor_running = False
vehicle_stopped = True
server_logs = ["System Ready"]
last_log_time = datetime.now()
last_gps_location = "https://www.google.com/maps?q=27.671167,85.439778"
pending_authorization = False
sms_sent_for_current_attempt = False
camera_lock = threading.Lock()
face_recognition_active = False

# Directory for reference face images
REFERENCE_IMAGE_DIR = "/home/mrd/Desktop"
if not os.path.exists(REFERENCE_IMAGE_DIR):
    log_message = f"Error: Desktop directory {REFERENCE_IMAGE_DIR} not found. Exiting."
    print(log_message)
    server_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_message}")
    exit()

# Load all reference face images
reference_encodings = []
reference_image_paths = glob.glob(os.path.join(REFERENCE_IMAGE_DIR, "owner_face*.png"))
if not reference_image_paths:
    log_message = f"Error: No reference images found in {REFERENCE_IMAGE_DIR}"
    print(log_message)
    server_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_message}")
    exit()

log_message = f"Found {len(reference_image_paths)} reference images"
print(log_message)
server_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_message}")
for image_path in reference_image_paths:
    try:
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            reference_encodings.append((image_path, encodings[0]))
            log_message = f"Loaded reference face from {image_path}"
            print(log_message)
            server_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_message}")
    except Exception as e:
        log_message = f"Error loading {image_path}: {e}"
        print(log_message)
        server_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_message}")

if not reference_encodings:
    log_message = "Error: No valid face encodings found"
    print(log_message)
    server_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_message}")
    exit()

# Initialize the camera
camera = None

def initialize_camera():
    global camera
    with camera_lock:
        if camera is not None and camera.isOpened():
            camera.release()
            log_message = "Previous camera released"
            print(log_message)
            server_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_message}")
        
        # Try GStreamer pipeline for Raspberry Pi Camera Module
        camera = cv2.VideoCapture('libcamerasrc ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! appsink', cv2.CAP_GSTREAMER)
        if camera.isOpened():
            log_message = "Camera initialized with GStreamer pipeline"
            print(log_message)
            server_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_message}")
            sleep(1)
            return True
        
        # Fallback to V4L2 for USB cameras
        for i in range(10):  # Try indices 0 to 9
            camera = cv2.VideoCapture(i, cv2.CAP_V4L2)
            if camera.isOpened():
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                log_message = f"Camera initialized on index {i} with V4L2"
                print(log_message)
                server_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_message}")
                sleep(1)
                return True
            else:
                log_message = f"Failed to open camera at index {i} with V4L2"
                print(log_message)
                server_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_message}")
        
        log_message = "Error: Failed to initialize camera after trying all backends"
        print(log_message)
        server_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_message}")
        return False

# Initial camera setup with retries
max_camera_retries = 3
camera_initialized = False
for attempt in range(max_camera_retries):
    if initialize_camera():
        camera_initialized = True
        break
    log_message = f"Camera initialization attempt {attempt + 1} failed, retrying..."
    print(log_message)
    server_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_message}")
    sleep(2)

if not camera_initialized:
    log_message = "Failed to initialize camera after multiple attempts, exiting..."
    print(log_message)
    server_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_message}")
    exit()

def capture_image_with_face():
    global last_log_time, camera
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    captured_image_path = os.path.join(REFERENCE_IMAGE_DIR, f"captured_face_{timestamp}.png")
    temp_images = sorted(glob.glob(os.path.join(REFERENCE_IMAGE_DIR, "captured_face_*.png")))
    for old_img in temp_images[:-1]:
        try:
            os.remove(old_img)
            log_message = f"Removed old temporary image: {old_img}"
            print(log_message)
            server_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_message}")
        except Exception as e:
            log_message = f"Error removing old image: {e}"
            print(log_message)
            server_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_message}")

    max_retries = 10
    retry_count = 0
    min_face_size = 100
    min_brightness = 50

    with camera_lock:
        if not camera.isOpened():
            if not initialize_camera():
                log_message = "Failed to reinitialize camera"
                print(log_message)
                server_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_message}")
                return None

        while retry_count < max_retries:
            for _ in range(5):
                camera.grab()
            ret, frame = camera.retrieve()
            if not ret or frame is None:
                retry_count += 1
                log_message = "Failed to capture frame from camera"
                print(log_message)
                server_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_message}")
                sleep(0.5)
                continue

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray_frame)
            if brightness < min_brightness:
                log_message = f"Image too dark (brightness: {brightness:.1f}) - please ensure good lighting"
                print(log_message)
                server_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_message}")
                sleep(0.5)
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model="hog", number_of_times_to_upsample=2)
            if not face_locations:
                log_message = "No face detected - please look directly at camera"
                print(log_message)
                server_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_message}")
                sleep(0.5)
                continue

            (top, right, bottom, left) = face_locations[0]
            face_width = right - left
            face_height = bottom - top
            face_center_x = (left + right) // 2
            face_center_y = (top + bottom) // 2

            if face_width < min_face_size or face_height < min_face_size:
                log_message = f"Face too small (w:{face_width}, h:{face_height}) - please move closer"
                print(log_message)
                server_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_message}")
                sleep(0.5)
                continue

            frame_center_x = frame.shape[1] // 2
            frame_center_y = frame.shape[0] // 2
            x_offset = abs(face_center_x - frame_center_x) / frame_center_x
            y_offset = abs(face_center_y - frame_center_y) / frame_center_y
            if x_offset > 0.2 or y_offset > 0.2:
                log_message = "Face not centered - please look straight at camera"
                print(log_message)
                server_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_message}")
                sleep(0.5)
                continue

            if len(face_locations) > 1:
                log_message = "Multiple faces detected - only one person should be in frame"
                print(log_message)
                server_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_message}")
                sleep(0.5)
                continue

            face_landmarks = face_recognition.face_landmarks(rgb_frame, face_locations)
            if face_landmarks:
                chin = face_landmarks[0]['chin']
                left_chin = chin[0]
                right_chin = chin[16]
                left_eye = face_landmarks[0]['left_eye']
                right_eye = face_landmarks[0]['right_eye']
                eye_angle = np.arctan2(right_eye[0][1] - left_eye[0][1], right_eye[0][0] - left_eye[0][0]) * 180 / np.pi
                if abs(eye_angle) > 15:
                    log_message = f"Face rotated (angle: {eye_angle:.1f}°) - please face straight forward"
                    print(log_message)
                    server_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_message}")
                    sleep(0.5)
                    continue

            cv2.imwrite(captured_image_path, frame)
            log_message = f"Captured good quality image: {captured_image_path}"
            print(log_message)
            server_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_message}")
            log_message = f"Face size: {face_width}x{face_height}, Position: ({face_center_x},{face_center_y})"
            print(log_message)
            server_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_message}")

            if os.path.exists(captured_image_path):
                return captured_image_path
            else:
                log_message = "Failed to save image properly"
                print(log_message)
                server_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_message}")
                retry_count += 1

        log_message = f"Failed to capture valid image after {max_retries} attempts"
        print(log_message)
        server_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_message}")
        return None

def compare_faces(captured_image_path):
    global last_log_time
    try:
        if not os.path.exists(captured_image_path):
            log_message = f"Error: Image file {captured_image_path} not found"
            print(log_message)
            server_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_message}")
            return False

        captured_image = face_recognition.load_image_file(captured_image_path)
        captured_encodings = face_recognition.face_encodings(captured_image, num_jitters=10, model="large")
        if not captured_encodings:
            log_message = "Error: No face found in the captured image"
            print(log_message)
            server_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_message}")
            return False

        captured_encoding = captured_encodings[0]
        for image_path, ref_encoding in reference_encodings:
            log_message = f"Comparing with reference: {os.path.basename(image_path)}"
            print(log_message)
            server_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_message}")
            face_distance = face_recognition.face_distance([ref_encoding], captured_encoding)[0]
            confidence = (1 - face_distance) * 100
            if face_distance < 0.7 and confidence > 50:
                log_message = f"Strong match found (distance: {face_distance:.4f}, confidence: {confidence:.1f}%)"
                print(log_message)
                server_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_message}")
                current_time = datetime.now()
                last_log_time = current_time
                return True
            else:
                log_message = f"No match (distance: {face_distance:.4f}, confidence: {confidence:.1f}%)"
                print(log_message)
                server_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_message}")

        log_message = "No strong matches found in reference images"
        print(log_message)
        server_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_message}")
        return False
    except Exception as e:
        log_message = f"Error in face comparison: {e}"
        print(log_message)
        server_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_message}")
        return False

def set_step(w1, w2, w3, w4):
    IN1.value = w1
    IN2.value = w2
    IN3.value = w3
    IN4.value = w4

def step_motor_continuous(delay=0.1):
    global motor_running, server_logs, last_log_time
    motor_running = True
    current_time = datetime.now()
    log_message = f"Vehicle started"
    print(log_message)
    server_logs.append(f"{current_time.strftime('%H:%M:%S')} - {log_message}")
    last_log_time = current_time
    while motor_running:
        for step in step_sequence:
            if not motor_running:
                break
            set_step(*step)
            sleep(delay / 0.0001)
    stop_motor()

def stop_motor():
    global motor_running, vehicle_stopped, server_logs, last_log_time, face_recognition_active
    motor_running = False
    vehicle_stopped = True
    face_recognition_active = False
    set_step(0, 0, 0, 0)
    current_time = datetime.now()
    log_message = f"Vehicle stopped"
    print(log_message)
    server_logs.append(f"{current_time.strftime('%H:%M:%S')} - {log_message}")
    last_log_time = current_time
    with camera_lock:
        if not camera.isOpened():
            initialize_camera()

def start_vehicle_with_face():
    global motor_running, vehicle_stopped, server_logs, last_log_time, pending_authorization, sms_sent_for_current_attempt, face_recognition_active
    if not face_recognition_active:
        return
    with camera_lock:
        if not camera.isOpened():
            max_retries = 3
            for attempt in range(max_retries):
                if initialize_camera():
                    break
                log_message = f"Camera reinitialization attempt {attempt + 1} failed, retrying..."
                print(log_message)
                server_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_message}")
                sleep(2)
            if not camera.isOpened():
                log_message = "Failed to reinitialize camera for face verification after multiple attempts"
                print(log_message)
                server_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_message}")
                return
    if vehicle_stopped:
        log_message = "Starting fresh face verification process..."
        print(log_message)
        server_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_message}")
        current_time = datetime.now()
        last_log_time = current_time
        log_message = "Waiting for IR sensor (finger) detection..."
        print(log_message)
        server_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_message}")
        while IR_SENSOR.value:
            if not face_recognition_active:
                return
            sleep(0.1)
        captured_image_path = capture_image_with_face()
        if not captured_image_path:
            log_message = "Failed to capture valid image"
            print(log_message)
            server_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_message}")
            return
        match_found = compare_faces(captured_image_path)
        if match_found:
            log_message = "Strict face verification passed"
            print(log_message)
            server_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_message}")
            try:
                os.remove(captured_image_path)
                log_message = f"Deleted temporary image after match: {captured_image_path}"
                print(log_message)
                server_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_message}")
            except Exception as e:
                log_message = f"Error deleting image: {e}"
                print(log_message)
                server_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_message}")
            threading.Thread(target=step_motor_continuous, args=(0.001,)).start()
            vehicle_stopped = False
            send_sms()
            sms_sent_for_current_attempt = False
        else:
            log_message = "Face verification failed - keeping image for authorization"
            print(log_message)
            server_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_message}")
            if not sms_sent_for_current_attempt:
                send_image_to_owner(captured_image_path)
                sms_sent_for_current_attempt = True
            pending_authorization = True

def send_sms():
    website_url = 'http://192.168.10.86:5000'
    message = f"""
    🚗 Vehicle Start Detected!!!
    The vehicle has been started. Control it here:
    {website_url}
    Thank you!
    """
    try:
        r = requests.post(
            "https://sms.aakashsms.com/sms/v3/send/",
            data={
                'auth_token': 'a5a492a8f95c9a887b799021d5a447890d9d7bf1e5eeb954b38ebfa9cf76b482Mrd',
                'to': '9818858567',
                'text': message
            },
            verify=certifi.where()
        )
        if r.status_code == 200:
            log_message = "SMS sent successfully!"
            print(log_message)
            server_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_message}")
        else:
            log_message = f"Failed to send SMS: {r.text}"
            print(log_message)
            server_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_message}")
    except Exception as e:
        log_message = f"Error sending SMS: {e}"
        print(log_message)
        server_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_message}")

def send_image_to_owner(captured_image_path):
    global server_logs, last_log_time
    current_time = datetime.now()
    image_url = f"http://192.168.10.86:5000/captured_image"
    message = f"""
    🚗 Vehicle Start Detected!!!
    An unknown person tried to start the vehicle.
    Image: {image_url}
    Check authorization on: http://192.168.10.86:5000
    """
    try:
        r = requests.post(
            "https://sms.aakashsms.com/sms/v3/send/",
            data={
                'auth_token': 'a5a492a8f95c9a887b799021d5a447890d9d7bf1e5eeb954b38ebfa9cf76b482mrd',
                'to': '9818858567',
                'text': message
            },
            verify=certifi.where()
        )
        if r.status_code == 200:
            log_message = "SMS with image link sent successfully!"
            print(log_message)
            server_logs.append(f"{current_time.strftime('%H:%M:%S')} - {log_message}")
        else:
            log_message = f"Failed to send SMS: {r.text}"
            print(log_message)
            server_logs.append(f"{current_time.strftime('%H:%M:%S')} - {log_message}")
        last_log_time = current_time
    except Exception as e:
        log_message = f"Error sending SMS: {e}"
        print(log_message)
        server_logs.append(f"{current_time.strftime('%H:%M:%S')} - {log_message}")
        last_log_time = current_time

def get_location():
    global server_logs, last_log_time, last_gps_location
    fixed_location = "https://www.google.com/maps?q=27.671167,85.439778"
    last_gps_location = fixed_location
    current_time = datetime.now()
    if (current_time - last_log_time).total_seconds() > 1:
        log_message = f"Using fixed location: {fixed_location}"
        print(log_message)
        server_logs.append(f"{current_time.strftime('%H:%M:%S')} - {log_message}")
        last_log_time = current_time
    return fixed_location

@app.route('/stream')
def stream():
    return Response(stream_logs(), mimetype="text/event-stream")

def stream_logs():
    global server_logs, initial_logs
    initial_logs = server_logs.copy()
    yield f"data: {json.dumps({'logs': server_logs})}\n\n"
    while True:
        sleep(0.1)
        if len(server_logs) > len(initial_logs):
            yield f"data: {json.dumps({'logs': server_logs})}\n\n"
            initial_logs = server_logs.copy()

@app.route('/stop_vehicle', methods=['GET'])
def stop_motor_web():
    global server_logs, last_log_time
    stop_motor()
    return jsonify({"logs": server_logs})

@app.route('/start_vehicle', methods=['GET'])
def start_motor_web():
    global server_logs, last_log_time, face_recognition_active
    face_recognition_active = True
    log_message = "Start vehicle command received"
    print(log_message)
    server_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_message}")
    threading.Thread(target=start_vehicle_with_face).start()
    return jsonify({"logs": server_logs})

@app.route('/send_location', methods=['GET'])
def send_location_web():
    global server_logs, last_log_time
    log_message = "Fetching GPS location..."
    print(log_message)
    server_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_message}")
    location = get_location()
    return jsonify({"logs": server_logs, "location": location})

@app.route('/redirect_to_map', methods=['GET'])
def redirect_to_map():
    fixed_location = "https://www.google.com/maps?q=27.671167,85.439778"
    return redirect(fixed_location)

@app.route('/authorize', methods=['GET', 'POST'])
def authorize():
    global server_logs, last_log_time, pending_authorization, vehicle_stopped, sms_sent_for_current_attempt
    current_time = datetime.now()
    latest_image = max(glob.glob(os.path.join(REFERENCE_IMAGE_DIR, "captured_face_*.png")), key=os.path.getctime, default=None)
    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'yes':
            log_message = "Owner authorized via web! Starting vehicle..."
            print(log_message)
            server_logs.append(f"{current_time.strftime('%H:%M:%S')} - {log_message}")
            last_log_time = current_time
            if latest_image and os.path.exists(latest_image):
                try:
                    os.remove(latest_image)
                    log_message = f"Deleted authorized image: {latest_image}"
                    print(log_message)
                    server_logs.append(f"{current_time.strftime('%H:%M:%S')} - {log_message}")
                except Exception as e:
                    log_message = f"Error deleting image: {e}"
                    print(log_message)
                    server_logs.append(f"{current_time.strftime('%H:%M:%S')} - {log_message}")
            threading.Thread(target=step_motor_continuous, args=(0.001,)).start()
            vehicle_stopped = False
            send_sms()
            pending_authorization = False
            sms_sent_for_current_attempt = False
        elif action == 'no':
            log_message = "Owner denied access via web."
            print(log_message)
            server_logs.append(f"{current_time.strftime('%H:%M:%S')} - {log_message}")
            last_log_time = current_time
            if latest_image and os.path.exists(latest_image):
                try:
                    os.remove(latest_image)
                    log_message = f"Deleted unauthorized image: {latest_image}"
                    print(log_message)
                    server_logs.append(f"{current_time.strftime('%H:%M:%S')} - {log_message}")
                except Exception as e:
                    log_message = f"Error deleting image: {e}"
                    print(log_message)
                    server_logs.append(f"{current_time.strftime('%H:%M:%S')} - {log_message}")
            pending_authorization = False
            sms_sent_for_current_attempt = False
        return redirect('/')
    elif pending_authorization and latest_image and os.path.exists(latest_image):
        return render_template_string("""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Authorize Access</title>
                <style>
                    body { background: #1a2535; color: #fff; font-family: Arial, sans-serif; text-align: center; padding: 20px; }
                    img { max-width: 100%; height: auto; margin: 20px 0; border: 2px solid #ff4d6d; border-radius: 5px; }
                    .button-group { display: flex; justify-content: center; gap: 10px; margin-top: 20px; }
                    button { padding: 10px 20px; font-size: 16px; cursor: pointer; border: none; border-radius: 5px; }
                    #yes-btn { background: #4CAF50; color: white; }
                    #no-btn { background: #f44336; color: white; }
                    button:hover { opacity: 0.8; }
                </style>
            </head>
            <body>
                <h1>Unauthorized Access Attempt</h1>
                <p>Please verify the captured image and authorize access:</p>
                <img src="/captured_image" alt="Captured Face">
                <form method="POST">
                    <div class="button-group">
                        <button type="submit" name="action" value="yes" id="yes-btn">Authorize</button>
                        <button type="submit" name="action" value="no" id="no-btn">Deny</button>
                    </div>
                </form>
            </body>
            </html>
        """)
    return redirect('/')

@app.route('/captured_image')
def captured_image():
    latest_image = max(glob.glob(os.path.join(REFERENCE_IMAGE_DIR, "captured_face_*.png")), key=os.path.getctime, default=None)
    if latest_image and os.path.exists(latest_image):
        return send_file(latest_image, mimetype='image/png')
    log_message = "No captured image available"
    print(log_message)
    server_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_message}")
    return "No captured image available", 404

@app.route('/')
def index():
    global pending_authorization, last_log_time
    start_url = 'http://192.168.10.86:5000/start_vehicle'
    stop_url = 'http://192.168.10.86:5000/stop_vehicle'
    location_url = 'http://192.168.10.86:5000/send_location'
    map_redirect_url = 'http://192.168.10.86:5000/redirect_to_map'
    if pending_authorization:
        return redirect('/authorize')
    return render_template_string("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Vehicle Control Dashboard</title>
            <style>
                * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Arial', sans-serif; }
                body { background: #1a2535; min-height: 100vh; display: flex; justify-content: center; align-items: center; padding: 20px; color: #fff; }
                .container { background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); border-radius: 15px; padding: 30px; width: 100%; max-width: 400px; box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3); border: 1px solid rgba(255, 255, 255, 0.1); text-align: center; }
                h1 { font-size: 2.2em; color: #ff4d6d; margin-bottom: 20px; text-shadow: 0 0 10px rgba(255, 77, 109, 0.5); }
                .button-group { display: flex; flex-direction: column; gap: 15px; margin-bottom: 20px; }
                .control-btn { display: inline-block; padding: 15px; font-size: 1.1em; text-decoration: none; color: #fff; background: linear-gradient(90deg, #ff4d6d, #4a69bd); border: none; border-radius: 25px; width: 100%; cursor: pointer; transition: transform 0.3s, box-shadow: 0.3s; box-shadow: 0 5px 15px rgba(255, 77, 109, 0.4); }
                .control-btn:hover { transform: translateY(-5px); box-shadow: 0 8px 25px rgba(255, 77, 109, 0.6); }
                .control-btn:active { transform: translateY(0); box-shadow: 0 3px 10px rgba(255, 77, 109, 0.3); }
                .status-box { margin-top: 20px; padding: 15px; background: rgba(255, 255, 255, 0.05); border-radius: 10px; font-size: 0.9em; color: #a3bffa; max-height: 150px; overflow-y: auto; text-align: left; border: 1px solid rgba(255, 255, 255, 0.1); }
                .status-log { margin-bottom: 10px; padding: 5px; background: rgba(255, 255, 255, 0.02); border-radius: 5px; }
                .status-log.new { animation: fadeIn 1s; }
                @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
                footer { margin-top: 20px; font-size: 0.9em; color: rgba(255, 255, 255, 0.6); }
                .bottom-button { margin-top: 20px; }
                @media (max-width: 480px) { .container { padding: 20px; } h1 { font-size: 1.8em; } .control-btn { font-size: 1em; padding: 12px; } .status-box { font-size: 0.8em; max-height: 120px; } }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Vehicle Control</h1>
                <div class="button-group">
                    <a href="{{ start_url }}" class="control-btn" onclick="updateStatus(event, this)">Start Vehicle</a>
                    <a href="{{ stop_url }}" class="control-btn" onclick="updateStatus(event, this)">Stop Vehicle</a>
                    <a href="{{ location_url }}" class="control-btn" onclick="updateStatus(event, this)">View GPS Location</a>
                </div>
                <div class="status-box" id="status-box">
                    {% for log in server_logs %}
                        <div class="status-log">{{ log }}</div>
                    {% endfor %}
                </div>
                <div class="bottom-button">
                    <a href="{{ map_redirect_url }}" class="control-btn">Open Latest Map</a>
                </div>
                <footer>2025</footer>
            </div>
            <script>
                const eventSource = new EventSource('/stream');
                const statusBox = document.getElementById('status-box');
                let lastLogCount = {{ server_logs|length }};
                eventSource.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    const currentLogCount = data.logs.length;
                    if (currentLogCount > lastLogCount) {
                        statusBox.innerHTML = '';
                        data.logs.forEach(log => {
                            const logDiv = document.createElement('div');
                            logDiv.className = 'status-log' + (currentLogCount > lastLogCount ? ' new' : '');
                            logDiv.textContent = log;
                            statusBox.appendChild(logDiv);
                        });
                        lastLogCount = currentLogCount;
                        statusBox.scrollTop = statusBox.scrollHeight;
                    }
                };
                eventSource.onerror = function() {
                    console.error('EventSource failed');
                };
                function updateStatus(event, element) {
                    event.preventDefault();
                    const url = element.href;
                    fetch(url)
                        .then(response => response.json())
                        .then(data => {
                            if (data.location) {
                                window.open(data.location, '_blank');
                            }
                        })
                        .catch(error => {
                            console.error('Error:', error);
                            const logDiv = document.createElement('div');
                            logDiv.className = 'status-log new';
                            logDiv.textContent = 'Error occurred';
                            statusBox.appendChild(logDiv);
                            statusBox.scrollTop = statusBox.scrollHeight;
                        });
                }
            </script>
        </body>
        </html>
    """, start_url=start_url, stop_url=stop_url, location_url=location_url, map_redirect_url=map_redirect_url, server_logs=server_logs)

initial_logs = server_logs.copy()

def run_flask():
    log_message = "Starting Flask server..."
    print(log_message)
    server_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_message}")
    app.run(host='0.0.0.0', port=5000, debug=False)

flask_thread = threading.Thread(target=run_flask)
flask_thread.daemon = True
flask_thread.start()

if __name__ == '__main__':
    try:
        log_message = "System ready - waiting for start command..."
        print(log_message)
        server_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_message}")
        while True:
            sleep(1)
    except KeyboardInterrupt:
        log_message = "Shutting down..."
        print(log_message)
        server_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_message}")
    finally:
        stop_motor()
        with camera_lock:
            if camera is not None and camera.isOpened():
                camera.release()
                log_message = "Camera released"
                print(log_message)
                server_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {log_message}")