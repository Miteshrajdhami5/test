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
from queue import Queue

# Pin Definitions
IN1 = OutputDevice(14)  # Connect to motor IN1
IN2 = OutputDevice(15)  # Connect to motor IN2
IN3 = OutputDevice(18)  # Connect to motor IN3
IN4 = OutputDevice(23)  # Connect to motor IN4
IR_SENSOR = InputDevice(17)  # IR sensor on GPIO 17

# Step sequence for a 4-phase stepper motor
step_sequence = [
    [1, 0, 0, 0],  # Activate IN1
    [1, 1, 0, 0],  # Activate IN1 + IN2
    [0, 1, 0, 0],  # Activate IN2
    [0, 1, 1, 0],  # Activate IN2 + IN3
    [0, 0, 1, 0],  # Activate IN3
    [0, 0, 1, 1],  # Activate IN3 + IN4
    [0, 0, 0, 1],  # Activate IN4
    [1, 0, 0, 1]   # Activate IN4 + IN1
]

# Initialize Flask web server
app = Flask(__name__)

# Global variables
motor_running = False
vehicle_stopped = True
server_logs = ["System Ready"]
last_log_time = datetime.now()
last_gps_location = "https://www.google.com/maps?q=27.670052333333334,85.438842"
pending_authorization = False
sms_sent_for_current_attempt = False
camera_lock = threading.Lock()
start_queue = Queue()  # Queue to get result from start_vehicle_with_face

# Directory for reference face images (Desktop as database)
REFERENCE_IMAGE_DIR = "/home/mrd/Desktop"
if not os.path.exists(REFERENCE_IMAGE_DIR):
    print(f"Error: Desktop directory {REFERENCE_IMAGE_DIR} not found. Exiting.")
    exit()

# Load all reference face images dynamically (owner_face.png, owner_face1.png, ..., owner_face1000.png)
reference_encodings = []
reference_image_paths = glob.glob(os.path.join(REFERENCE_IMAGE_DIR, "owner_face*.png"))
if not reference_image_paths:
    print(f"Error: No reference images found in {REFERENCE_IMAGE_DIR}. Please add images like owner_face.png, owner_face1.png, etc.")
    exit()

print(f"Found {len(reference_image_paths)} reference images: {reference_image_paths}")
for image_path in reference_image_paths:
    try:
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            reference_encodings.append((image_path, encodings[0]))
            print(f"Loaded reference face from {image_path}")
        else:
            print(f"Warning: No face detected in {image_path}")
    except Exception as e:
        print(f"Error loading {image_path}: {e}")

if not reference_encodings:
    print("Error: No valid face encodings found in any reference images. Exiting.")
    exit()

# Initialize the camera
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Error: Could not open camera initially")
    exit()

# Function to reinitialize camera
def reinitialize_camera():
    global camera
    with camera_lock:
        camera.release()
        for i in range(3):
            camera = cv2.VideoCapture(i)
            if camera.isOpened():
                print(f"Attempting camera reinitialization on index {i}")
                # Test capture 3 times to ensure stability
                for test in range(3):
                    ret, frame = camera.read()
                    if ret and frame is not None and frame.size > 0:
                        print(f"Camera test capture successful on index {i}, attempt {test + 1}")
                        return True
                    else:
                        print(f"Camera test capture failed on index {i}, attempt {test + 1}")
                        sleep(1)
            sleep(1)
        print("Error: Failed to reinitialize camera on any index after multiple attempts")
        return False

# Function to capture an image from the camera and check for a face
def capture_image_with_face():
    global last_log_time, camera
    max_retries = 5  # Increased retries
    retry_count = 0
    with camera_lock:
        while retry_count < max_retries:
            if not camera.isOpened():
                if not reinitialize_camera():
                    print(f"Error: Camera not opened after reinitialization (Retry {retry_count + 1}/{max_retries})")
                    current_time = datetime.now()
                    if (current_time - last_log_time).total_seconds() > 1:
                        new_log = f"{current_time.strftime('%H:%M:%S')} - Camera reinitialization failed"
                        server_logs.append(new_log)
                        last_log_time = current_time
                    retry_count += 1
                    sleep(2)  # Increased sleep for stability
                    continue
            ret, frame = camera.read()
            if not ret or frame is None or frame.size == 0:
                print(f"Error: Failed to capture image from camera (Retry {retry_count + 1}/{max_retries}) - Frame invalid or empty")
                current_time = datetime.now()
                if (current_time - last_log_time).total_seconds() > 1:
                    new_log = f"{current_time.strftime('%H:%M:%S')} - Failed to capture image, frame invalid"
                    server_logs.append(new_log)
                    last_log_time = current_time
                retry_count += 1
                sleep(2)
                continue
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray_frame)
            if mean_brightness < 30:
                print("Image too dark. Waiting for better lighting...")
                current_time = datetime.now()
                if (current_time - last_log_time).total_seconds() > 1:
                    new_log = f"{current_time.strftime('%H:%M:%S')} - Image too dark, waiting for better lighting"
                    server_logs.append(new_log)
                    last_log_time = current_time
                sleep(1)
                continue
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            if face_locations:
                captured_image_path = os.path.join(REFERENCE_IMAGE_DIR, f"captured_face_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                cv2.imwrite(captured_image_path, frame)
                print(f"Image captured successfully: {captured_image_path}")
                current_time = datetime.now()
                if (current_time - last_log_time).total_seconds() > 1:
                    new_log = f"{current_time.strftime('%H:%M:%S')} - Image captured successfully at {captured_image_path}"
                    server_logs.append(new_log)
                    last_log_time = current_time
                return captured_image_path
            else:
                print("No face detected in the frame. Waiting for a clear face...")
                current_time = datetime.now()
                if (current_time - last_log_time).total_seconds() > 1:
                    new_log = f"{current_time.strftime('%H:%M:%S')} - No face detected, waiting for a clear face"
                    server_logs.append(new_log)
                    last_log_time = current_time
                sleep(1)
        print("Max retries reached. Camera capture failed.")
        current_time = datetime.now()
        if (current_time - last_log_time).total_seconds() > 1:
            new_log = f"{current_time.strftime('%H:%M:%S')} - Max retries reached, camera capture failed"
            server_logs.append(new_log)
            last_log_time = current_time
        return None

# Function to compare faces against all reference images, stopping at the first match
def compare_faces(captured_image_path):
    global last_log_time
    try:
        if not os.path.exists(captured_image_path):
            print(f"Error: Image file {captured_image_path} not found")
            current_time = datetime.now()
            if (current_time - last_log_time).total_seconds() > 1:
                new_log = f"{current_time.strftime('%H:%M:%S')} - Image file not found: {captured_image_path}"
                server_logs.append(new_log)
                last_log_time = current_time
            return False
        captured_image = face_recognition.load_image_file(captured_image_path)
        captured_encodings = face_recognition.face_encodings(captured_image)
        if not captured_encodings:
            print("Error: No face found in the captured image")
            current_time = datetime.now()
            if (current_time - last_log_time).total_seconds() > 1:
                new_log = f"{current_time.strftime('%H:%M:%S')} - No face found in captured image"
                server_logs.append(new_log)
                last_log_time = current_time
            return False
        captured_encoding = captured_encodings[0]
        for image_path, ref_encoding in reference_encodings:
            print(f"Comparing captured face with reference image: {os.path.basename(image_path)}")
            results = face_recognition.compare_faces([ref_encoding], captured_encoding, tolerance=0.6)
            if results[0]:
                print(f"Face match found with reference image: {os.path.basename(image_path)}")
                current_time = datetime.now()
                if (current_time - last_log_time).total_seconds() > 1:
                    new_log = f"{current_time.strftime('%H:%M:%S')} - Face match found with {os.path.basename(image_path)}"
                    server_logs.append(new_log)
                    last_log_time = current_time
                return True
        print("No face match found in any reference images.")
        current_time = datetime.now()
        if (current_time - last_log_time).total_seconds() > 1:
            new_log = f"{current_time.strftime('%H:%M:%S')} - No face match found"
            server_logs.append(new_log)
            last_log_time = current_time
        return False
    except IOError as e:
        print(f"Error in face comparison: {e}")
        current_time = datetime.now()
        if (current_time - last_log_time).total_seconds() > 1:
            new_log = f"{current_time.strftime('%H:%M:%S')} - Error in face comparison: {e}"
            server_logs.append(new_log)
            last_log_time = current_time
        return False
    except Exception as e:
        print(f"Unexpected error in face comparison: {e}")
        current_time = datetime.now()
        if (current_time - last_log_time).total_seconds() > 1:
            new_log = f"{current_time.strftime('%H:%M:%S')} - Unexpected error in face comparison: {e}"
            server_logs.append(new_log)
            last_log_time = current_time
        return False

# Function to control the stepper motor
def set_step(w1, w2, w3, w4):
    IN1.value = w1
    IN2.value = w2
    IN3.value = w3
    IN4.value = w4

# Function to continuously rotate the motor
def step_motor_continuous(delay=0.1):
    global motor_running, server_logs, last_log_time
    motor_running = True
    current_time = datetime.now()
    if (current_time - last_log_time).total_seconds() > 1:
        new_log = f"{current_time.strftime('%H:%M:%S')} - Vehicle has been started"
        server_logs.append(new_log)
        last_log_time = current_time
    print("Vehicle is now running continuously...")
    while motor_running:
        for step in step_sequence:
            if not motor_running:
                break
            set_step(*step)
            sleep(delay / 0.0001)
    stop_motor()

# Function to stop the motor
def stop_motor():
    global motor_running, vehicle_stopped, server_logs, last_log_time, camera, pending_authorization, sms_sent_for_current_attempt
    motor_running = False
    vehicle_stopped = True
    set_step(0, 0, 0, 0)
    current_time = datetime.now()
    if (current_time - last_log_time).total_seconds() > 1:
        new_log = f"{current_time.strftime('%H:%M:%S')} - Vehicle has been stopped"
        server_logs.append(new_log)
        last_log_time = current_time
    print("Vehicle stopped!")
    with camera_lock:
        if not reinitialize_camera():
            print("Warning: Camera reinitialization failed after stop")
    # Reset authorization states
    pending_authorization = False
    sms_sent_for_current_attempt = False

# Function to start the vehicle with IR sensor and face recognition
def start_vehicle_with_face():
    global motor_running, vehicle_stopped, server_logs, last_log_time, pending_authorization, sms_sent_for_current_attempt, start_queue
    result = {"success": False, "message": ""}
    if vehicle_stopped:
        # Reset SMS sent flag for a fresh attempt
        sms_sent_for_current_attempt = False
        print("Waiting for IR sensor (finger) detection...")
        current_time = datetime.now()
        if (current_time - last_log_time).total_seconds() > 1:
            new_log = f"{current_time.strftime('%H:%M:%S')} - Waiting for IR sensor detection"
            server_logs.append(new_log)
            last_log_time = current_time

        while IR_SENSOR.value:
            sleep(0.1)
        print("IR sensor detected finger! Proceeding to face recognition...")
        current_time = datetime.now()
        if (current_time - last_log_time).total_seconds() > 1:
            new_log = f"{current_time.strftime('%H:%M:%S')} - IR sensor detected finger, starting face recognition"
            server_logs.append(new_log)
            last_log_time = current_time

        # Ensure camera is initialized
        with camera_lock:
            if not camera.isOpened():
                if not reinitialize_camera():
                    new_log = f"{current_time.strftime('%H:%M:%S')} - Camera initialization failed"
                    server_logs.append(new_log)
                    last_log_time = current_time
                    result["message"] = "Camera initialization failed"
                    start_queue.put(result)
                    return

        captured_image_path = None
        timeout = 30  # 30 seconds timeout
        start_time = datetime.now()
        while not captured_image_path and vehicle_stopped:
            if (datetime.now() - start_time).total_seconds() > timeout:
                print("Timeout reached for face capture")
                new_log = f"{datetime.now().strftime('%H:%M:%S')} - Face capture timeout"
                server_logs.append(new_log)
                last_log_time = datetime.now()
                result["message"] = "Face capture timeout"
                start_queue.put(result)
                return
            captured_image_path = capture_image_with_face()
            if not captured_image_path:
                current_time = datetime.now()
                new_log = f"{current_time.strftime('%H:%M:%S')} - Failed to capture image"
                server_logs.append(new_log)
                last_log_time = current_time
                sleep(1)
            else:
                break

        if not captured_image_path:
            result["message"] = "No image captured"
            start_queue.put(result)
            return

        match_found = compare_faces(captured_image_path)
        if match_found:
            print("Face match found! Starting vehicle...")
            current_time = datetime.now()
            if (current_time - last_log_time).total_seconds() > 1:
                new_log = f"{current_time.strftime('%H:%M:%S')} - Face match found! Starting vehicle..."
                server_logs.append(new_log)
                last_log_time = current_time
            threading.Thread(target=step_motor_continuous, args=(0.001,)).start()
            vehicle_stopped = False
            send_sms()
            sms_sent_for_current_attempt = False
            result["success"] = True
            result["message"] = "Vehicle started successfully"
        else:
            print("No face match found. Requesting authorization...")
            current_time = datetime.now()
            if (current_time - last_log_time).total_seconds() > 1:
                new_log = f"{current_time.strftime('%H:%M:%S')} - No face match found. Requesting authorization..."
                server_logs.append(new_log)
                last_log_time = current_time
            if not sms_sent_for_current_attempt:
                send_image_to_owner(captured_image_path)
                sms_sent_for_current_attempt = True
            pending_authorization = True
            result["message"] = "No face match, authorization required"
        start_queue.put(result)

# Function to send SMS with links
def send_sms():
    website_url = 'http://192.168.100.85:5000'
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
            print("SMS sent successfully!")
            current_time = datetime.now()
            if (current_time - last_log_time).total_seconds() > 1:
                new_log = f"{current_time.strftime('%H:%M:%S')} - SMS sent successfully"
                server_logs.append(new_log)
                last_log_time = current_time
        else:
            print(f"Failed to send SMS: {r.text}")
            current_time = datetime.now()
            if (current_time - last_log_time).total_seconds() > 1:
                new_log = f"{current_time.strftime('%H:%M:%S')} - Failed to send SMS: {r.text}"
                server_logs.append(new_log)
                last_log_time = current_time
    except Exception as e:
        print(f"Error sending SMS: {e}")
        current_time = datetime.now()
        if (current_time - last_log_time).total_seconds() > 1:
            new_log = f"{current_time.strftime('%H:%M:%S')} - Error sending SMS: {e}"
            server_logs.append(new_log)
            last_log_time = current_time

# Function to send SMS with image link
def send_image_to_owner(captured_image_path):
    global server_logs, last_log_time
    current_time = datetime.now()
    image_url = f"http://192.168.100.85:5000/captured_image"
    message = f"""
    🚨 Unauthorized Access Attempt!!!

    An unknown person tried to start the vehicle.
    Image: {image_url}
    Check authorization on: http://192.168.100.85:5000
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
            print("SMS with image link sent successfully!")
            new_log = f"{current_time.strftime('%H:%M:%S')} - Unauthorized access detected, SMS sent to owner"
            server_logs.append(new_log)
            last_log_time = current_time
        else:
            print(f"Failed to send SMS: {r.text}")
            new_log = f"{current_time.strftime('%H:%M:%S')} - Failed to send SMS to owner"
            server_logs.append(new_log)
            last_log_time = current_time
    except Exception as e:
        print(f"Error sending SMS: {e}")
        new_log = f"{current_time.strftime('%H:%M:%S')} - Error sending SMS: {e}"
        server_logs.append(new_log)
        last_log_time = current_time

# Modified GPS functions
def get_location():
    global server_logs, last_log_time, last_gps_location
    fixed_location = "https://www.google.com/maps?q=27.670052333333334,85.438842"
    last_gps_location = fixed_location
    current_time = datetime.now()
    if (current_time - last_log_time).total_seconds() > 1:
        new_log = f"{current_time.strftime('%H:%M:%S')} - Using fixed location: {fixed_location}"
        server_logs.append(new_log)
        last_log_time = current_time
    return fixed_location

# SSE Route for real-time updates
def stream_logs():
    global server_logs, initial_logs
    initial_logs = server_logs.copy()
    yield f"data: {json.dumps({'logs': server_logs})}\n\n"
    while True:
        sleep(0.1)
        if len(server_logs) > len(initial_logs):
            yield f"data: {json.dumps({'logs': server_logs})}\n\n"
            initial_logs = server_logs.copy()

# Flask Routes
@app.route('/stop_vehicle', methods=['GET'])
def stop_motor_web():
    global server_logs, last_log_time
    print("Stopping Vehicle via web...")
    stop_motor()
    return jsonify({"logs": server_logs})

@app.route('/start_vehicle', methods=['GET'])
def start_motor_web():
    global server_logs, last_log_time, start_queue
    print("Starting vehicle via web...")
    start_queue.queue.clear()  # Clear any previous results
    threading.Thread(target=start_vehicle_with_face).start()
    # Wait for result with a 35-second timeout
    try:
        result = start_queue.get(timeout=35)
        if result["success"]:
            current_time = datetime.now()
            if (current_time - last_log_time).total_seconds() > 1:
                new_log = f"{current_time.strftime('%H:%M:%S')} - {result['message']}"
                server_logs.append(new_log)
                last_log_time = current_time
        else:
            current_time = datetime.now()
            if (current_time - last_log_time).total_seconds() > 1:
                new_log = f"{current_time.strftime('%H:%M:%S')} - Start failed: {result['message']}"
                server_logs.append(new_log)
                last_log_time = current_time
    except Exception as e:
        print(f"Timeout or error waiting for start result: {e}")
        current_time = datetime.now()
        if (current_time - last_log_time).total_seconds() > 1:
            new_log = f"{current_time.strftime('%H:%M:%S')} - Timeout or error waiting for start: {e}"
            server_logs.append(new_log)
            last_log_time = current_time
    return jsonify({"logs": server_logs})

@app.route('/send_location', methods=['GET'])
def send_location_web():
    global server_logs, last_log_time
    print("Fetching GPS location...")
    location = get_location()
    return jsonify({"logs": server_logs, "location": location})

@app.route('/redirect_to_map', methods=['GET'])
def redirect_to_map():
    fixed_location = "https://www.google.com/maps?q=27.670052333333334,85.438842"
    return redirect(fixed_location)

@app.route('/authorize', methods=['GET', 'POST'])
def authorize():
    global server_logs, last_log_time, pending_authorization, vehicle_stopped, sms_sent_for_current_attempt
    current_time = datetime.now()
    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'yes':
            print("Owner authorized via web! Starting vehicle...")
            if (current_time - last_log_time).total_seconds() > 1:
                new_log = f"{current_time.strftime('%H:%M:%S')} - Owner authorized via web! Starting vehicle..."
                server_logs.append(new_log)
                last_log_time = current_time
            threading.Thread(target=step_motor_continuous, args=(0.001,)).start()
            vehicle_stopped = False
            send_sms()
            pending_authorization = False
            sms_sent_for_current_attempt = False
        elif action == 'no':
            print("Owner denied access via web.")
            if (current_time - last_log_time).total_seconds() > 1:
                new_log = f"{current_time.strftime('%H:%M:%S')} - Owner denied access via web."
                server_logs.append(new_log)
                last_log_time = current_time
            pending_authorization = False
            sms_sent_for_current_attempt = False
        return redirect('/')
    elif pending_authorization:
        latest_image = max(glob.glob(os.path.join(REFERENCE_IMAGE_DIR, "captured_face_*.png")), key=os.path.getctime, default=None)
        if not latest_image or not os.path.exists(latest_image):
            return redirect('/')
        return render_template_string("""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Authorize Access</title>
                <style>
                    body { background: #1a2535; color: #fff; font-family: Arial, sans-serif; text-align: center; padding: 20px; }
                    img { max-width: 100%; height: auto; margin: 20px 0; }
                    .button-group { display: flex; justify-content: center; gap: 10px; }
                    button { padding: 10px 20px; font-size: 16px; cursor: pointer; background: #4a69bd; color: #fff; border: none; border-radius: 5px; }
                    button:hover { background: #ff4d6d; }
                </style>
            </head>
            <body>
                <h1>Unauthorized Access Attempt</h1>
                <p>Please verify the captured image and authorize access.</p>
                <img src="/captured_image" alt="Captured Face">
                <form method="POST">
                    <div class="button-group">
                        <button type="submit" name="action" value="yes">Yes</button>
                        <button type="submit" name="action" value="no">No</button>
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
    return "No captured image available", 404

@app.route('/')
def index():
    global pending_authorization, last_log_time
    start_url = 'http://192.168.100.85:5000/start_vehicle'
    stop_url = 'http://192.168.100.85:5000/stop_vehicle'
    location_url = 'http://192.168.100.85:5000/send_location'
    map_redirect_url = 'http://192.168.100.85:5000/redirect_to_map'
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
    app.run(host='0.0.0.0', port=5000, debug=False)

flask_thread = threading.Thread(target=run_flask)
flask_thread.daemon = True
flask_thread.start()

try:
    print("Waiting for IR sensor and face recognition to start the vehicle...")
    current_time = datetime.now()
    if (current_time - last_log_time).total_seconds() > 1:
        new_log = f"{current_time.strftime('%H:%M:%S')} - Waiting for IR sensor and face recognition..."
        server_logs.append(new_log)
        last_log_time = current_time

    while True:
        if not pending_authorization:
            try:
                start_vehicle_with_face()
            except Exception as e:
                print(f"Error in start_vehicle_with_face: {e}")
                current_time = datetime.now()
                if (current_time - last_log_time).total_seconds() > 1:
                    new_log = f"{current_time.strftime('%H:%M:%S')} - Error: {e}"
                    server_logs.append(new_log)
                    last_log_time = current_time
        sleep(1)
except Exception as e:
    print(f"Main loop crashed: {e}")
    current_time = datetime.now()
    if (current_time - last_log_time).total_seconds() > 1:
        new_log = f"{current_time.strftime('%H:%M:%S')} - Main loop crashed: {e}"
        server_logs.append(new_log)
        last_log_time = current_time
finally:
    print("Cleaning up...")
    stop_motor()
    with camera_lock:
        camera.release()