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
face_recognition_active = False  # New flag to control face recognition

# Directory for reference face images (Desktop as database)
REFERENCE_IMAGE_DIR = "/home/mrd/Desktop"
if not os.path.exists(REFERENCE_IMAGE_DIR):
    print(f"Error: Desktop directory {REFERENCE_IMAGE_DIR} not found. Exiting.")
    exit()

# Load all reference face images
reference_encodings = []
reference_image_paths = glob.glob(os.path.join(REFERENCE_IMAGE_DIR, "owner_face*.png"))
if not reference_image_paths:
    print(f"Error: No reference images found in {REFERENCE_IMAGE_DIR}")
    exit()

print(f"Found {len(reference_image_paths)} reference images")
for image_path in reference_image_paths:
    try:
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            reference_encodings.append((image_path, encodings[0]))
            print(f"Loaded reference face from {image_path}")
    except Exception as e:
        print(f"Error loading {image_path}: {e}")

if not reference_encodings:
    print("Error: No valid face encodings found")
    exit()

# Initialize the camera
camera = None

def initialize_camera():
    global camera
    with camera_lock:
        if camera is not None and camera.isOpened():
            camera.release()
        
        for i in range(3):
            camera = cv2.VideoCapture(i)
            if camera.isOpened():
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                print(f"Camera initialized on index {i}")
                sleep(1)  # Warm-up time
                return True
        
        print("Error: Failed to initialize camera")
        return False

# Initial camera setup
if not initialize_camera():
    print("Failed to initialize camera initially")
    exit()

# Modified image capture function to ensure fresh images
def capture_image_with_face():
    global last_log_time, camera
    
    # First, clean up any old captured images
    for old_img in glob.glob(os.path.join(REFERENCE_IMAGE_DIR, "captured_face_*.png")):
        try:
            os.remove(old_img)
            print(f"Removed old image: {old_img}")
        except:
            pass

    max_retries = 5  # Increased retries for better reliability
    retry_count = 0
    
    with camera_lock:
        if not camera.isOpened():
            if not initialize_camera():
                return None

        while retry_count < max_retries:
            # Force a few blank reads to clear camera buffer
            for _ in range(3):
                camera.read()
            
            # Now capture fresh image
            ret, frame = camera.read()
            if not ret or frame is None:
                retry_count += 1
                sleep(0.5)
                continue

            # Check image quality
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if np.mean(gray_frame) < 40:  # Slightly brighter threshold
                print("Image too dark - please ensure good lighting")
                sleep(0.5)
                continue

            # Convert to RGB and detect faces
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            
            if not face_locations:
                print("No face detected - please look directly at camera")
                sleep(0.5)
                continue
                
            # Save the new image with timestamp
            captured_image_path = os.path.join(REFERENCE_IMAGE_DIR, f"captured_face_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png")
            cv2.imwrite(captured_image_path, frame)
            print(f"Captured new image: {captured_image_path}")
            return captured_image_path

        print(f"Failed to capture valid image after {max_retries} attempts")
        return None

# Modified face comparison function with stricter matching
def compare_faces(captured_image_path):
    global last_log_time
    try:
        if not os.path.exists(captured_image_path):
            print(f"Error: Image file {captured_image_path} not found")
            return False

        # Load the captured image
        captured_image = face_recognition.load_image_file(captured_image_path)
        captured_encodings = face_recognition.face_encodings(captured_image)
        
        if not captured_encodings:
            print("Error: No face found in the captured image")
            return False

        captured_encoding = captured_encodings[0]
        
        # Compare with each reference image using stricter parameters
        for image_path, ref_encoding in reference_encodings:
            print(f"Comparing with reference: {os.path.basename(image_path)}")
            
            # Use lower tolerance (0.5 instead of 0.6) for stricter matching
            # Also use face_distance for more precise comparison
            face_distance = face_recognition.face_distance([ref_encoding], captured_encoding)[0]
            
            # Only consider it a match if distance is very small (more strict)
            if face_distance < 0.4:  # Lower than default 0.6
                print(f"Strong match found with {os.path.basename(image_path)} (distance: {face_distance:.2f})")
                current_time = datetime.now()
                if (current_time - last_log_time).total_seconds() > 1:
                    new_log = f"{current_time.strftime('%H:%M:%S')} - Verified match with {os.path.basename(image_path)} (confidence: {(1-face_distance)*100:.1f}%)"
                    server_logs.append(new_log)
                    last_log_time = current_time
                return True
            else:
                print(f"No match with {os.path.basename(image_path)} (distance: {face_distance:.2f})")

        print("No strong matches found in reference images")
        return False
        
    except Exception as e:
        print(f"Error in face comparison: {e}")
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
    new_log = f"{current_time.strftime('%H:%M:%S')} - Vehicle started"
    server_logs.append(new_log)
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
    face_recognition_active = False  # Disable face recognition when stopped
    set_step(0, 0, 0, 0)
    
    current_time = datetime.now()
    new_log = f"{current_time.strftime('%H:%M:%S')} - Vehicle stopped"
    server_logs.append(new_log)
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
            if not initialize_camera():
                return

    if vehicle_stopped:
        print("Starting fresh face verification process...")
        current_time = datetime.now()
        new_log = f"{current_time.strftime('%H:%M:%S')} - Starting face verification"
        server_logs.append(new_log)
        last_log_time = current_time

        # Wait for IR sensor
        print("Waiting for IR sensor (finger) detection...")
        while IR_SENSOR.value:
            if not face_recognition_active:
                return
            sleep(0.1)

        # Capture and compare fresh image
        captured_image_path = None
        attempts = 0
        max_attempts = 3
        
        while attempts < max_attempts and not captured_image_path and face_recognition_active:
            captured_image_path = capture_image_with_face()
            attempts += 1
            if not captured_image_path:
                sleep(1)

        if not captured_image_path or not face_recognition_active:
            return

        # Perform strict comparison
        match_found = compare_faces(captured_image_path)
        
        if match_found:
            print("Strict face verification passed")
            threading.Thread(target=step_motor_continuous, args=(0.001,)).start()
            vehicle_stopped = False
            send_sms()
            sms_sent_for_current_attempt = False
        else:
            print("Face verification failed")
            if not sms_sent_for_current_attempt:
                send_image_to_owner(captured_image_path)
                sms_sent_for_current_attempt = True
            pending_authorization = True

# [Rest of your existing functions (send_sms, send_image_to_owner, etc.) remain the same]
def send_sms():

    website_url = 'http://192.168.10.86:5000'

    message = f"""

    ðŸš— Vehicle Start Detected!!!



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

        else:

            print(f"Failed to send SMS: {r.text}")

    except Exception as e:

        print(f"Error sending SMS: {e}")



# Function to send SMS with image link

def send_image_to_owner(captured_image_path):

    global server_logs, last_log_time

    current_time = datetime.now()

    image_url = f"http://192.168.10.86:5000/captured_image"

    message = f"""

    ðŸš¨ Unauthorized Access Attempt!!!



    An unknown person tried to start the vehicle.

    Image: {image_url}

    Check authorization on: http://192.168.10.86:5000

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

@app.route('/stop_vehicle', methods=['GET'])
def stop_motor_web():
    global server_logs, last_log_time
    stop_motor()
    return jsonify({"logs": server_logs})

@app.route('/start_vehicle', methods=['GET'])
def start_motor_web():
    global server_logs, last_log_time, face_recognition_active
    face_recognition_active = True  # Enable face recognition when start is clicked
    threading.Thread(target=start_vehicle_with_face).start()
    return jsonify({"logs": server_logs})

# [Rest of your existing Flask routes remain the same]
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



from flask import send_file



initial_logs = server_logs.copy()

def run_flask():

    app.run(host='0.0.0.0', port=5000, debug=False)



flask_thread = threading.Thread(target=run_flask)

flask_thread.daemon = True

flask_thread.start()


if __name__ == '__main__':
    try:
        print("System ready - waiting for start command...")
        while True:
            sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        stop_motor()
        with camera_lock:
            if camera.isOpened():
                camera.release()