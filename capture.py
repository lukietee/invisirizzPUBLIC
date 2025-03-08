import cv2
import openai
import base64
import time
import mediapipe as mp

# OpenAI API key
openai.api_key = "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"

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Virtual camera 1
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# Set resolution
WIDTH = 1280
HEIGHT = 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

if not cap.isOpened():
    print("Error: Could not open OBS Virtual Camera.")
    exit()

def encode_image(image_path):
    """Convert image to Base64 format for OpenAI API."""
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode("utf-8")

def analyze_emotion(image_path):
    """Send the image to OpenAI Vision API and return detected emotions + Valentine pickup lines."""
    image_base64 = encode_image(image_path)

    response = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": """You are my AI dating assistant that will be integrated into my smart glasses. 
            I am currently single and looking for a date. Your job will be to help me find a girlfriend. As I walk around, 
            the camera from my smart glasses will pick up whatever I am seeing around me and will detect people. 
            If I am interested in a person, these smart glasses will take a frame (picture) of the girl that I am interested in. 
            You will be presented with this image. You will respond in short and concise phrases that will be read aloud to me 
            through my smart glasses to help me get this girl to be interested in me.  

            Your job at first will be to identify their emotion. Are they stressed because they are studying? Are they disgusted? 
            Are they locked on their phone? You will identify how they are feeling based on their BODY LANGUAGE. You will then state 
            at the beginning of the phrase **'Subject looks [INSERT EMOTION], try saying:'** followed by a smooth pickup line.  

            The pickup line that follows the emotion identifier will be centered around **asking them to be my valentine**.  
            - Be as smooth as possible.  
            - Use wordplay, pop culture references, or even slightly NSFW humor if needed.  
            - Make sure the pickup line ends with **"Will you be my valentine?"**  
            - Keep it short and natural—if it’s too long, it will be awkward to say out loud.  

            **Examples:**  
            - "Subject appears to be studying, try saying: Are you studying chemistry? Because I think we’ve got some undeniable reaction going on here. Will you be my valentine?"  
            - "Subject appears to be on their phone, try saying: Are you scrolling for Valentine's plans? Because I think I just popped up as the best option. Be my valentine?"  
            - "Subject appears to be on their phone, try saying: Is that Google Calendar? ‘Cause I think you just found the perfect date—me. Will you be my valentine?"  

            If you **cannot identify their emotion**, **do not send error messages**. Since these will be read in my ear, I **don’t want to hear error messages**. Instead, respond with something like:  
            - "Move closer so I can get a better read!"  
            
            ONLY respond with **one** formatted phrase and nothing else."""},

            {"role": "user", "content": [
                {"type": "text", "text": "Here is an image for analysis:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]}
        ],
        max_tokens=200
    )

    return response.choices[0].message.content 


last_capture_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb_frame)

    # Draw pose landmarks if detected
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )

    cv2.imshow("OBS Virtual Camera Feed", frame)

    # Capture a frame every 15 seconds for analysis
    if time.time() - last_capture_time >= 15:
        cv2.imwrite("frame.jpg", frame)
        print(" Frame captured. Sending to OpenAI for analysis...")

        # Analyze emotions
        try:
            detected_emotion = analyze_emotion("frame.jpg")
            print(detected_emotion)
        except openai.OpenAIError as e:
            print(" OpenAI API Error:", e)

        last_capture_time = time.time()  # Reset timer

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

 
