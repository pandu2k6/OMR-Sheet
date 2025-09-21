import cv2
import numpy as np
import os
import pandas as pd
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

# --- Flask App Configuration ---
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
DEBUG_FOLDER = 'static/debug' # Folder for debug images
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure all necessary folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
os.makedirs(DEBUG_FOLDER, exist_ok=True)

########################################################################
# --- OMR PROCESSING LOGIC (CONTOUR-GRID METHOD) ---
########################################################################

SUBJECT_MAP = {
    "Python": list(range(0, 20)), "EDA": list(range(20, 40)),
    "SQL": list(range(40, 60)), "POWER BI": list(range(60, 80)),
    "Statistics": list(range(80, 100))
}
TOTAL_QUESTIONS = 100
TOTAL_CHOICES = 4
ANSWER_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
# This threshold is for checking the average pixel darkness inside a bubble contour
# Lower value for darker marks (pen), higher value for lighter marks (pencil)
BUBBLE_DARKNESS_THRESHOLD = 115 

def load_key_from_csv(filepath):
    try:
        df = pd.read_csv(filepath, header=0)
        df.columns = [str(col).strip().lower() for col in df.columns]
        required_cols = ['question', 'answers']
        if not all(col in df.columns for col in required_cols):
            return None, f"Error: CSV file must contain 'Question' and 'Answers' columns. Found: {list(df.columns)}"

        answer_key = {}
        for index, row in df.iterrows():
            q_num_str = str(row['question']).strip()
            ans_str = str(row['answers']).strip()
            if not q_num_str.isdigit():
                return None, f"Invalid question number '{q_num_str}' at row {index + 2}."
            q_num = int(q_num_str) - 1
            correct_answers_set = set()
            ans_parts = [a.strip().upper() for a in ans_str.split(',')]
            for ans in ans_parts:
                if ans in ANSWER_MAP:
                    correct_answers_set.add(ANSWER_MAP[ans])
                else:
                    return None, f"Invalid answer character '{ans}' for question {q_num + 1}."
            if correct_answers_set:
                answer_key[q_num] = correct_answers_set

        if not answer_key: return None, "No valid answers found in the CSV file."
        return answer_key, None
    except Exception as e:
        return None, f"Error reading CSV file: {e}"

def four_point_transform(image, pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0], [maxWidth-1, 0], [maxWidth-1, maxHeight-1], [0, maxHeight-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method in ("right-to-left", "bottom-to-top"): reverse = True
    if method in ("top-to-bottom", "bottom-to-top"): i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    if not boundingBoxes: return ([], [])
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
    return (cnts, boundingBoxes)

def process_omr_sheet(image_path, answer_key, original_filename):
    try:
        image = cv2.imread(image_path)
        if image is None: return {"error": "Could not read image."}

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Threshold the image to get a binary black-and-white image
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find the main paper contour
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        doc_contour = None
        if len(contours) > 0:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for c in contours:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4:
                    doc_contour = approx
                    break
        if doc_contour is None:
            return {"error": "Could not find OMR sheet outline. Image may be too dark, blurry, or cropped."}

        # Apply perspective transform to the original grayscale image to get a flat view
        warped = four_point_transform(gray, doc_contour.reshape(4, 2))
        
        # Re-threshold the clean, warped image to get clear bubbles
        _, warped_thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find all contours on the warped, thresholded image
        contours, _ = cv2.findContours(warped_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        question_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            # Filter for shapes that have a reasonable area to be a bubble
            if 100 < area < 1500:
                peri = cv2.arcLength(c, True)
                if peri > 0:
                    # Filter for shapes that are reasonably circular
                    circularity = 4 * np.pi * (area / (peri * peri))
                    if circularity > 0.7:
                        question_contours.append(c)

        warning = ""
        if len(question_contours) < TOTAL_QUESTIONS * TOTAL_CHOICES:
            warning = f"Warning: Found only {len(question_contours)} of {TOTAL_QUESTIONS * TOTAL_CHOICES} bubbles. Scoring may be incomplete."
        
        # Sort all found bubbles from top to bottom
        question_contours = sort_contours(question_contours, method="top-to-bottom")[0]
        
        user_answers = {}
        questions_per_col = 20
        # Group bubbles into rows of 20 (5 columns * 4 choices)
        for (q_row, i) in enumerate(np.arange(0, len(question_contours), TOTAL_CHOICES * 5)):
            # Get all the bubbles for the current row
            row_contours = question_contours[i : i + (TOTAL_CHOICES * 5)]
            # Sort the bubbles in the row from left to right
            row_contours = sort_contours(row_contours, method="left-to-right")[0]
            
            # Now process each column within the row
            for (col, j) in enumerate(np.arange(0, len(row_contours), TOTAL_CHOICES)):
                question_num = col * questions_per_col + q_row
                bubbles = row_contours[j : j + TOTAL_CHOICES]
                user_marked_set = set()
                
                for (choice_idx, c) in enumerate(bubbles):
                    mask = np.zeros(warped.shape, dtype="uint8")
                    cv2.drawContours(mask, [c], -1, 255, -1)
                    
                    # Check the average darkness within the bubble on the GRAYSCALE image
                    mask = cv2.bitwise_and(warped, warped, mask=mask)
                    masked_pixels = mask[mask > 0]
                    
                    if masked_pixels.size > 0:
                        average_intensity = np.mean(masked_pixels)
                        if average_intensity < BUBBLE_DARKNESS_THRESHOLD:
                             user_marked_set.add(choice_idx)
                
                user_answers[question_num] = user_marked_set
        
        scores = {subject: 0 for subject in SUBJECT_MAP}
        total_correct = 0
        for q_num, correct_set in answer_key.items():
            user_set = user_answers.get(q_num, set())
            if user_set == correct_set:
                total_correct += 1
                for subject, q_list in SUBJECT_MAP.items():
                    if q_num in q_list:
                        scores[subject] += 1
                        break
        
        percentage = (total_correct / len(answer_key)) * 100 if len(answer_key) > 0 else 0
        scores['Total'] = f"{total_correct}/{len(answer_key)}"
        scores['Percentage'] = f"{percentage:.2f}%"

        return {"scores": scores, "warning": warning}

    except Exception as e:
        return {"error": f"Unexpected error: {e}"}

########################################################################
# --- FLASK ROUTES ---
########################################################################

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        image_files = request.files.getlist('image_files')
        answer_file = request.files.get('answer_file')
        answer_key, error_message = None, None

        if not (answer_file and answer_file.filename):
            error_message = "Please upload a CSV answer key file."
        else:
            answer_filename = secure_filename(answer_file.filename)
            answer_filepath = os.path.join(app.config['UPLOAD_FOLDER'], answer_filename)
            answer_file.save(answer_filepath)
            answer_key, error_message = load_key_from_csv(answer_filepath)
        
        if error_message:
            return render_template('index.html', error=error_message)
        if not image_files or image_files[0].filename == '':
            return render_template('index.html', error="Please upload OMR images.")
        
        results = []
        for image_file in image_files:
            if image_file and image_file.filename:
                image_filename = secure_filename(image_file.filename)
                image_filepath = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
                image_file.save(image_filepath)
                result_data = process_omr_sheet(image_filepath, answer_key, image_filename)
                result_data['filename'] = image_filename
                results.append(result_data)
                
        return render_template('results.html', results=results)
        
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

