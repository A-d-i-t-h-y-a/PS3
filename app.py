from flask import Flask, render_template, request, jsonify, send_from_directory
from roboflow import Roboflow
import base64
from PIL import Image, ImageDraw
from io import BytesIO
import easyocr
from difflib import SequenceMatcher
import cv2
import os
from tensorflow.keras.models import load_model
import re
import time
import shutil

app = Flask(__name__)

l = ["aadharno", "details", "qr", "emblem", "goi", "image"]
t = {}
txtbbs = {"aadharno":[0, 0, 0, 0], "details":[0, 0, 0, 0], "qr":[0, 0, 0, 0], "emblem":[0, 0, 0, 0]}

def emblem_class():
    model = load_model('your_model.h5')
    image = cv2.imread("emblem.jpg")
    image = cv2.resize(image, (100, 100))
    y_pred = model.predict(image.reshape(1, 100, 100, 3))
    print("real" if y_pred>0.5 else "fake")

# def overlay_boxes(image, predictions):
#     draw = ImageDraw.Draw(image)
#     for prediction in predictions:
#         width, height = image.size
#         x_center, y_center, w, h = (
#             prediction["x"],
#             prediction["y"],
#             prediction["width"],
#             prediction["height"],
#         )
#         x, y = x_center - w / 2, y_center - h / 2  # Calculate top-left coordinates
#         class_name = prediction["class"]

#         # Set background color based on class
#         class_colors = {
#             "details": "blue",
#             "qr": "green",
#             "image": "black",
#             "aadharno": "red",
#             "goi": "purple",
#             "emblem": "orange",
#             "name": "blue",
#             "dob" : "pink",
#             "gender" : "yellow"
#         }
#         if(class_name in l):
#             txtbbs[class_name] = [x, y, x + w, y + h]
#             t[class_name] = ""
#         # Draw thick filled rectangle as background
#         draw.rectangle([x, y, x + w, y + h], outline=class_colors.get(class_name, "white"), width=2)

#         # Draw class name on top-left corner in white
#         draw.rectangle([x, y, x+50, y+20], fill=class_colors.get(class_name, "white"))
#         draw.text((x, y), class_name, fill="white")
#     print(txtbbs)
#     return image

def extraction_of_text(image):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(image,paragraph=True)
    top_left = tuple(result[0][0][0])
    bottom_right = tuple(result[0][0][2])
    text = result[0][1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    return text


def aadhar_number_search(text):
    aadhar_pattern = re.compile(r'\b\d{4}\s\d{4}\s\d{4}\b')
    match = aadhar_pattern.search(text)
    if match:
        return match.group()
    else:
        return None

# def extract_text_from_image(image, bounding_boxes):
    
#     for box in bounding_boxes:
#         x, y, width, height = box
#         text_region = image.crop((x, y, x + width, y + height))
        
#         # Perform OCR on the text region
#         text = pytesseract.image_to_string(text_region)
#         extracted_text.append(text.strip())

#     return extracted_text

def read_qr(imgpth):
    img = cv2.imread(imgpth)
    qcd = cv2.QRCodeDetector()
    retval, decoded_info, points, straight_qrcode = qcd.detectAndDecodeMulti(img)
    print("qr data",decoded_info)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    try:
        global t
        t.clear()
        # Retrieve image data
        image_data_uri = request.json.get('image')

        # Extract base64-encoded part
        _, image_data_base64 = image_data_uri.split(',', 1)

        # Decode base64 image string
        image_bytes = base64.b64decode(image_data_base64)

        # Use BytesIO to create a stream from the image data
        image_stream = BytesIO(image_bytes)

        # Open the image using PIL
        image = Image.open(image_stream).convert('RGB')

        # Save the image to a file
        image.save("input_image.jpg")

        # rf = Roboflow(api_key="RMzZna7r8BabI0Fz7SJV")
        # project = rf.workspace().project("aadhardetect")
        # model = project.version(3).model

        rf = Roboflow(api_key="ZLHtscupsDPcPqZc4ouB")
        project = rf.workspace().project("aadhardetect")
        model = project.version(6).model
        
        # jsonimage = model.predict("input_image.jpg", confidence=40, overlap=30).json();
        prediction_result = model.predict("input_image.jpg", confidence=40, overlap=30)

        # Get predictions from the JSON response
        predictions = prediction_result.json()["predictions"]

        # Overlay bounding boxes on the input image
        image_with_boxes = overlay_boxes(image.copy(), predictions)

        # Save the image with bounding boxes (optional)
        image_with_boxes.save("output_image.jpg")
        # details_region = image.crop(txtbbs["details"])
        pth = "static/detected"
        if(os.path.exists(pth)):
            shutil.rmtree(pth)
            os.mkdir(pth)
        else:
            os.mkdir(pth)
        for i in t:
            region = image.crop(txtbbs[i])
            val = i + str(time.time())
            t[i] = val
            region.save(f"./static/detected/{val}.jpg")
        read_qr(f"./static/detected/{t[i]}.jpg")
        # qr_region = image.crop(txtbbs["qr"])
        # qr_region.save("./static/detected/qr.jpg")
        # read_qr("./static/detected/qr.jpg")
        # qr_region = image.crop(txtbbs["emblem"])
        # qr_region.save("./static/detected/emblem.jpg")
        # emblem_class()
        # aadharno_region = image.crop(txtbbs["aadharno"])
        # aadharno_region.save("./static/detected/ano.jpg")
        # print("aadhar number =", extraction_of_text("./static/detected/ano.jpg"))
        reader = easyocr.Reader(['hi', 'en'])
        # Perform OCR on the cropped regions
        # details_text = pytesseract.image_to_string(details_region)
        # aadharno_text = pytesseract.image_to_string(aadharno_region)
        # txtbbsread = {}
        # bounding_boxes = [txtbbs["aadharno"], txtbbs["details"]]
        # extracted_text = {0:None, 1:None}
        # temp = -1
        # for box in bounding_boxes:
        #     temp+=1
        #     if(all(x == 0 for x in box)):
        #         continue
        #     print(box)
        #     x_min, y_min, x_max, y_max = box
        #     # Extract text using easyocr with bounding box
        #     result = reader.readtext(r"input_image.jpg", allowlist='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ',
        #                             text_threshold=0.7, link_threshold=0.4, low_text=0.4,
        #                             decoder='beamsearch', blocklist='')

        #     # Filter results within the bounding box
        #     filtered_results = [text_info for text_info in result if any(x_min < x < x_max and y_min < y < y_max for (x, y) in text_info[0])]
        #     print(filtered_results)
        #     # Print the extracted text for each bounding box
        #     res = ""
        #     for text_info in filtered_results:
        #         res += (text_info[1] + " ")
        #         print(f'Text in bounding box {box}: {text_info[1]}')
        #     extracted_text[temp] = res
        # print(extracted_text)
        # Print the extracted text
        # print("Details:", details_text)
        # print("Aadhar Number:", aadharno_text)
        # Convert the resulting image to base64
        # buffered = BytesIO()
        # image_with_boxes.save(buffered, format="JPEG")
        # base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        # Add your Roboflow API integration logic here with the processed image data
        with open("output_image.jpg", "rb") as image_file:
            # Convert the image to base64
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        # try:
        #     with open("input_image.jpg", "rb") as image_file:
        #         image = Image.open(image_file)
        #         image.show()
        # except Exception as e:
        #     print(f"Error opening image: {e}")
        response = jsonify({"roboflow_result": base64_image, "detected": t})
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        return response
        # return jsonify({"roboflow_result": 1})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
