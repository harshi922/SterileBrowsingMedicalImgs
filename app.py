from cnnImageClassification.pipeline.prediction import PredictionPipeline
from cnnImageClassification.utils.helpers import perform_img_operation
import os 
import cv2
from flask import Flask, request, render_template, redirect, url_for
from flask_cors import CORS, cross_origin
import imutils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

bg = None


os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__,template_folder="templates", static_folder='static_files/') # initializing a flask app
CORS(app)
app_root = os.path.dirname(os.path.abspath(__file__))

def run_avg(image, accumWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, accumWeight)



def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)



class ClientApp:

    def __init__(self):
        self.prediction = PredictionPipeline()


@app.route('/', methods = ["GET"])# route to display the home page
def home():
    message = request.args.get('message')
    return render_template("intro.html", message=message)

@app.route('/about', methods = ["GET"])# route to display the home page
def intro():
    return render_template("about.html")

@app.route('/train', methods = ["POST"])# route to display the home page
@cross_origin()
def trainRoute():
    os.system("dvc repro") # Run dvc repro to train the model
    return redirect(url_for('home', message='Model trained successfully!'))


@app.route('/predict', methods = ["GET","POST"])# route to display the home page
@cross_origin()
def predict():
    if request.method == "POST":
        # Save the medical image obtained from the user
        target = os.path.join(app_root, 'static_files/Images/')
        f = request.files['filea']
        file_name = f.filename or ''
        destination = '/'.join([target, file_name])
        f.save(destination)
   

        #Use video feed of user to determine gesture 
        accumWeight = 0.5
        camera = cv2.VideoCapture(0)
        global bg

        fps = int(camera.get(cv2.CAP_PROP_FPS))
        top, right, bottom, left = 10, 350, 225, 590
        num_frames = 0
        calibrated = False
        k = 0
        while (True):
            # get the current frame
            (grabbed, frame) = camera.read()

            # resize the frame
            frame = imutils.resize(frame, width=700)
            # flip the frame so that it is not the mirror view
            frame = cv2.flip(frame, 1)

            # clone the frame
            clone = frame.copy()

            # get the height and width of the frame
            (height, width) = frame.shape[:2]

            # get the ROI
            roi = frame[top:bottom, right:left]

            # convert the roi to grayscale and blur it
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)

            # to get the background, keep looking till a threshold is reached
            # so that our weighted average model gets calibrated
            if num_frames < 30:
                run_avg(gray, accumWeight)
                if num_frames == 1:
                    print("[STATUS] please wait! calibrating...")
                elif num_frames == 29:
                    print("[STATUS] calibration successfull...")
            else:
                # segment the hand region
                hand = segment(gray)

                # check whether hand region is segmented
                if hand is not None:
                    # if yes, unpack the thresholded image and
                    # segmented region
                    (thresholded, segmented) = hand

                    # draw the segmented region and display the frame
                    cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))

                    # count the number of fingers
                    # fingers = count(thresholded, segmented)
                    if k % (fps / 6) == 0:
                        cv2.imwrite('static_files/Images/Temp.png', thresholded)
                        predictedClass = clApp.prediction.predict(Path('static_files/Images/Temp.png'))
                        cv2.putText(clone, str(predictedClass), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # show the thresholded image
                    cv2.imshow("Thesholded", thresholded)
            k = k + 1
            # draw the segmented hand
            cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)

            # increment the number of frames
            num_frames += 1

            # display the frame with segmented hand
            cv2.imshow("Video Feed", clone)

            # observe the keypress by the user
            keypress = cv2.waitKey(1) & 0xFF

            # if the user pressed "q", then stop looping
            if keypress == ord("q"):
                break
        camera.release()
        cv2.destroyAllWindows()

        # Perform corresponding operation on user uploaded image 
        perform_img_operation(destination,predictedClass= predictedClass)

    return render_template('index.html')
   
if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host='0.0.0.0', port=8080) #local host
    