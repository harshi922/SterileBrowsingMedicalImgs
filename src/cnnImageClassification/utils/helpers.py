import cv2
from pathlib import Path

def run_avg(image, accumWeight,bg):
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, accumWeight)


    # find the absolute difference between background and current frame
    
def perform_img_operation(destination, predictedClass):
    image1=cv2.imread(destination)
    if predictedClass == "Blank":       
            resized = cv2.resize(image1, (200, 200))
            cv2.imshow("Resizing", resized)
            cv2.imwrite(Path('static_files/Images/Image_Operated.png'), resized)
            key=cv2.waitKey(3000)
            if (key & 0xFF) == ord("1"):
                cv2.destroyWindow("Resizing")
    elif predictedClass== "Ok":   
            resized = cv2.rectangle(image1, (480, 170), (650, 420), (0, 0, 255), 2)
            cv2.imshow("Rectangle", image1)
            cv2.imwrite(Path('static_files/Images/Image_Operated.png'), resized)
            cv2.waitKey(0)
            key=cv2.waitKey(3000)
            if (key & 0xFF) == ord("0"):
                cv2.destroyWindow("Rectangle")         
    elif predictedClass=='Thumbs Up':
            print("Thumbs up")
            (h, w, d) = image1.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, -45, 1.0)
            rotated = cv2.warpAffine(image1, M, (w, h))
            cv2.imshow("OpenCV Rotation", rotated)
            key=cv2.waitKey(3000)
            if (key & 0xFF) == ord("2"):
                cv2.destroyWindow("OpenCV Rotation")
    elif predictedClass=='Thumbs Down':
            blurred = cv2.GaussianBlur(image1, (21, 21), 0)
            cv2.imshow("Blurred", blurred)
            cv2.imwrite(Path('static_files/Images/Image_Operated.png'), blurred)
            key=cv2.waitKey(3000)
            if (key & 0xFF) == ord("3"):
                cv2.destroyWindow("Blurred")
    elif predictedClass=='Fist':
            resized = cv2.resize(image1, (400, 400))
            cv2.imshow("Fixed Resizing", resized)
            print("fixed")
            cv2.imwrite(Path('static_files/Images/Image_Operated.png'), resized)
            key=cv2.waitKey(3000)
            if (key & 0xFF) == ord("4"):
                cv2.destroyWindow("Fixed Resizing")
    elif predictedClass=='High Five':
            gray = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
            cv2.imshow("OpenCV Gray Scale", gray)
            print("fixed")
            cv2.imwrite(Path('static_files/Images/Image_Operated.png'), gray)
            key=cv2.waitKey(3000)
            if (key & 0xFF) == ord("5"):
                cv2.destroyWindow("OpenCV Gray Scale")