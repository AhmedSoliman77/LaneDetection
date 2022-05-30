import cv2
import numpy as np


import matplotlib.pyplot as plt
#function to crop image to the region of interest.




def masking(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

#function to draw the lines on a blank image then merging the image with the captured frame.

def draw_the_lines(img, lines):

    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)




    for line in lines:
        for x1, y1, x2, y2 in line:

            x1,y1,x2,y2=line[0]
            cv2.line(blank_image, (x1,y1), (x2,y2), (255, 0, 255), thickness=5)
            if x1<600:
                poly = np.array([(x2, y2), (x1, 660), (x1+260, 660), (x2+55,y2)], np.int32)
                vertices = [(595, y2), (450, 660), (850, 660), (650, y2)]
            if x1>600:
                poly = np.array([(x1-55, y1), (x2-260, y2), (x2, y2), (x1,y1)], np.int32)
                vertices = [(595, y1), (450, 660), (850, 660), (650, y1)]
            center = np.array(vertices, np.int32)
            cv2.fillConvexPoly(blank_image, poly, 250)
            cv2.fillConvexPoly(blank_image, center, 250)
    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img


def function(image):

    #getting image information.

    #print(image.shape)
    height = image.shape[0]
    width = image.shape[1]
    img_size=(width,height)

    region_of_interest_vertices = [
        (560,450),(310,660),(1200,660),(800,470) #identifying coordiantes of the region of interest
    ]



    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)   #convert to grayscale
    canny_image = cv2.Canny(gray_image, 200, 250) #get the images edges using canny
    cropped_image = masking(canny_image,
                    np.array([region_of_interest_vertices], np.int32),) #crop image to region of interest

    lines = cv2.HoughLinesP(cropped_image,
                            rho=7,
                            theta=np.pi/180,
                            threshold=150,
                            lines=np.array([]),
                            minLineLength=0.5,
                            maxLineGap=300) #using hough lines probabilistic transform to get lane lines



    """" cascade_src = 'cars.xml'
       rectangles = []
       cascade = cv2.CascadeClassifier(cascade_src)
       cars = cascade.detectMultiScale(gray_image,1.6,4)

       for (x, y, w, h) in cars:
           cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

       """

    image_with_lines = draw_the_lines(image, lines)
    return image_with_lines


# initialize object detection

capture = cv2.VideoCapture('project_video.mp4')

while capture.isOpened():
    ret, frame = capture.read() #capture frames of the video




    frame = function(frame) #process the code on the frame
    cv2.imshow('frame', frame) #show the processed frame

    if cv2.waitKey(17) & 0xFF == ord('q'):
        break  #use keyboard key "q" to stop the video

capture.release()
cv2.destroyAllWindows()
