#Automatic face detection 
import cv2
face_cascade=cv2.CascadeClassifier(r"D:\Courses\AI & ML\haarcascade_frontalface_alt.xml")
img=cv2.imread(r"D:\my stuff\Ganesh puja'19\DSC_0254.JPG")
resized=cv2.resize(img,(720,640))
gray=cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
faces=face_cascade.detectMultiScale(gray)
for(x, y, w, h) in faces:
    cv2.rectangle(resized,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow("group",resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

##face detection in videos
import cv2
vid=cv2.VideoCapture(r"D:\Courses\AI & ML\videoplayback.mp4")
#for webcam pot zeroin videocapture(0) 
face_cascade=cv2.CascadeClassifier(r"D:\Courses\AI & ML\haarcascade_frontalface_alt.xml")

ret,frame=vid.read()
#ret-it is a boolean variable that retuns True if the frame is available.
#frame-it is an image array captured based on frames per second
while vid.isOpened() and ret:#to open particular video.
    ret,frame=vid.read()
    
    if ret:
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray)
        
        for(x, y, w, h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            
        cv2.imshow('video',frame)
        key=cv2.waitKey(10)
        if key==ord('q') or not ret:#getting the ascii value of q
            break

vid.release()#to release the video or else it will run indefinately.
cv2.destroyAllWindows()


#Artificial neural network
#it is a biological network of artificial neurons configured to perform specific tasks.
#it adapts itself during a training period ,based on examples of similar problems even without a desired solutions of a problem.
#able to generelize or handle incomplete data.
#arificial neuron is called perceptron.
#two types of activation function
#playground.tensorflow
#linear and non linear.

