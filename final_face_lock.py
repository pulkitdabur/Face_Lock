import cv2
import face_recognition as f_p
import numpy as np

cam = cv2.VideoCapture(0)

cv2.namedWindow("Capture")

count=1

while True:
    ret, frame = cam.read()
    cv2.imshow("Capture", frame)
    if not ret:
        break

    k = cv2.waitKey(1)

    # ESC pressed for exiting the camera
    if k%256 == 27:
        print("Closing...")
        break

    # SPACE pressed for saving the image
    elif k%256 == 32:
        if count == 1 :
            img_name="User_capture_image.jpg"
            cv2.imwrite(img_name, frame)
            print("Captured")
            count = count + 1
        elif count>1:
            print("already exist")
            break



cam.release()

cv2.destroyAllWindows()






# PART 2
video_capture=cv2.VideoCapture(0)

#known image file of the user taken from the capture code from the opencv
image_file=f_p.load_image_file("User_capture_image.jpg")


#finding the face encodings of the known face of the user.
# Here num_jitters gives us the more acurate encodings, more jitter more acurate
image_face_encoding=f_p.face_encodings(image_file)[0]

known_face_encodings=[image_face_encoding]

face_locations=[]
face_encodings=[]
process_this_frame= True
sample_num=0

while True:
    ret,frame=video_capture.read()

    #breaking the frame into 1/4 parts for fastrer processing
    small_frame=cv2.resize(frame,(0,0),fx=0.25,fy=0.25)

    #converrting bgr(opencv gives output) to rgb
    rgb_small_frame=small_frame[:,:,::-1]
    sample_num=sample_num+1
    if process_this_frame:
        face_locations=f_p.face_locations(rgb_small_frame)
        face_encodings=f_p.face_encodings(rgb_small_frame,face_locations)

        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = f_p.compare_faces(known_face_encodings, face_encoding)
            face_distances = f_p.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

    process_this_frame = not process_this_frame

    for (top, right, bottom, left) in (face_locations):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 255), 3)


    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if sample_num>20:
        break
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break

if matches[best_match_index] == True:
    print("Unlocked")
# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()