import sys
import os
import dlib
import glob
import cv2

if len(sys.argv) != 3:
    print(
        "Give the path to the trained shape predictor model as the first "
        "argument and then the directory containing the facial images.\n"
        "For example, if you are in the python_examples folder then "
        "execute this program by running:\n"
        "    ./face_landmark_detection.py shape_predictor_68_face_landmarks.dat ../examples/faces\n"
        "You can download a trained facial shape predictor from:\n"
        "    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    exit()

scale_percent = 25



predictor_path = sys.argv[1]
faces_folder_path = sys.argv[2]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

dict = {}
for count, f in enumerate(glob.glob(os.path.join(faces_folder_path, "*.jpg"))):
    print("Processing file: {}".format(f))
    img = dlib.load_rgb_image(f)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dsize = (width, height)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        
        d_smutek = shape.part(48).y-shape.part(66).y
        print(d_smutek)
        d_smutek2 = shape.part(54).y-shape.part(66).y
        print(d_smutek2)
        d_radosc = shape.part(48).y-shape.part(62).y
        print(d_radosc)
        d_radosc2 = shape.part(54).y-shape.part(62).y
        print(d_radosc2)
        stosunek1 = (shape.part(66).y-shape.part(62).y)/(shape.part(64).x-shape.part(60).x)
        print(stosunek1)
        stosunek2 = 0.5*((shape.part(40).y-shape.part(38).y)/(shape.part(39).x-shape.part(36).x) + (shape.part(46).y-shape.part(44).y)/(shape.part(45).x-shape.part(42).x))
        print(stosunek2)
        
        norma  = shape.part(54).x-shape.part(48).x

        for i in range(68):
            img = cv2.putText(img, str(i), (shape.part(i).x, shape.part(i).y), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,255), 2)
        
        if 0.5*(d_smutek + d_smutek2)/norma>10/norma:  
            string = "Smutek"
            img = cv2.putText(img, string, (50,150), cv2.FONT_HERSHEY_SIMPLEX, 7,(0,0,255), 4)
        elif d_radosc<0 and d_radosc2<0: 
            string = "Radosc"
            img = cv2.putText(img, string, (50,150), cv2.FONT_HERSHEY_SIMPLEX, 7,(0,0,255), 4)
        elif stosunek1 > 0.21 or stosunek2 > 0.34:
            string ="Zdziwko"
            img = cv2.putText(img, string, (50,150), cv2.FONT_HERSHEY_SIMPLEX, 7,(0,0,255), 4)
        else:
            string ="Neutral"
            img = cv2.putText(img, string, (50,150), cv2.FONT_HERSHEY_SIMPLEX, 7, (0,0,255), 4)
        print(string)
        dict[str(f)] = string
        
    output = cv2.resize(img, dsize)
    cv2.imshow("", output);

    var = cv2.waitKey(2000)
    if var == ord("q"): break
    

cv2.destroyAllWindows()
print(dict)