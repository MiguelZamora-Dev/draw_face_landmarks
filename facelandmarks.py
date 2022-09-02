import cv2
import mediapipe as mp
import os

images_route = './images'
results_route = './results/'
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

image_files = []
if not os.listdir(images_route):
  print('ERROR: Please enter an image in the images folder')
  raise SystemExit

for my_file in os.listdir(images_route):
  image_files.append(images_route + '/' + my_file)

with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5) as face_mesh:
  for idx, my_file in enumerate(image_files):
    image = cv2.imread(my_file)
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
      continue
    annotated_image = image.copy()
    shape = annotated_image.shape 
    _radius = int(shape[1]/1000)
    for face_landmarks in results.multi_face_landmarks:
        for landmark in face_landmarks.landmark:
            relative_x = int(landmark.x * shape[1])
            relative_y = int(landmark.y * shape[0])
            cv2.circle(annotated_image, (relative_x, relative_y), radius=_radius, color=(202, 125, 249), thickness=-1)
    cv2.imwrite(results_route + image_files[idx][9:-4] + '.png', annotated_image)
tempimg = cv2.imread(results_route + image_files[-1][9:-4] + '.png')
new_width = 1000
resized_tempimg = cv2.resize(tempimg, (new_width, int(new_width*annotated_image.shape[0]/annotated_image.shape[1])))
cv2.imshow('Last result', resized_tempimg)
cv2.waitKey(0)
cv2.destroyAllWindows()