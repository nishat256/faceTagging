import pickle
import cv2
from face_detection import detect_face
from face_onboarding import get_embeddings
from scipy.spatial.distance import cosine

def load_pickled_data():
	"""
	  This function loads pickle object that contains embedding of characters with their details like realname etc.
	"""
	file_name = 'face_to_embedding.pickle'
	with open(file_name,'rb') as file_object:
		db = pickle.load(file_object)
	return db['mapping']

def is_match(known_embedding, candidate_embedding, thresh=0.5):
	# calculate distance between embeddings
	score = cosine(known_embedding, candidate_embedding)
	print(score)
	if score <= thresh:
		return True
	else:
		return False

def recognize_faces_in_frame(image):
	"""
	  This function detects and recognizes faces present in images.
	"""
	db = load_pickled_data()
	faces_present_in_frame = detect_face(image)
	characters_found_in_frame = []
	if faces_present_in_frame:
		embeddings_of_faces_in_frame = get_embeddings(faces_present_in_frame)
		for known_embedding in embeddings_of_faces_in_frame:
			resp_mapping = check_embedding_present_in_db(db,known_embedding)
			if not resp_mapping:
				continue
			characters_found_in_frame.append(resp_mapping)
	return characters_found_in_frame 
		

def check_embedding_present_in_db(db,known_embedding):
	"""
		This is used to find details of detected faces by comparing their embeddings with embeddings present in db.
	"""
	for record in db:
		candidate_embedding = record['embedding']
		resp_match = is_match(known_embedding,candidate_embedding)
		if resp_match:
			return record
	return None

def attach_recognized_character_to_frame(img,faces_in_frame):
	"""
	   This function put detected faces with their details in image provided by user as input.
	"""
	img = make_portion_of_image_black_at_left_side(img)
	start_x = 0
	start_y = 0
	cand_width = int(img.shape[1]/100)*12
	cand_height = int(img.shape[0]/100)*20
	for face in faces_in_frame:
		real_name = face['name']
		character_name = face['character_name']
		cand_img = face['image']
		cand_img = cv2.resize(cand_img,(cand_width,cand_height))
		img = mask_candidate_image_in_frame(img,cand_img,start_x,start_y,cand_height,cand_width,\
 			                                real_name,character_name)
		start_x += cand_height
	return img

def mask_candidate_image_in_frame(img,candidate_image,start_x,start_y,cand_height,cand_width,real_name,character_name):
	"""
	   This function masks candidate image(detected face) into original image one by one.
	"""
	img[start_x:start_x+cand_height,start_y:cand_width] = candidate_image[0:,0:]
	font = cv2.FONT_HERSHEY_SIMPLEX
	org = (cand_width,start_x+50)
	font_scale = 1
	color = (0,0,0)
	thickness = 2
	img = cv2.putText(img,real_name,org,font,font_scale,color,thickness,cv2.LINE_AA)
	org = (cand_width,start_x+75)
	thickness = 1
	img = cv2.putText(img,character_name,org,font,font_scale,color,thickness,cv2.LINE_AA)
	return img

def make_portion_of_image_black_at_left_side(img):
	height = img.shape[0]
	width = img.shape[1]
	trip_width = int(width/100)*12
	trip_height = int(height)
	img[0:trip_height,0:trip_width] = 0
	return img

def tag_faces_in_video():
	""" 
	   INPUT : video path
	   OUTPUT : video with tagged faces in each frame 
	"""
	cap =cv2.VideoCapture('input/input_video.mp4')
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter('output/output_video.avi',fourcc,20.0,(640,480))
	while(cap.isOpened()):
		print("Processing in Progress")
		ret, frame = cap.read()
		if ret == True:
			faces_in_frame = recognize_faces_in_frame(frame)
			if faces_in_frame:
				img = attach_recognized_character_to_frame(frame,faces_in_frame)
				frame = img
			out.write(frame)
		else:
			break
	cap.release()
	cv2.destroyAllWindows()

def tag_faces_in_photo(file_name):
	"""
	   This function is used to tag faces present into image provided as input.
	   INPUT : image path
	   OUTPUT : tagged image
	"""
	img = cv2.imread(file_name)
	faces_in_frame = recognize_faces_in_frame(img)
	# filter unique values from faces_in_frames
	faces_in_frame_unique = {}
	for face in faces_in_frame:
		faces_in_frame_unique[face['name']] = face
	if faces_in_frame:
		img = attach_recognized_character_to_frame(img,list(faces_in_frame_unique.values())[:5])
	cv2.imwrite('output/sample1.jpg',img)
	cv2.imshow('imshow',img)
	cv2.waitKey()
	cv2.destroyWindow('imshow')

if __name__ == "__main__":
	tag_faces_in_photo('input/image1.jpg')