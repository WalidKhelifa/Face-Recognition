#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2020

@author: Walid Khelifa
"""
#importing the required libraries
import cv2
import face_recognition

#loading the image to detect
original_image = cv2.imread('images/testing/walid.png')

#load the sample images and get the 128 face embeddings from them
walid_image = face_recognition.load_image_file('images/walid1.jpg')
walid_face_encodings = face_recognition.face_encodings(walid_image)[0]

kenzo_image = face_recognition.load_image_file('images/kenzo.jpg')
kenzo_face_encodings = face_recognition.face_encodings(kenzo_image)[0]

#save the encodings and the corresponding labels in seperate arrays in the same order
known_face_encodings = [walid_face_encodings,kenzo_face_encodings]
known_face_names = ["Walid Khelifa","Yacine Kenzo"]

#load the unknown image to recognize faces in it
image_to_recognize = face_recognition.load_image_file('images/testing/walid.png')

#detect all faces in the image
#arguments are image,no_of_times_to_upsample, model
all_face_locations = face_recognition.face_locations(image_to_recognize,model='hog')
#detect face encodings for all the faces detected
all_face_encodings = face_recognition.face_encodings(image_to_recognize,all_face_locations)

#print the number of faces detected
print('There are {} no of faces in this image'.format(len(all_face_locations)))

#looping through the face locations and the face embeddings
for current_face_location,current_face_encoding in zip(all_face_locations,all_face_encodings):
    #splitting the tuple to get the four position values of current face
    top_pos,right_pos,bottom_pos,left_pos = current_face_location
    
    
    #find all the matches and get the list of matches
    all_matches = face_recognition.compare_faces(known_face_encodings, current_face_encoding)
   
    #string to hold the label
    name_of_person = 'Unknown face'
    
    #check if the all_matches have at least one item
    #if yes, get the index number of face that is located in the first index of all_matches
    #get the name corresponding to the index number and save it in name_of_person
    if True in all_matches:
        first_match_index = all_matches.index(True)
        name_of_person = known_face_names[first_match_index]
    
    #draw rectangle around the face    
    cv2.rectangle(original_image,(left_pos,top_pos),(right_pos,bottom_pos),(0,0,255),2)
    
    #display the name as text in the image
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(original_image, name_of_person, (left_pos+30,bottom_pos+35), font, 1, (255,255,255),2)
    
    #display the image
    cv2.imshow("Faces Identified",original_image)


