#import necessary modules
import cv2
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity as ssim
from flask import Flask,render_template, request

#MODULE: HISTOGRAM

#Function to extract color histograms as features from a video
def extract_color_histograms(video_path, bins=(8, 8, 8)):
    cap = cv2.VideoCapture(video_path)
    features = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.append(hist)
    cap.release()
    return np.array(features)

#Function to calculate video similarity using color histograms
def calculate_video_similarity(video1_path, video2_path):
    vf1 = extract_color_histograms(video1_path)
    vf2 = extract_color_histograms(video2_path)
    similarity_matrix = cosine_similarity(vf1, vf2)
    average_similarity = np.mean(similarity_matrix)
    return average_similarity

#MODULE: SSIM

#Function to calculate SSIM between two videos
def calculate_ssim(video1, video2):
    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)
    ssim_scores = []
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break
        (H, W, C) = frame1.shape
        frame2 = cv2.resize(frame2, (W, H))
        # Convert frames to grayscale
        frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        # Calculate SSIM between frames
        score = ssim(frame1_gray, frame2_gray)
        ssim_scores.append(score)
    cap1.release()
    cap2.release()
    # Calculate average SSIM score
    avg_ssim = np.mean(ssim_scores)
    return avg_ssim

#MODULE: MAIN

def count_videos(db_path):
    v_count = 0
    for _, _, videos in os.walk(db_path):
        v_count += len(videos)
    return v_count

def get_file_names(db_path):
    vid_list = []
    for v_name in os.listdir(db_path):
        v_path = os.path.join(db_path, v_name)
        if os.path.isfile(v_path) and v_path.endswith((".mp4", ".avi", ".mov")):
            vid_list.append(v_path)
    return vid_list

def main(input_link,db_link):
    ssim_results=[]
    histo_results=[]
    # Load the input video
    input_cap = cv2.VideoCapture(input_link)
    if not input_cap.isOpened():
        return render_template("error.html")
    video_list=get_file_names(db_link)
    for video_path in video_list:
        video_cap = cv2.VideoCapture(video_path)
        if not video_cap.isOpened():
            continue
        h_score=calculate_video_similarity(input_link, video_path)
        histo_results.append([h_score,video_path])
        s_score=calculate_ssim(input_link,video_path)
        ssim_results.append([s_score,video_path])
    input_cap.release()
    histo_results.sort(reverse=True)
    ssim_results.sort(reverse=True)
    temp1,temp2=histo_results[0][0],ssim_results[0][0]
    if temp1<=0 and temp2<=0:
        return []
    return ([histo_results[0][1],ssim_results[0][1]])

#MODULE: FLASK

#Creating Flask Instance
sim_functions=Flask(__name__)

@sim_functions.route("/")
@sim_functions.route("/home")
def home():
    return render_template("index.html")

@sim_functions.route("/result",methods=["post","get"])
def result():
    output=request.form.to_dict()
    link1=output["Input"]
    link2=output["DB"]
    max_sim_video=main(link1,link2)
    h=max_sim_video[0]
    s=max_sim_video[1]
    return render_template("results.html",h=h,s=s)

if __name__=='__main__':
    sim_functions.run(debug=True,port=5001)