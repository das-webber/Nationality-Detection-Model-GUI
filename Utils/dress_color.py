# %%
import cv2
import numpy as np
from sklearn.cluster import KMeans

BASIC_COLORS = {
    "Black" : (0,0,0), "White" : (255,255,255), "Red" : (255,0,0), "Green" : (0,255,0), "Blue" : (0,0,255), "Yellow" : (255,255,0),
    "Cyan" : (0,255,255), "Magenta" : (255,0,255), "Orange" : (255,128,0), "purple": (128,0,128), "Gray" : (128,128,128), "Brown" : (128,75,0)
}

def nearest_colour_name(rgb):
    r,g,b= rgb
    best,best_d= None, float("inf")
    for name, (cr,cg,cb) in BASIC_COLORS.items():
        d= (r-cr)**2 + (g-cg)**2 + (b-cb)**2
        if d < best_d:
            best_d, best = d, name
    return best

def dominant_colour_kmean(bgr_img, k=3):
    small = cv2.resize(bgr_img, (64,64), interpolation= cv2.INTER_AREA)
    data= small.reshape((-1,3))
    kmeans=KMeans(n_clusters = k, random_state=0).fit(data)
    centers = kmeans.cluster_centers_.astype(int)
    counts = np.bincount(kmeans.labels_)
    dominant = centers[counts.argmax()]
    b,g,r = int(dominant[0]), int(dominant[1]), int(dominant[2])

    return (r,g,b)

def estimate_dress_colour(frame_bgr, face_box= None):
    h,w = frame_bgr.shape[:2]
    if face_box is not  None:
        x1, y1, x2, y2 = face_box
        top= max(0,y2)
        bottom = min(h, y2 + int(1.6*(y2-y1)) )
        left = max(0, x1 - int((x2-x1)*0.5))
        right = min(w, x2 + int((x2-x1)*0.5))
        crop = frame_bgr[top:bottom, left:right]
    else:
        crop = frame_bgr[h//2 : min(h, h//2+h//3), w//4 : 3*w//4]
    
    if crop is None or crop.size==0:
        crop = frame_bgr
    
    rgb= dominant_colour_kmean(crop,k=3)
    name= nearest_colour_name(rgb)
    return name



# %%



