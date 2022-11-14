"""
This the the main file that control all classes
"""
import numpy as np
import cv2
import time
import os
os.chdir('e:\Akram\Courses\Machine Learning & Deep Learning\Projects\Gesture Volume Control')
from HandTrackingModule import HandDetector
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


def draw_volume_bar(img,dist,vol):
    # outer rect
    rect_x1,rect_y1 = 20,50
    rect_x2,rect_y2 = 70,200
    cv2.rectangle(img, (rect_x1,rect_y1),(rect_x2,rect_y2),(0,255,0),2)
    # inner rect
    bar_height = np.interp(dist,(20,180),(55,205))
    cv2.rectangle(img,(rect_x1+5,rect_y2-5),(rect_x2-5,rect_y2-bar_height),(255,0,255),-1)
    cv2.putText(img,str(int(vol))+"%",(20,230),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)




pTime = 0

cap = cv2.VideoCapture(0)

#########################################
detector = HandDetector(detectionCon=0.7)
#########################################


##########################################################################
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volMin,volMax = volume.GetVolumeRange()[:2]
##########################################################################


while True:
    ret,frame = cap.read()
    if ret:
        frame = cv2.flip(frame,1)
        # gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
        "---------- FPS : ----------"
        cTime = time.time()
        fps = int(1/(cTime-pTime))
        pTime = cTime # update
        
        "------- Show fps : -------"
        cv2.putText(frame,f"FPS : {fps}",(30,30),cv2.FONT_HERSHEY_COMPLEX,1,(255),2)
        
        
        # Let's use our module !
        hands = detector.findHands(frame,draw=True)
        
        lmList,bbox = detector.findPosition(frame,handNo=0,draw=True)
        
        if lmList:
            dist,_,pos = detector.findDistance(4, 8, frame)
            print(dist)
            if dist < 30:
                X1,Y1 = pos[:2]
                X2,Y2 = pos[2:4]
                cX,cY = pos[4:]
                cv2.circle(frame, (X1, Y1), 15, (0, 255, 255), cv2.FILLED)
                cv2.circle(frame, (X2, Y2), 15, (0, 255, 255), cv2.FILLED)
                cv2.circle(frame, (cX, cY), 15, (0, 255, 0), cv2.FILLED)
                
            
            vol = np.interp(dist,[20,180],(volMin,volMax))
            volume.SetMasterVolumeLevel(vol, None)
            
            
            cx,cy = pos[:2]
            vol_percent = np.interp(dist,[20,180],(0,100))
            draw_volume_bar(frame,dist,vol_percent)
        

        # show
        cv2.imshow("frame",frame)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()