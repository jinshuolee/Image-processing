from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import cv2
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
np.set_printoptions(threshold=sys.maxsize)


#高斯模糊
def Gaussian(img):
    kernel_size = 5
    blur = cv2.GaussianBlur(img,(kernel_size,kernel_size),0)
    return blur

#霍夫轉換
def HoughLinesP(img,canny):
    rho = 1     #距離精度
    theta = np.pi/180   #角度精度
    threshold = 50      #累加平面的閾值參數
    lines = cv2.HoughLinesP(canny,rho,theta,threshold,maxLineGap=50,minLineLength=20)
    for line in lines:
        x1,y1,x2,y2 = line[0]
        img_line = cv2.line(img,(x1,y1),(x2,y2),(0,0,255),3)
    return img_line

#ROI感興趣區域
def ROI(canny):
    roi_mask = np.zeros(canny.shape,dtype=np.uint8)
    ROI = np.array([[(0,720),(1280,720),(1280,350),(0,350)]])
    cv2.fillPoly(roi_mask,ROI,255)
    
    ROI_canny = cv2.bitwise_and(canny,roi_mask)
    return ROI_canny

###=================================================

#-----------------輸出圖片
img = cv2.imread('road.jpg',-1)
blur = Gaussian(img)
hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)

#遮罩
low_white = np.array([0,0,230])
high_white = np.array([100,30,255])
mask = cv2.inRange(hsv,low_white,high_white)

canny = cv2.Canny(mask,50,150)

img = HoughLinesP(img,canny)
    
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
blur = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
canny = cv2.cvtColor(canny, cv2.COLOR_BGR2RGB)


plt.figure()
plt.imshow(img,vmin=0,vmax=255)
plt.axis('off')
Image.fromarray(img).save('img1.jpg')

plt.figure()
plt.imshow(blur,vmin=0,vmax=255)
plt.axis('off')
Image.fromarray(blur).save('blur.jpg')

plt.figure()
plt.imshow(mask,vmin=0,vmax=255)
plt.axis('off')
Image.fromarray(mask).save('mask.jpg')

plt.figure()
plt.imshow(canny,vmin=0,vmax=255)
plt.axis('off')
Image.fromarray(canny).save('canny.jpg')


#視角透視轉換
def warp(img):
    img_size = (img.shape[1], img.shape[0])
    src = np.float32([[570, 300],[770, 300],[920, 420],[420, 420]])
    dst = np.float32([[200, 0], [1000, 0],[1000, 421],[200, 421]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size)
    return warped

filename = 'lane.png'
img2 = cv2.imread(filename)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

warped = warp(img2)

pts = np.array([[570, 300],[770, 300],[920, 420],[420, 420]])
img3 = cv2.polylines(img2,[pts],True,(0,0,255),5)


plt.figure()
plt.imshow(img3,vmin=0,vmax=255)
plt.axis('off')
Image.fromarray(img3).save('img3.jpg')

plt.figure()
plt.imshow(warped,vmin=0,vmax=255)
plt.axis('off')
Image.fromarray(warped).save('warped.jpg')


###================================================
##-------------輸出影片

cap = cv2.VideoCapture('road_car_view_2.mp4')
fig,ax = plt.subplots()

num = 0
while(cap.isOpened()):
    ret,frame = cap.read()
    
    if(ret == True):
        
        frame = np.array(frame)
        [h,w,d] = frame.shape
        
        blur = Gaussian(frame)
        hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)

        low_yellow = np.array([15,90,140])
        high_yellow = np.array([50,160,255])
        mask = cv2.inRange(hsv,low_yellow,high_yellow)

        canny = cv2.Canny(mask,50,150)
        
        ROI_canny=ROI(canny)
        
        ax.cla()
        try:
            frame = HoughLinesP(frame,ROI_canny)
                
        except:
            pass
        
    ###---------------------執行影片時關掉 ---------------------
    
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pts = np.array([[500, 500],[700, 500],[1100, 650],[100, 650]])
        frame = cv2.polylines(frame,[pts],True,(0,0,255),5)
        plt.axis('off')
        ax.imshow(frame)
        plt.savefig('OutFig/Out_'+str(num)+'.jpg')  #建一個Outfig資料夾
        
        num = num+1;
    else:
        
        break
    
    #### ------------------------影片-------------------------
    
        ##正常視角影片
        # ax.cla()
        # pts = np.array([[400, 500],[900, 500],[1100, 650],[200, 650]])
        # frame2 = cv2.polylines(frame,[pts],True,(255,0,0),5)
        # cv2.imshow('frame2',frame2)
        # frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        # Image.fromarray(frame2.astype(np.uint8)).save('frame2.jpg')
        
        
        ##視角轉換影片
        # ax.cla()
        # frame_size = (frame.shape[1], frame.shape[0])
        # src = np.float32([[400, 500],[900, 500],[1100, 650],[200, 650]])
        # dst = np.float32([[200, 0], [1000, 0],[1000, 720],[200, 720]])
        # M = cv2.getPerspectiveTransform(src, dst)
        # frame3 = cv2.warpPerspective(frame, M, frame_size)
        # cv2.imshow('frame3',frame3)
        # frame3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGB)
        # Image.fromarray(frame3.astype(np.uint8)).save('frame3.jpg')
        
        
    # """按q結束影片"""
    # if cv2.waitKey(1) == ord('q'):
    #     ax.imshow(frame2)
    #     # ax.imshow(frame3)
    #     plt.axis('off')
    #     break


    ####-------------------------------------------------------
    

cap.release()
cv2.destroyAllWindows()