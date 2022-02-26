import csv
from dis import dis
import math
import string
from time import sleep
from turtle import distance
from cv2 import sort, threshold
from matplotlib import pyplot as plt
import cv2
from cv2 import cvtColor
import numpy as np

#image = cv2.imread('M40967-1-E.jpg')

def resize_img(scale_percent,img):
   
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    #print('Resized Dimensions : ',resized.shape)
    #cv2.imshow("Resized image", resized)
    return resized

def contain(point1,point2,delta):
    x1=point1[0]
    y1=point1[1]
    w1=point1[2]
    h1=point1[3]
    x2=point2[0]
    y2=point2[1]
    w2=point2[2]
    h2=point2[3]
    if x1<x2-delta and y1<y2-delta and x2+w2+delta<x1+w1 and y2+h2+delta<y1+h1:
            return 1
    else:
            return 0
def contain_2(point1,point2,delta):
    x1=point1[0]
    y1=point1[1]
    x2=point1[2]
    y2=point1[3]
    x3=point2[0]
    y3=point2[1]
    x4=point2[2]
    y4=point2[3]
    if x1<x3-delta and y1<y3-delta and x4+delta<x2 and y4+delta<y2:
            return 1
    else:
            return 0
def overlap(point1,point2,delta):
    x1=point1[0]
    y1=point1[1]
    x2=point1[2]
    y2=point1[3]
    x3=point2[0]
    y3=point2[1]
    x4=point2[2]
    y4=point2[3]

    #   (x1,y1)->(x2,y2)            (x3,y3)->(x4,y4)
    
    if (x1<x3<x2 and y1<y4<y2)or(x1<x3<x2 and y1<y3<y2)or(x1<x4<x2 and y1<y3<y2) or(x1<x4<x2 and y1<y4<y2):
       #right up overlap         #right down overlap        #left down overlap        #left up overlap
        return True       
    else:
            return False
def sort_by_distance(valid_countor):
    distance_array=[]
    for row in valid_countor:
        distance_array.append((round(math.dist((row[0],row[1]),(0,0)),2),row))
    sorted_array = sorted(distance_array,key=lambda x: (x[0]))
    
    sorted_points=[]
    for row in sorted_array:
        sorted_points.append(row[1])
    return sorted_points

def write_number(img,number,x1,y1,x2,y2):
    font= cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText=(((x1+x2)//2)-2,((y1+y2)//2)+2)
    font_Scale= 0.3
    font_Color= (255,0,0)
    thickness= 1
    line_Type= 1

    cv2.putText(img,str(number), 
        bottomLeftCornerOfText, 
        font, 
        font_Scale,
        font_Color,
        thickness,
        line_Type)
    return img
scale_percent = 10 # percent of original size

def close_countors(R1,R2,delta):
    if (R1[0]>R2[2]+delta) or (R1[2]<R2[0]-delta) or (R1[3]<R2[1]) or (R1[1]>R2[3]):
    #if the center of r2 is in r1
    #center=[(R2[2]+R2[0])//2,(R2[3]-R2[1])//2]
    #print(center[0],R1[0],R1[2])
    #print(R1[0]<center[0]<R1[2])
    #print(center[1],R1[1],R1[3])
    #print( R1[1]<center[1]<R1[3])
    #if(R1[0]-delta<center[0]<R1[2]+delta and R1[1]-delta<center[1]<R1[3]+delta):
        return False
    else:
        return True
    
#showing the images with normal scale so my pc screen would be enogh
def find_stuff(imagename,delta=10):
    image=cv2.imread(imagename)
    image=resize_img(scale_percent,image)
    cv2.imshow("image",image)
    
    image_copy=image
    image_copy_copy=np.zeros((image.shape),np.uint8)
    _,image=cv2.threshold(image,125,255,cv2.THRESH_BINARY)
    
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #blur=cv2.GaussianBlur(gray,(3,3),0)
    #cv2.imshow("blur",blur)
    _,thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #cv2.imshow("thresh",thresh)
    kernal=cv2.getStructuringElement(cv2.MORPH_RECT,(3,13))
    dialte=cv2.dilate(thresh,kernal,iterations=1)
    #cv2.imshow("dialte",dialte)
    
    
    cnts=cv2.findContours(dialte,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    cnts=cnts[0] if len(cnts)==2 else cnts[1]
    cnts=sorted(cnts,key=lambda x:cv2.boundingRect(x)[0])
    pointarray=[]
    for c in cnts:
        x,y,w,h =cv2.boundingRect(c)
        pointarray.append([x,y,w,h])
        
    valid_contour=[]
    for c in cnts:    
        x,y,w,h =cv2.boundingRect(c)
        #x=x-2
        w=w+10
        #y=y-2
        h=h+10
        if 400>math.sqrt(math.pow(h,2)+math.pow(w,2))>5 and delta<x<image_copy.shape[1]-delta and delta<y<image_copy.shape[0]-delta:
            count=0
           
            for point1 in pointarray:
                #if point2 in point1
                count+=contain(point1,[x,y,w,h],delta=4)
            
                #print("point",x,y,count)
            if(count<=2):
                #cv2.rectangle(image_copy,(x,y),(x+w,y+h),(0,0,255),2)
                #cv2.rectangle(image_copy_copy,(x,y),(x+w,y+h),(0,0,255),2)
                valid_contour.append([x,y,x+w,y+h])
    #cv2.imshow("box",image_copy)
    #cv2.imshow("red",image_copy_copy)
    #with open('box.csv','w') as f:
     #   write=csv.writer(f)
      #  write.writerows(cnts)
   

    valid_contour_temp=[]
    if True:
        for i,row1 in enumerate(valid_contour):
            temp_row=row1
            for j,row2 in enumerate(valid_contour):
                if((math.dist((row2[0],row2[1]),(row2[2],row2[3])))<35 and i!=j) :
                    close=overlap(row1,row2,delta=2)
                    #print("x")
                else:
                    close=False
                if(close):
                    print("y")
                    print(i,j)
                    print(row1,row2)
                    temp_row[0]=min(row1[0],row2[0])
                    temp_row[1]=min(row1[1],row2[1])
                    temp_row[2]=max(row1[2],row2[2])
                    temp_row[3]=max(row1[3],row2[3])
                    #valid_contour.remove(row1)
            valid_contour_temp.append(temp_row)
                    #valid_contour=valid_contour[:i]+valid_contour[i+1:]
   
    sorted_valid_countor=sort_by_distance(valid_contour_temp)
    
    non_valid_contour_temp2=[]
    for i,row1 in enumerate(valid_contour_temp):
            temp_row2=row1
            for j,row2 in enumerate(valid_contour_temp):
                if(contain_2(row1,row2,delta=0)==1):
                  non_valid_contour_temp2.append(row2)   

    for element in valid_contour_temp:
        if element in non_valid_contour_temp2:
            valid_contour_temp.remove(element)
    
    
    sorted_valid_countor_2=sort_by_distance(valid_contour_temp)

    def add_number(sorted_valid_countor,image_copy,image_copy_copy):
        alpha=0
        for i,row1 in enumerate(sorted_valid_countor):
            for j,row2 in enumerate(sorted_valid_countor):
                #print(row1,row2)
                if row1!=row2:
                    if abs(row1[0]-row2[0])+abs(row1[1]-row2[1])+abs(row1[2]-row2[2])+abs(row1[3]-row2[3])<=alpha:
                    
                                row2[0]=min(row1[0],row2[0])
                                row2[1]=min(row1[1],row2[1])
                                row2[2]=max(row1[2],row2[2])
                                row2[3]=max(row1[3],row2[3])
                                
                                sorted_valid_countor=sorted_valid_countor[:j]+sorted_valid_countor[j+1:]
        
        for i,row in enumerate(sorted_valid_countor):
                image_copy_copy=write_number(image_copy_copy,i,row[0],row[1],row[2],row[3])
                image_copy=write_number(image_copy,i,row[0],row[1],row[2],row[3])
                cv2.rectangle(image_copy,(row[0],row[1]),(row[2],row[3]),(0,0,255),2)
                cv2.rectangle(image_copy_copy,(row[0],row[1]),(row[2],row[3]),(0,0,255),2)
        
        cv2.imshow("after numbers",image_copy)
        cv2.imshow("after numbers_copy",image_copy_copy)
        with open('valid_countor.csv','w',newline='') as f:
            write=csv.writer(f)
            write.writerows(sorted_valid_countor)



    #add_number(sorted_valid_countor,image_copy,image_copy_copy)
    add_number(sorted_valid_countor_2,image_copy,image_copy_copy)
  

    
find_stuff(r"C:\Users\Yuval\Desktop\image pro\FinalProject\images\M40967-1-E.jpg",delta=20)

#find_stuff(r"C:\Users\Yuval\Desktop\image pro\FinalProject\images\M42966-1-E.jpg",delta=20)

#find_stuff(r"C:\Users\Yuval\Desktop\image pro\FinalProject\images\M43025-1-E.jpg",delta=20)

#find_stuff(r"C:\Users\Yuval\Desktop\image pro\FinalProject\images\M43291-1-E.jpg",delta=20)

cv2.waitKey(0)
cv2.destroyAllWindows()
