# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 20:1 7:29 2022

@author: Alfonso Blanco
"""
######################################################################
# PARAMETERS
######################################################################

dir=""


dirname= "test6Training\\images"
#dirname= "archiveLabeled"
dirnameYolo="runs\\detect\\train9\\weights\\best.pt"
# https://docs.ultralytics.com/python/
from ultralytics import YOLO
model = YOLO(dirnameYolo)
class_list = model.model.names
#print(class_list)


######################################################################

import pytesseract

import numpy as np

import cv2

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

X_resize=220
Y_resize=70

import os
import re

import imutils
#####################################################################
"""
Copied from https://gist.github.com/endolith/334196bac1cac45a4893#

other source:
    https://stackoverflow.com/questions/46084476/radon-transformation-in-python
"""

from skimage.transform import radon

import numpy
from numpy import  mean, array, blackman, sqrt, square
from numpy.fft import rfft

try:
    # More accurate peak finding from
    # https://gist.github.com/endolith/255291#file-parabolic-py
    from parabolic import parabolic

    def argmax(x):
        return parabolic(x, numpy.argmax(x))[0]
except ImportError:
    from numpy import argmax


def GetRotationImage(image):

   
    I=image
    I = I - mean(I)  # Demean; make the brightness extend above and below zero
    
    
    # Do the radon transform and display the result
    sinogram = radon(I)
   
    
    # Find the RMS value of each row and find "busiest" rotation,
    # where the transform is lined up perfectly with the alternating dark
    # text and white lines
      
    # rms_flat does no exist in recent versions
    #r = array([mlab.rms_flat(line) for line in sinogram.transpose()])
    r = array([sqrt(mean(square(line))) for line in sinogram.transpose()])
    rotation = argmax(r)
    #print('Rotation: {:.2f} degrees'.format(90 - rotation))
    #plt.axhline(rotation, color='r')
    
    # Plot the busy row
    row = sinogram[:, rotation]
    N = len(row)
    
    # Take spectrum of busy row and find line spacing
    window = blackman(N)
    spectrum = rfft(row * window)
    
    frequency = argmax(abs(spectrum))
   
    return rotation, spectrum, frequency

#####################################################################
def ThresholdStable(image):
    # -*- coding: utf-8 -*-
    """
    Created on Fri Aug 12 21:04:48 2022
    Author: Alfonso Blanco Garc??a
    
    Looks for the threshold whose variations keep the image STABLE
    (there are only small variations with the image of the previous 
     threshold).
    Similar to the method followed in cv2.MSER
    https://datasmarts.net/es/como-usar-el-detector-de-puntos-clave-mser-en-opencv/https://felipemeganha.medium.com/detecting-handwriting-regions-with-opencv-and-python-ff0b1050aa4e
    """
  
    thresholds=[]
    Repes=[]
    Difes=[]
    
    gray=image 
    grayAnt=gray

    ContRepe=0
    threshold=0
    for i in range (255):
        
        ret, gray1=cv2.threshold(gray,i,255,  cv2.THRESH_BINARY)
        Dife1 = grayAnt - gray1
        Dife2=np.sum(Dife1)
        if Dife2 < 0: Dife2=Dife2*-1
        Difes.append(Dife2)
        if Dife2<22000: # Case only image of license plate
        #if Dife2<60000:    
            ContRepe=ContRepe+1
            
            threshold=i
            grayAnt=gray1
            continue
        if ContRepe > 0:
            
            thresholds.append(threshold) 
            Repes.append(ContRepe)  
        ContRepe=0
        grayAnt=gray1
    thresholdMax=0
    RepesMax=0    
    for i in range(len(thresholds)):
        #print ("Threshold = " + str(thresholds[i])+ " Repeticiones = " +str(Repes[i]))
        if Repes[i] > RepesMax:
            RepesMax=Repes[i]
            thresholdMax=thresholds[i]
            
    #print(min(Difes))
    #print ("Threshold Resultado= " + str(thresholdMax)+ " Repeticiones = " +str(RepesMax))
    return thresholdMax

 
 
# Copied from https://learnopencv.com/otsu-thresholding-with-opencv/ 
def OTSU_Threshold(image):
# Set total number of bins in the histogram

    bins_num = 256
    
    # Get the image histogram
    
    hist, bin_edges = np.histogram(image, bins=bins_num)
   
    # Get normalized histogram if it is required
    
    #if is_normalized:
    
    hist = np.divide(hist.ravel(), hist.max())
    
     
    
    # Calculate centers of bins
    
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.
    
    
    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    
    weight1 = np.cumsum(hist)
    
    weight2 = np.cumsum(hist[::-1])[::-1]
   
    # Get the class means mu0(t)
    
    mean1 = np.cumsum(hist * bin_mids) / weight1
    
    # Get the class means mu1(t)
    
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]
    
    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    
    # Maximize the inter_class_variance function val
    
    index_of_max_val = np.argmax(inter_class_variance)
    
    threshold = bin_mids[:-1][index_of_max_val]
    
    #print("Otsu's algorithm implementation thresholding result: ", threshold)
    return threshold

#########################################################################
def ApplyCLAHE(gray):
#https://towardsdatascience.com/image-enhancement-techniques-using-opencv-and-python-9191d5c30d45
    
    gray_img_eqhist=cv2.equalizeHist(gray)
    hist=cv2.calcHist(gray_img_eqhist,[0],None,[256],[0,256])
    clahe=cv2.createCLAHE(clipLimit=200,tileGridSize=(3,3))
    gray_img_clahe=clahe.apply(gray_img_eqhist)
    return gray_img_clahe

#########################################################################
def FindLicenseNumber (gray, x_offset, y_offset,  License, x_resize, y_resize, \
                       Resize_xfactor, Resize_yfactor, BilateralOption):
#########################################################################

    
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
   
    TotHits=0  
    
    X_resize=x_resize
    Y_resize=y_resize
     
    
    gray=cv2.resize(gray,None,fx=Resize_xfactor,fy=Resize_yfactor,interpolation=cv2.INTER_CUBIC)
    
    gray = cv2.resize(gray, (X_resize,Y_resize), interpolation = cv2.INTER_AREA)
    
    rotation, spectrum, frquency =GetRotationImage(gray)
    rotation=90 - rotation
    #print("Car" + str(NumberImageOrder) + " Brillo : " +str(SumBrightnessLic) +   
    #      " Desviacion : " + str(DesvLic))
    if (rotation > 0 and rotation < 30)  or (rotation < 0 and rotation > -30):
      
        gray=imutils.rotate(gray,angle=rotation)
    
    
    
    
    TabLicensesFounded=[]
    ContLicensesFounded=[]
    
    
    ##########################################################
    #gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
  
    
    #print(gray)
    
    X_resize=x_resize
    Y_resize=y_resize
    print("gray.shape " + str(gray.shape)) 
    Resize_xfactor=1.5
    Resize_yfactor=1.5
    
    rotation, spectrum, frquency =GetRotationImage(gray)
    #print("rotation = "+ str(rotation))
    rotation=90 - rotation
    #print("Car" + str(NumberImageOrder) + " Brillo : " +str(SumBrightnessLic) +   
    #      " Desviacion : " + str(DesvLic))
    
    if (rotation > 0 and rotation < 30)  or (rotation < 0 and rotation > -30):
        #cv2.imshow("Gray", gray)
        #cv2.waitKey(0)
        print(License + " rotate "+ str(rotation))
        gray=imutils.rotate(gray,angle=rotation)
        #cv2.imshow("Gray", gray)
        #v2.waitKey(0)
    
    
    TabLicensesFounded=[]
    ContLicensesFounded=[]
    
    TotHits=0
   
    #https://mattmaulion.medium.com/the-digital-image-an-introduction-to-image-processing-basics-fbdf9fd7f462
    from skimage import img_as_uint
    # for this demo, set threshold to average value
    gray1 = img_as_uint(gray > gray.mean())
    text = pytesseract.image_to_string(gray1, lang='eng',  \
      config='--psm 6 --oem 3')
    text = ''.join(char for char in text if char.isalnum()) 
    text=ProcessText(text)
    if ProcessText(text) != "":
    
        ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
        if text==License:
           print(text +  "  Hit with threshold media"  )
           TotHits=TotHits+1
        else:
            print(License + " detected with threshold media" + text)
    
   
    # https://medium.com/@marizombie/computer-vision-interview-convolutional-neural-network-48e4567e4bed
    kernel = np.array([[1,4,6,4,1], [4,16,24,16,4],[6,24,-476,24,6], [4,16,24,16,4], [1,4,6,4,1]])
    kernel=kernel*(-1)
    kernel=kernel/256
    im = cv2.filter2D(gray, -1, kernel)
    
    text = pytesseract.image_to_string(im, lang='eng',  \
      config='--psm 6 --oem 3')
    text = ''.join(char for char in text if char.isalnum()) 
    text=ProcessText(text)
    if ProcessText(text) != "":
    
        ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
        if text==License:
           print(text +  "  Hit with wikipedia filter gaussian blur"  )
           TotHits=TotHits+1
        else:
            print(License + " detected with wikipedia filter gaussian blur" + text) 
    
    
    
      
    # https://medium.com/@sarcas0705/computer-vision-derivative-over-image-e1020354ddb5
    #sobel
    kernel = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
    gray1 = cv2.filter2D(gray, -1, kernel)
    kernel = np.array([[1,2,1], [0,0,0], [-1,-2,-1]])
    gray2 = cv2.filter2D(gray, -1, kernel) 
    
    gray1=gray1+gray2
   
    gray1 = cv2.threshold(gray1, 180, 255,
                         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]   
    gray1= cv2.GaussianBlur(gray1, (5,5), 0)
    text = pytesseract.image_to_string(gray1, lang='eng',  \
       config='--psm 6 --oem 3')
    text = ''.join(char for char in text if char.isalnum()) 
    text=ProcessText(text)
    if ProcessText(text) != "":
       
        ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
        if text==License:
           print(text +  "  Hit with  filter sobel "  )
           TotHits=TotHits+1
        else:
           print(License + " detected with  filter sobel as "+ text) 
    
  
    #https://towardsdatascience.com/morphological-operations-for-image-preprocessing-in-opencv-in-detail-15fccd1e5745
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 3))
    blackhat1 = cv2.morphologyEx(gray.copy(), cv2.MORPH_BLACKHAT, kernel1)
    text = pytesseract.image_to_string(blackhat1, lang='eng',  \
      config='--psm 6 --oem 3')
    text = ''.join(char for char in text if char.isalnum())
    text=ProcessText(text)
    if ProcessText(text) != "":
    #if Detect_International_LicensePlate(text)== 1:
           TabLicensesFounded, ContLicensesFounded =ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
           if text==License:
              print(text + "  Hit with Filter blackhat1 ")
              TotHits=TotHits+1
           else:
               print(License + " detected with Filter blackhat1"+ text) 
   
    kernel = np.ones((3,3),np.float32)/90
    gray1 = cv2.filter2D(gray,-1,kernel)   
    #gray_clahe = cv2.GaussianBlur(gray, (5, 5), 0) 
    gray_img_clahe=ApplyCLAHE(gray1)
    
    th=OTSU_Threshold(gray_img_clahe)
    max_val=255
    
    ret, o3 = cv2.threshold(gray_img_clahe, th, max_val, cv2.THRESH_TOZERO)
    
    text = pytesseract.image_to_string(o3, lang='eng',  \
      config='--psm 6 --oem 3')
    text = ''.join(char for char in text if char.isalnum()) 
    text=ProcessText(text)
    if ProcessText(text) != "":
    #if Detect_International_LicensePlate(text)== 1:
            TabLicensesFounded, ContLicensesFounded =ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
            if text==License:
               print(text + "  Hit with CLAHE  and THRESH_TOZERO" )
               #TotHits=TotHits+1
            else:
                print(License + " detected with CLAHE and THRESH_TOZERO as "+ text) 
    
    #   Otsu's thresholding
    ret2,gray1 = cv2.threshold(gray,0,255,cv2.THRESH_TRUNC+cv2.THRESH_OTSU)
  
    text = pytesseract.image_to_string(gray1, lang='eng',  \
      config='--psm 6 --oem 3')
    text = ''.join(char for char in text if char.isalnum()) 
    text=ProcessText(text)
    if ProcessText(text) != "":
    
        ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)
        if text==License:
            print(text + "  Hit with Otsu's thresholding of cv2 and THRESH_TRUNC" )
            TotHits=TotHits+1
        else:
            print(License + " detected with Otsu's thresholding of cv2 and THRESH_TRUNC as "+ text)
    
    ####################################################
    # experimental formula based on the brightness
    # of the whole image 
    ####################################################
   
    SumBrightness=np.sum(gray)  
    threshold=(SumBrightness/177600.00) 
        
    for z in range(4,8):
    #for z in range(8,8):
       kernel = np.array([[0,-1,0], [-1,z,-1], [0,-1,0]])
       gray1 = cv2.filter2D(gray, -1, kernel)
              
      
       text = pytesseract.image_to_string(gray1, lang='eng',  \
          config='--psm 6 --oem 3')
       text = ''.join(char for char in text if char.isalnum()) 
       text=ProcessText(text)
       if ProcessText(text) != "":
      
           ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
           if text==License:
              print(text +  "  Hit with Sharpen filter z= "  +str(z))
              TotHits=TotHits+1
           else:
               print(License + " detected with Sharpen filter z= "  +str(z) + " as "+ text) 
    
    gray_img_clahe=ApplyCLAHE(gray)
    
    
    ###################################################################
    # ANTES
    th=OTSU_Threshold(gray_img_clahe)
    max_val=255
    
    threshold=ThresholdStable(gray)
    
    ret, gray1=cv2.threshold(gray,threshold,255,  cv2.THRESH_TRUNC) 
    #gray1 = cv2.GaussianBlur(gray1, (1, 1), 0)
    text = pytesseract.image_to_string(gray1, lang='eng',  \
    config='--psm 6 --oem 3')
    text = ''.join(char for char in text if char.isalnum())
    if Detect_International_LicensePlate(text)== 1:
        ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)    
        if text==Licenses[i]:
            print(text + "  Hit with Stable and THRESH_TRUNC" )
            TotHits=TotHits+1
        else:
            print(Licenses[i] + " detected as "+ text)         
      
    ret, gray1=cv2.threshold(gray,threshold,255,  cv2.THRESH_TOZERO) 
    #gray1 = cv2.GaussianBlur(gray1, (1, 1), 0)
    text = pytesseract.image_to_string(gray1, lang='eng',  \
    config='--psm 6 --oem 3')
    text = ''.join(char for char in text if char.isalnum())
    if Detect_International_LicensePlate(text)== 1:
        ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
        if text==Licenses[i]:
           print(text + "  Hit with Stable and THRESH_TOZERO" )
           TotHits=TotHits+1
        else:
            print(Licenses[i] + " detected as "+ text) 
        
     
    ####################################################
    # experimental formula based on the brightness
    # of the whole image 
    ####################################################
    
    SumBrightness=np.sum(gray)  
    threshold=(SumBrightness/177600.00) 
    
    #####################################################
      
    ret, gray1=cv2.threshold(gray,threshold,255,  cv2.THRESH_TOZERO)
    #gray1 = cv2.GaussianBlur(gray1, (1, 1), 0)
    text = pytesseract.image_to_string(gray1, lang='eng',  \
    config='--psm 6 --oem 3')
    text = ''.join(char for char in text if char.isalnum())
    
    if Detect_International_LicensePlate(text)== 1:
        ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
        if text==Licenses[i]:
           print(text + "  Hit with Brightness and THRESH_TOZERO" )
           TotHits=TotHits+1
        else:
            print(Licenses[i] + " detected as "+ text)
    
    for z in range(5,10):
       if z==6:continue
       if z==10:continue
       kernel = np.array([[0,-1,0], [-1,z,-1], [0,-1,0]])
       gray1 = cv2.filter2D(gray, -1, kernel)
       #gray1 = cv2.GaussianBlur(gray1, (1, 1), 0)       
       text = pytesseract.image_to_string(gray1, lang='eng',  \
       config='--psm 6 --oem 3 -c tessedit_char_whitelist= ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890 ')
      
       text = ''.join(char for char in text if char.isalnum())
       
       if Detect_International_LicensePlate(text)== 1:
           ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
           
           if text==Licenses[i]:
              print(text +  "  Hit with Sharpen filter z = " + str(z) )
              TotHits=TotHits+1
           else:
               print(Licenses[i] + " detected with sharpen filter as "+ text+ " z= "+str(z)) 
    
   
      
    for z in range(10,12):
       
       kernel = np.array([[-1,-1,-1], [-1,z,-1], [-1,-1,-1]])
       gray1 = cv2.filter2D(gray, -1, kernel)
       gray1 = cv2.GaussianBlur(gray1, (1, 1), 2)
       text = pytesseract.image_to_string(gray1, lang='eng',  \
       config='--psm 6 --oem 3 -c tessedit_char_whitelist= ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890 ')
      
       text = ''.join(char for char in text if char.isalnum())
       
       if Detect_International_LicensePlate(text)== 1:
           ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
           if text==Licenses[i]:
              print(text +  "  Hit with Sharpen filter modified z= " +str(z) )
              TotHits=TotHits+1
           else:
               print(Licenses[i] + " detected with Sharpen filter modified as "+ text+ " z= "+str(z)) 
    
    
    #https://anishgupta1005.medium.com/building-an-optical-character-recognizer-in-python-bbd09edfe438
     
    bilateral = cv2.bilateralFilter(gray,9,75,75)
    median = cv2.medianBlur(bilateral,3)
    
    adaptive_threshold_mean = cv2.adaptiveThreshold(median,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
        cv2.THRESH_BINARY,11,2)
        
    text = pytesseract.image_to_string( adaptive_threshold_mean , lang='eng',  \
        config='--psm 6 --oem 3') 
    text = ''.join(char for char in text if char.isalnum())
    if Detect_International_LicensePlate(text)== 1:
        ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text)   
        if text==Licenses[i]:
           print(text + "  Hit with  adaptive_threshold_mean " )
           TotHits=TotHits+1
        else:
            print(Licenses[i] + " detected with  adaptive_threshold_mean  "+ text)          
            
         
    ################################################################
    return TabLicensesFounded, ContLicensesFounded

 ########################################################################
def loadimagesRoboflow (dirname):
 #########################################################################
 # adapted from:
 #  https://www.aprendemachinelearning.com/clasificacion-de-imagenes-en-python/
 # by Alfonso Blanco Garc??a
 ########################################################################  
     imgpath = dirname + "\\"
     
     images = []
     Licenses=[]
     
     
     print("Reading imagenes from ",imgpath)
     NumImage=-2
     
     Cont=0
     for root, dirnames, filenames in os.walk(imgpath):
         
         
         NumImage=NumImage+1
         
         for filename in filenames:
             
             if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                 
                 
                 filepath = os.path.join(root, filename)
                 License=filename[:len(filename)-4]
                 # International license plate is NNNNAAA
                 #if Detect_International_LicensePlate(License)== -1: continue
                 
                 
                 image = cv2.imread(filepath)
                 #image=cv2.resize(image,(416,416)) 
                 #image=cv2.resize(image, (640,640))
                 
                 #Color Balance
                #https://blog.katastros.com/a?ID=01800-4bf623a1-3917-4d54-9b6a-775331ebaf05
                
                 img = image
                    
                 r, g, b = cv2.split(img)
                
                 r_avg = cv2.mean(r)[0]
                
                 g_avg = cv2.mean(g)[0]
                
                 b_avg = cv2.mean(b)[0]
                
                 
                 # Find the gain occupied by each channel
                
                 k = (r_avg + g_avg + b_avg)/3
                
                 kr = k/r_avg
                
                 kg = k/g_avg
                
                 kb = k/b_avg
                
                 
                 r = cv2.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
                
                 g = cv2.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
                
                 b = cv2.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)
                
                 
                 balance_img = cv2.merge([b, g, r])
                 
                 image=balance_img
                 
                 
                 images.append(image)
                 Licenses.append(License)
                 
                 
                
                 Cont+=1
     
     return images, Licenses


# COPIED FROM https://programmerclick.com/article/89421544914/
def gamma_trans (img, gamma): # procesamiento de la funci??n gamma
         gamma_table = [np.power (x / 255.0, gamma) * 255.0 for x in range (256)] # Crear una tabla de mapeo
         gamma_table = np.round (np.array (gamma_table)). astype (np.uint8) #El valor del color es un n??mero entero
         return cv2.LUT (img, gamma_table) #Tabla de b??squeda de color de imagen. Adem??s, se puede dise??ar un algoritmo adaptativo de acuerdo con el principio de homogeneizaci??n de la intensidad de la luz (color).
def nothing(x):
    pass
  
def Detect_International_LicensePlate(Text):
    if len(Text) < 3 : return -1
    for i in range(len(Text)):
        if (Text[i] >= "0" and Text[i] <= "9" )   or (Text[i] >= "A" and Text[i] <= "Z" ):
            continue
        else: 
          return -1 
       
    return 1

def ProcessText(text):
    """
    if len(text)  > 9:
      text=text[len(text)-9]
    else:
        if len(text)  > 8:
          text=text[len(text)-8]
        else:
    """
    if len(text)  > 7:
       text=text[len(text)-7:] 
    if Detect_International_LicensePlate(text)== -1: 
       return ""
    else:
       return text  

def ApendTabLicensesFounded (TabLicensesFounded, ContLicensesFounded, text):
    
    SwFounded=0
    for i in range( len(TabLicensesFounded)):
        if text==TabLicensesFounded[i]:
            ContLicensesFounded[i]=ContLicensesFounded[i]+1
            SwFounded=1
            break
    if SwFounded==0:
       TabLicensesFounded.append(text) 
       ContLicensesFounded.append(1)
    return TabLicensesFounded, ContLicensesFounded


# ttps://medium.chom/@chanon.krittapholchai/build-object-detection-gui-with-yolov8-and-pysimplegui-76d5f5464d6c
def DetectLicenseWithYolov8 (img):
  
    
    TabcropLicense=[]
    results = model.predict(img)
    
    result=results[0]
    
    xyxy= result.boxes.xyxy.numpy()
    confidence= result.boxes.conf.numpy()
    class_id= result.boxes.cls.numpy().astype(int)
    # Get Class name
    class_name = [class_list[x] for x in class_id]
    # Pack together for easy use
    sum_output = list(zip(class_name, confidence,xyxy))
    # Copy image, in case that we need original image for something
    out_image = img.copy()
    for run_output in sum_output :
        # Unpack
        #print(class_name)
        label, con, box = run_output
        if label == "vehicle":continue
        cropLicense=out_image[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
        #cv2.imshow("Crop", cropLicense)
        #cv2.waitKey(0)
        TabcropLicense.append(cropLicense)
    return TabcropLicense


###########################################################
# MAIN
##########################################################

imagesComplete, Licenses=loadimagesRoboflow(dirname)

print("Number of imagenes : " + str(len(imagesComplete)))

print("Number of   licenses : " + str(len(Licenses)))

ContDetected=0
ContNoDetected=0
TotHits=0
TotFailures=0
with open( "LicenseResults.txt" ,"w") as  w:
    for i in range (len(imagesComplete)):
          
            gray=imagesComplete[i]
            
            License=Licenses[i]
            
            #if License < "EATTHE":
            #    print("SALTA " + License)
            #    continue
            TabImgSelect =DetectLicenseWithYolov8(gray)
            if TabImgSelect==[]:
                print(License + " NON DETECTED")
                ContNoDetected=ContNoDetected+1 
                continue
            else:
                ContDetected=ContDetected+1
                print(License + " DETECTED ")
                
            gray=TabImgSelect[0]  
           
            x_off=3
            y_off=2
            
            x_resize=220
            y_resize=70
            
            Resize_xfactor=1.78
            Resize_yfactor=1.78
            
            ContLoop=0
            
            SwFounded=0
            
            BilateralOption=0
            
            TabLicensesFounded, ContLicensesFounded= FindLicenseNumber (gray, x_off, y_off,  License, x_resize, y_resize, \
                                   Resize_xfactor, Resize_yfactor, BilateralOption)
              
            
            print(TabLicensesFounded)
            print(ContLicensesFounded)
            
            ymax=-1
            contmax=0
            licensemax=""
          
            for y in range(len(TabLicensesFounded)):
                if ContLicensesFounded[y] > contmax:
                    contmax=ContLicensesFounded[y]
                    licensemax=TabLicensesFounded[y]
            
            if licensemax == License:
               print(License + " correctly recognized") 
               TotHits+=1
            else:
                print(License + " Detected but not correctly recognized")
                TotFailures +=1
            print ("")  
            lineaw=[]
            lineaw.append(License) 
            lineaw.append(licensemax)
            lineaWrite =','.join(lineaw)
            lineaWrite=lineaWrite + "\n"
            w.write(lineaWrite)
              
print("")           
print("Total Hits = " + str(TotHits ) + " from " + str(len(imagesComplete)) + " images readed")


      
                 
        