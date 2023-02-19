LicensePlate_Yolov8_MaxFilters: recognition of car license plates that are detected by Yolov8 and recognized with pytesseract after processing with a pipeline of filters choosing the most repeated car license plate.
In a test with 21 images, 17 hits are achieved

Requirements:

yolo must be installed, to do so follow the instructions indicated in:
  https://learnopencv.com/ultralytics-yolov8/#How-to-Use-YOLOv8?

In this case, the option to install on the computer itself was chosen.

pip install ultralytics

also must be installed the normal modules in vomputer vision: pytesseract, numpy, cv2, os, re, imutils,  parabolic

Functioning:


Download the project to a folder on disk.

Download to that folder the roboflow files that will be used for training:

https://public.roboflow.com/object-detection/license-plates-us-eu/3

In that folder you should find the train and valid folders necessary to build the model

To ensure the version, the used roboflo file roboflow.zip is attached

Unzip the file with the test images test6Training.zip, taking into account when unzipping you can create a folder
test6Training inside the test6Training folder,there should be only one test6Training folder, otherwise you will not find the
test images

Model Train:

the train and valid folders of the roboflow folder, resulting from the unziping of robflow.zip, must be placed in the same directory where the execution program LicensePlateYolov8Train.py is located, according to the requirements indicated in license_data.yaml

run the program

LicensePlateYolov8Train.py

which only has 4 lines, but the line numbered 11 should indicate the full path where the license_data.yaml file is located.

Running from a simple laptop, the 100 epochs of the program will take a long time, but you can always pull the lid off the laptop and
continue the next day. As obtaining best.pt is problematic, the one used in the project tests is attached, adjusting the route of instruction 15 in GetNumberSpanishLicensePlate_Yolov8_MaxFilters

As a result, inside the download folder, the directory runs\detect\trainN\weights( where in trainN, N indicates
  the last train directory created) in which the best.pt file is located, which is the base of the model and
  which is referenced in line 15 of the GetNumberSpanishLicensePlate_Yolov8_MaxFilters.py program (modify the route, the name of trainN, so that it points to the last train and best.pt created

As obtaining best.pt is problematic, the one used in the project tests is attached,it must be  adjusted the route of instruction 15 in GetNumberSpanishLicensePlate_Yolov8_MaxFilters.py

Run the program.

GetNumberSpanishLicensePlate_Yolov8_MaxFilters.py

The car license plates and successes or failures through the different filters appear on the screen.

The LicenseResults.txt file lists the car license plates with their corresponding recognized ones.

In a test with 21 images, 17 hits are achieved, the same that achieved with car licenses detected with yolo
in and manual labeling in the LicensePlate_Labeled_MaxFilters project https://github.com/ablanco1950/LicensePlate_Labeled_MaxFilters and
better than wit wpod_net https://github.com/ablanco1950/LicensePlate_Wpod-net_MaxFilters .

Also is attached GetNumberInternationalLicensePlate_Yolov8_MaxFiltersA.py, that tries to detect licenses plates in any format, but the hit rate is descending to 53 hits in 117 images ( 70 from 117 in https://github.com/ablanco1950/LicensePlate_Labeled_MaxFilters), because of not filtering erronous detection in base of format of license)

References:

https://learnopencv.com/ultralytics-yolov8/#How-to-Use-YOLOv8?

https://public.roboflow.com/object-detection/license-plates-us-eu/3

https://docs.ultralytics.com/python/

https://medium.com/@chanon.krittapholchai/build-object-detection-gui-with-yolov8-and-pysimplegui-76d5f5464d6c

https://medium.com/@alimustoofaa/how-to-load-model-yolov8-onnx-cv2-dnn-3e176cde16e6

https://gist.github.com/endolith/334196bac1cac45a4893#

https://stackoverflow.com/questions/46084476/radon-transformation-in-python

https://gist.github.com/endolith/255291#file-parabolic-py

https://learnopencv.com/otsu-thresholding-with-opencv/ 

https://towardsdatascience.com/image-enhancement-techniques-using-opencv-and-python-9191d5c30d45

https://blog.katastros.com/a?ID=01800-4bf623a1-3917-4d54-9b6a-775331ebaf05

https://programmerclick.com/article/89421544914/

https://anishgupta1005.medium.com/building-an-optical-character-recognizer-in-python-bbd09edfe438

https://datasmarts.net/es/como-usar-el-detector-de-puntos-clave-mser-en-opencv/https://felipemeganha.medium.com/detecting-handwriting-regions-with-opencv-and-python-ff0b1050aa4e
     
