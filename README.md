LicensePlate_Yolov8_MaxFilters: recognition of car license plates that are detected by Yolov8 and recognized with pytesseract after processing with a pipeline of filters choosing the most repeated car license plate.
In a test with 21 images, 16 hits are achieved

Requirements:

yolo must be installed, to do so follow the instructions indicated in:
  https://learnopencv.com/ultralytics-yolov8/#How-to-Use-YOLOv8?

In this case, the option to install on the computer itself was chosen.

pip install ultralytics

Functioning:


Download the project to a folder on disk.

Download to that folder the roboflow files that will be used for training:

https://public.roboflow.com/object-detection/license-plates-us-eu/3

In that folder you should find the train and valid folders necessary to build the model

Unzip the file with the test images test6Training.zip, taking into account when unzipping you can create a folder

test6Training inside the test6Training folder, there should be only one test6Training folder, otherwise you will not find the
test images

Model Train:

run the program

LicensePlateYolov8Train.py

which only has 4 lines, but the line numbered 11 should indicate the full path where the license_data.yaml file is located.

Running from a simple laptop, the 100 epochs of the program will take a long time, but you can always pull the lid off the laptop and
continue the next day.

As a result, inside the download folder, the directory runs\detect\trainN\weights( where in trainN, N indicates
  the last train directory created) in which the best.pt file is located, which is the base of the model and
  which is referenced in line 15 of the GetNumberSpanishLicensePlate_Yolov8_MaxFilters.py program (modify the route so that it points to the last
train and best.pt created

Run the program.

GetNumberSpanishLicensePlate_Yolov8_MaxFilters.py

The car license plates and successes or failures through the different filters appear on the screen.

The LicenseResults.txt file lists the car license plates with their corresponding recognized ones.

In a test with 21 images, 16 hits are achieved, practically those achieved with car licenses detected with yolo
in and manual labeling in the LicensePlate_Labeled_MaxFilters project.

References:

https://learnopencv.com/ultralytics-yolov8/#How-to-Use-YOLOv8?

https://public.roboflow.com/object-detection/license-plates-us-eu/3

https://medium.chom/@chanon.krittapholchai/build-object-detection-gui-with-yolov8-and-pysimplegui-76d5f5464d6c

https://medium.com/@alimustoofaa/how-to-load-model-yolov8-onnx-cv2-dnn-3e176cde16e6
