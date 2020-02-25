#Object Detection

- Cane See smart glasses ai models
- This branch contain an object detection model based on YOLOv3(You Only Look Once) .

- By using the next two points you can generate your model :   
   * You can find the building model at URL :    
                 " https://raw.githubusercontent.com/experiencor/keras-yolo3/master/yolo3_one_file_to_detect_them_all.py "    
        here you can find the model and some function to load the weight to the model.
             
   * Here you can find the needed weights : 
                " https://pjreddie.com/media/files/yolov3.weights "

- Create and Save Model
    The first step is to download the pre-trained model weights.
    These were trained using the DarkNet code base on the MSCOCO dataset.
    Download the model weights and place them into your current working directory with the filename “yolov3.weights.”
   
   
- you can more documentation for create object detection model here : 
     " https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras/ "     
     
- After create and save the model as "model.h5"; you can use it with the file " objectDetection.py " in the repo in your pc 
    to make detection for the input images.
    But if you want, you can transform the model from h5 to tensorflow lite model and use it with  the file " objectDetection_lite.py ",
    and the requirments are :
    
                    tensorflow  2.0.0  
                    numpy       1.18.1   
                    cv2         4.2 
       
- Python: 3.6.4

- Now i can say to you have a fun !!!!!