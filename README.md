# ECE-3432-Final_DeepLearning
 
 
 ## Summary
 
 This is modified code to train and test deep learning models. The key differences in my code to the original repo code is listed below. 

 This repo was created to document and show off the final project for ECE-3432 (Robotic Control using RASPI). The goal of the project was to achieve a mean squared loss (MSELoss) of below 0.7 while also improving the overall accuracy of the model within a margin of 2. WThe purpose of this is to generate a model that can accurately predict the optimal servo motor value to send to keep an RC car on a path. The end results were that we achieved a loss of 0.1359 and an accuracy of 94.28%.
 
 Shown below is the layer features that were utilized in this project. We wanted to reduce the overall amount of layers to improve the speed of predicitons once the model is loaded onto the raspberry pi.

![Image](https://github.com/DeanCavaliere/ECE-3432-Final_DeepLearning/blob/master/Results/Training_1/ModelLayers.PNG)

Next, we have the test information which consists of some runtimes and the overall value of loss.

![Testing](https://github.com/DeanCavaliere/ECE-3432-Final_DeepLearning/blob/master/Results/Training_6/ModelTestInfo.PNG)


Lastly, we have the accuracy test which runs our model to generate an predicted output that is then cross referenced to the actual output. For this particular test, an error margin of 2 is used.

![Accuracy](https://github.com/DeanCavaliere/ECE-3432-Final_DeepLearning/blob/master/Results/Training_6/AccuracyTest.PNG)

 For this project, we did rerun the training software multiple times to create 'generational' models that built off of each other.
 During this process, nothing changed in terms of code, we just continued to train our model with the same layers and settings.
 In total, we had 6 trained models; by the 6th model, the improvements made to loss and accuracy seemed too small to continue 
 testing. Further improvements can probably be made by changing the batch size and layering for better loss and accuracy.
 
 
 ## Installation and Running on the RASPI
 
 To install, the project requires some dependencies which are all listed in the 'requirements.txt'.The most important
 files that need to be installed to the raspberry pi is that torch and torchvision packages which have been provided in 
 the 'PyTorch Files' folder. The commands to run are also follows:
 > sudo -H pip3 install torch-1.6.0a0+521910e-cp36-cp36m-linux_armv7l.whl
 > sudo -H pip3 install torchvision-0.7.0a0+fed843d-cp36-cp36m-linux_armv71.whl
 
 To clone this repo to the raspberry pi, run the following:
 > git clone https://github.com/DeanCavaliere/ECE-3432-Final_DeepLearning.git

 Shown below is a screenshot of the raspberry pi installation. Everything mentioned above is highlighted in yellow.
 
 ![Install](https://github.com/DeanCavaliere/ECE-3432-Final_DeepLearning/blob/master/Results/Installation/InstallTorch_gitClone.PNG)
 
 To run the prediction script, you must navigate into the project and run the file with python 3, this is shown below.
 > cd ECE-3432-Final_DeepLearning 
 > python3 racecarPredict.py

 The python script is pre-set to run the prediction function for the following randomly chosen image 
 '/03_12_2020_0/output_0002/i0000990_s15_m17.jpg' which is shown below:
 
 ![predictimage](https://github.com/DeanCavaliere/ECE-3432-Final_DeepLearning/blob/master/data/images/03_12_2020_0/output_0002/i0000990_s15_m17.jpg)
 
 You can change the predicted image by changing the 'IMAGE_FILE' parameter in the racecarPredict.py file.
 
 
 ## Results
 
 As shown above, we have chosen to run an arbitrary image through the prediction function. The predicted value comes out
 to be 16. If we observe the next image (i.e. image '...i0000991_s16_m17.jpg'), we see the next servo value is 16.
 Shown below is the image we used to predict the value of 16.
 
 ![predicted](https://github.com/DeanCavaliere/ECE-3432-Final_DeepLearning/blob/master/Results/Predicted_Value_RPI.PNG)
 
 We can also see that the total time to run the prediction is 0.170 seconds although this number seems to fluctuate as
 a value as low as 0.135 seconds has been achieved. In the 'Results' folder is some additional images of raspberry pi screenshots
 and screenshots of multiple models after being iteratively trained. 
 

 ## What has changed in this repo compared to the original?
 
 - Added new JSON features 
 >  timeEnable = when enabled, a CSV of time per batch is saved for training and testing as a CSV format. Examples of these CSV files are shown in the "TimeData" folder.
 
 >  cudaID = select what cuda device is being used in-case multiple GPU's are present
 
 - Added a cuda search file 'testForCuda.py'
 >  automatically searches for cuda devices and displays information about the device!

 - Added the 'Loop' folder
 >  'Loop' contains preconfigured files to loop model training! All iterative models are saved and corresponding data is documented in a CSV file.
 
 
 ## Additional Testing:
 
Enclosed in the 'TestTimeData' and 'TrainTimeData' are data sets that compare the runtime of a fairly new AMD CPU versus an 'older' 750ti GPU versus a 'newer' 1070 GPU. The time per batch and runtime between both GPU's are fairly close; the CPU definetly struggles and is significantly slower than the GPU's (no surpirse here).

 
 ## Future Implementations? 
 
 Multi-GPU support
