# ECE-3432-Final_DeepLearning
 

 Modifed code to train and test deep learning code. 


 ## What has changed in this repo compared to the original?
 
 - Added new JSON features 
 >  timeEnable = when enabled, a CSV of time per batch is saved for training and testing as a CSV format. Examples of these CSV files are shown in the "TimeData" folder.
 
 >  cudaID = select what cuda device is being used in-case multiple GPU's are present
 
 - Added a cuda search file 'testForCuda.py'
 >  automatically searches for cuda devices and displays information about the device!
 
 
 ## Additional Testing:
 
Enclosed in the 'TestTimeData' and 'TrainTimeData' are data sets that compare the runtime of a fairly new AMD CPU versus an 'older' 750ti GPU versus a 'newer' 1070 GPU. The time per batch and runtime between both GPU's are fairly close; the CPU definetly struggles and is significantly slower than the GPU's (no surpirse here).

 
 ## Future Implimentations? 
 
 Multi-GPU support
