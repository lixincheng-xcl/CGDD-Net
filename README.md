# 🔥CGDD-Net

### The code will be released publicly before October, barring any unexpected issues.

### Introduction
This project is a retinal blood vessel segmentation code based on python and pytorch framework, including data preprocessing, model training and testing, visualization, etc. 

### Requirements  
The main package and version of the python environment are as follows
```
# Name                    Version         
python                    3.7.9                    
pytorch                   1.7.0         
torchvision               0.8.0         
cudatoolkit               10.2.89       
cudnn                     7.6.5           
matplotlib                3.3.2              
numpy                     1.19.2        
opencv                    3.4.2         
pandas                    1.1.3        
pillow                    8.0.1         
scikit-learn              0.23.2          
scipy                     1.5.2           
tensorboardX              2.1        
tqdm                      4.54.1             
```  
The above environment is successful when running the code of the project. In addition, it is well known that pytorch has very good compatibility (version>=1.0). Thus, __I suggest you try to use the existing pytorch environment firstly.__  
    
The current version has problems reading the `.tif` format image in the DRIVE dataset on Windows OS. __It is recommended that you use Linux for training and testing__
