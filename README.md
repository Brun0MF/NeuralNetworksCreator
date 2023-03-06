# NeuralNetworksCreator
 A simple neural network creator...
 
 Neural Network Creator is a simple program that allows you to create, train and run simple image comparison neural networks.
 
 ## Usage:
 
 ### Data Source
 To create a Neural Network, you need a data source. This source should be organized as follows:
 ```
 DatasetName
     -Class1
         -Class1Image_1.jpg
         -Class1Image_2.jpg
 
     -Class2
         -Class2Image_1.jpg
         -Class2Image_2.jpg
         
     -Class3
         -Class3Image_1.jpg
         -Class3Image_2.jpg
     
     ...
 ```
 
 The "DatasetName" is the name of the Image set. Within the main folder (DatasetName) there are subfolders, these must have the name of the class. Within each subfolder, there should be a set of images related to the class.
 
 An example dataset for an Animal Identification Neural Network:

  ```
 Animals
     -Cat
         -Cat_1.jpg
         -Cat_2.jpg
 
     -Dog
         -Dog_1.jpg
         -Dog_2.jpg
         
     -Parrot
         -Parrot_1.jpg
         -Parrot_2.jpg
     
     ...
 ```
 
 If you use the [@WMCDownloader](https://github.com/Brun0MF/WMCDownloader) Tool, it will automatically separate the class images into folders with the class name, saving you some work.
 
 Keep in mind that the more images a class has, the more accurate its recognition will be.
 
 
 **Class and image nomenclatures must not contain spaces or characters (excluding '_' or '-').**
 
 **All images must be in '.jpg' format**
 
 
  ### Create and Train Neural Network
 Let's create a new neural network,to do this just go to the console and write the following command (inside the project folder):
```
python NeuralNetworkCreator.py create [NeuralNetworkName] [Epochs] [MainDir]
```


 + **[Neural Network Name]** - Give your project a name (Same conditions as for class naming).
 + **[Epochs]** - Number of Training Phases (the higher the number, the more accurate the predictions will be, but the longer the training will be).
 + **[MainDir]** - The exact path to the main data source folder (DatasetName).

 
Now just wait for the whole process to finish.
Two files will be generated in the project folder:

 + **Sample** - Contains a list of class names.
 + **Model** - Contains the trained Neural Network model.
 
 
   ### Run the Neural Network
Let's run the neural network,to do this just go to the console and write the following command (inside the project folder):
```
python NeuralNetworkCreator.py run [Sample] [Model] [File]
```
 + **[Sample]** - The exact path to the Sample file (generated when creating the neural network)
 + **[Model]** - The exact path to the Model file (generated when creating the neural network)
 + **[File]** - The exact path to the image that will be analyzed by the Neural Network (in '.jpg' format).

 Now just wait for the results!
 
 
 ## Creator's message
 
   I don't understand much about programming Artificial Neural Networks, this whole project was created from many documents, tutorials, help, ...
   
   Basically a compilation of a lot of things that ended up working.
   
   Use however you like, it's just a base, build something on top of it or just enjoy it as it is now.
   
   All the way to use it is on Github or in the project folder.
   
    
 ## Extra
 
 ! - This repository contains some test datasets. They were extracted from Wikimedia Commons with the [@WMCDownloader](https://github.com/Brun0MF/WMCDownloader) Tool.
