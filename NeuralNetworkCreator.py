'''
Created by Bruno Ferreira

Creator's message:

    I don't understand much about programming Artificial Neural Networks, this whole project was created from many documents, tutorials, help, ...
    Basically a compilation of a lot of things that ended up working.

    Use however you like, it's just a base, build something on top of it or just enjoy it as it is now.

    All the way to use it is on Github or in the project folder.

'''

import os
import numpy as np
import cv2
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import sys
import cv2
import numpy as np
from keras.models import load_model
import sys





def Create():
    # Args
    # ---.py create [NeuralNetworkName] [Epochs] [MainDir]
    NeuralNetworkName = sys.argv[2].strip().lower().replace(" ","_")
    MainDir = sys.argv[4].strip()
    Epochs = sys.argv[3]

    # list of subdirectories (obj names)
    names_obj = sorted(os.listdir(MainDir))

    # Write Sample document
    f = open("sample_"+NeuralNetworkName+".txt", "w")
    for x in names_obj:
        f.write(str(x.strip())+"\n")
    f.close()


    # Initialize lists of images and labels
    images = []
    labels = []

    # Go through each subdirectory and read the images
    for i, obj in enumerate(names_obj):
        dir_obj = os.path.join(MainDir, obj)
        
        # Browse the images in the obj directory
        for image_name in os.listdir(dir_obj):
            # Read the image and resize it to 100x100 pixels
            image = cv2.imread(os.path.join(dir_obj, image_name))
            image = cv2.resize(image, (100, 100))
            
            # Add the image and label to matched lists
            images.append(image)
            labels.append(i)

    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    # Convert labels to one-hot encoding vectors
    labels = np_utils.to_categorical(labels)

    # Split the data into training and test sets
    training_Ind = int(len(images) * 0.8)
    training_images, test_images = images[:training_Ind], images[training_Ind:]
    training_labels, test_labels = labels[:training_Ind], labels[training_Ind:]

    # Normalize image pixels between 0 and 1
    training_images = training_images.astype('float32') / 255
    test_images = test_images.astype('float32') / 255

    # Create the neural network model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(100, 100, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(names_obj), activation='softmax'))

    # Compile the model with optimizer, loss function and evaluation metrics
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model with training data and evaluate with test data
    model.fit(training_images, training_labels, epochs=int(Epochs), batch_size=32, validation_data=(test_images, test_labels))

    # Save the trained model
    model.save('model_'+NeuralNetworkName+'.h5')



# Function to load and resize the image
def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (100, 100))
    image = np.array(image)
    image = image.astype('float32') / 255
    image = np.expand_dims(image, axis=0)
    return image

# Function to make predictions using the model
def predict(image_path,model,class_names):
    image = load_image(image_path)
    forecast = model.predict(image)[0]
    index_max = np.argmax(forecast)
    classObj = class_names[index_max]
    return classObj, forecast




def Run():
    #Args
    # ---.py run [sample] [model] [file]
    modelInp = sys.argv[3].strip()
    samples = sys.argv[2].strip()
    file = sys.argv[4].strip()



    # Load the trained model
    model = load_model(modelInp)

    f = open(samples,'r')
    lines = f.readlines()
    f.close()

    # List of class names
    class_names = sorted(lines)

    # Example of using the predict function
    classObj, forecast = predict(file,model,class_names)

    # Show the class name and a preview for each class
    for i, class_name in enumerate(class_names):
        if((forecast[i]*100)>0.01):
            print('{}: {:.2f}%'.format(class_name, forecast[i]*100))
    print('Identified Class: ', classObj)




if(len(sys.argv)>=5):
    if(sys.argv[1] == "create"):
        Create()
    elif(sys.argv[1] == "run"):
        Run()
    else:
        print("Help:")
        print(" .Create Neural Network --> create [NeuralNetworkName] [Epochs] [MainDir]")
        print(" .Run Neural Network--> run [sample] [model] [file]")
else:
    print("Help:")
    print(" .Create Neural Network --> create [NeuralNetworkName] [Epochs] [MainDir]")
    print(" .Run Neural Network--> run [sample] [model] [file]")
