# Vehicle Color Recognition

This project focuses on vehicle color recognition using deep learning techniques.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Installation

To get started with this project, follow these steps:

1. Install the required Python packages, including "ultralytics," by running the following command:

   ```bash
   !pip install ultralytics
Make sure you have Python and pip installed on your system.

2. Clone the GitHub repository for this project:

   ```bash
   !git clone https://github.com/ultralytics/ultralytics.git

This will download the project's source code to your local machine.

Install other dependencies mentioned in the code (e.g., TensorFlow, scikit-learn) using pip if you haven't already.

## Usage
This project consists of several Python scripts for vehicle color recognition, using YOLO for object detection and a custom classification model. Here's how you can use these scripts:

The main YOLO object detection and training script is located in the project directory.

    from ultralytics import YOLO
    model=YOLO('yolov8m-seg.pt')
    model.train(
        project="VCOR",
        name="yolov8m-seg",
        deterministic=True,
        seed=43,
        data="coco128-seg.yaml", 
        save=True,
        save_period=5,
        pretrained=True,
        imgsz=224,
        epochs=8,
        batch=4,
        workers=8,
        val=True,
        lr0=0.018,
        patience=10,
        optimizer="SGD",
        momentum=0.947,
        weight_decay=0.0005,
        close_mosaic=3,
    )
The classification model is implemented using TensorFlow/Keras and is also present in the code.

    model_name='EfficientNetB3'
    base_model=tf.keras.applications.EfficientNetB3(include_top=False, weights='imagenet',input_shape=(isize, isize, 3), pooling='max')
    x=base_model.output
    x=keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 )(x)
    x = Dense(256, activation='relu')(x)
    x=Dropout(rate=.45, seed=123)(x)        
    output=Dense(len(class_subset)-3, activation='softmax')(x)
    model1=Model(inputs=base_model.input, outputs=output)
    model1.compile(Adamax(learning_rate=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
##  License
This project is licensed under the MIT License
