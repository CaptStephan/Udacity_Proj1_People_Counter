# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## Explaining Custom Layers

According to the OpenVino Toolkit Custom Layers Guide, "Custom layers are layers that are not included in the list of known layers." This means that any layer in your model that is not already defined inside OpenVino would be defined as a custom layer and the user would need to add definitions for these layers by adding extensions to both the model optimizer and the inference engine.

The process behind converting custom layers involves...
For simplicity, I will list the options available for TensorFlow since I am using a TensorFlow model for my project. The general process behind converting custom layers has three main options:
1) Register the layers as extensions to the model optimizer.
2) Replace the subgraph.
3) Offload to TensorFlow during inference.

The first option consists of the following steps:
a) Generate the template extension files using the Model Extension Generator.
b) Edit the Extractor Extension template file to define the layer parameters.
c) Edit the Operation Extension template file to define the shape / dimensions of the layer.
d) Generate the model files with the model optimizer.
e) Edit the CPU / GPU extension files to define the function.
f) Compile the Extensions Library and execute the model.

Some of the potential reasons for handling custom layers are...
Reasons for implementing a custom layer might be to test a new topology or test different versions of a topology for overall performance and accuracy.

## Comparing Model Performance

I first did a quick Google search for possible pre-trained models and came up with ALEXNET that won the Imagenet Large Scale Visual Recognition Challenge (ILSVRC) in 2012. This seemed a little lofty, so I did more searching and came across the TensorFlow Model Zoo. Available models in the Model Zoo provided options for accuracy and speed on pre-trained TensorFlow models. I chose the "Faster RCN Inception V2" model based on high relative performace and above average accuracy. Only models that output boxes were considered.

Relative performace was evaluated using the table here:
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

The model files were downloaded from:
http://download.tensorflow.org/models/object_detection/rfcn_resnet101_coco_2018_01_28.tar.gz

I am working from a 2014 gen MacBook Pro with an Intel Movidius Neural Compute Stick (first gen), so the next step
was to install OpenVino Toolkit, full verion with all options. I also installed TensorFlow in case I needed to
offload any layer operations to TensorFlow.

I then went to the OpenVino docs to follow the steps for converting the model from TensorFlow to OpenVino, docs
can be found here:
https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html

My method(s) to compare models before and after conversion to Intermediate Representations
were made using runs in the Classroom Workspace and comparisons provided in the documentation for the model.

The model inference time pre-conversion according to the documentation (based on timings performed using an Nvidia GeForce GTX TITAN X card) was 92mS. This is a very good performance and would likely be far better than the Classroom Workspace.  Since my Python skills are a bit limited, I struggled to complete this in a timely manner and was not able to compare the two models myself.  Using numbers provided by a classmate I was able to see that a similar model had similar accuracy before and after conversion.  It also had an improvement in speed of approximately 98.8%!! This is an incredible boost for a small device and would be a great advantage.

The size of the model pre- and post-conversion was 57MB and 250KB respectively.  The reduction was significant and would save disk storage space on a smaller device like the Raspberry Pi.

## Assess Model Use Cases

Some of the potential use cases of the people counter app are detecting the maximum number of people allowed into a room at one time or counting the number of people entering a transit station to predict the number of train cars or busses required to be active at any one time.

Each of these use cases would be useful because it would improve public safety and/or transportation efficiency without having a human stuck doing a repetitive and boring task, which could lead to errors.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are similar in that the all impact the model accuracy in the end.  Having poor lighting can make the detection less accurate.  Having a poorly trained model will result in lower accuracy. Having a camera with a too narrow focal length will also reduce the visibility of a person in the frame and result in lower accuracy.
