## About the Project
The target of the project is to build a image classification model and use django framework to run inference from a webpage. The approaches which are being embraced are:
<ol>
  <li>Building a deeplearning classification model from freely available datset. In this case <a href="https://www.kaggle.com/alessiocorrado99/animals10">This dataset is being used</a>.</li>
  <li>Set up a Django App.</li>
  <li>Accept input from a web page..</li>
  <li>Convert every form of input into the exact format that the machine learning model needs as input.</li>
  <li>Make predictions with the trained deep learning-based model.</li>
  <li>Show the predicted labels and the image in website.</li>
</ol>

<div>
  <h2>Building a VGG16 Model</h2>
  <p>For training this model I am using pytorch library. It provides torchvision module which will be used to download the pretrained model on Imagenet. I will describe the main steps in subsequents points.<a href="https://github.com/KaziShawon/imageclassification-docker-fastapi/blob/main/vgg16_vision_multiclass.ipynb"> Find the notebook here</a></p>
  <ol>
    <li>The pretrained model is downloaded, as I am having 10 classes finetuning is done to classify the desired 10 classes, it can be seen that at the very last layer it has 10 FC layers. The classes are dog,horse,elephant,butterfly,gallina,chicken,cat,cow,sheep,spider,squirrel. <br> <img src="https://i.ibb.co/xqBgqY2/getmodel.jpg" alt="getmodel" border="0"></li>
    <li>Data augmentation like: RandomRotation, RandomResizedCrop, RandomHorizontalFlip is being used to create data loader. The images are resized into 224,224 as height and width. For training set 80% of the images being used. And for training and validation set 10% per each is being used.</li>
    <li>Loss function CrossEntropyLoss is used. Adam optimizer is used to update learning parameters of each neural network parameter during during. CosineAnnealingLR is used to set learning rate accoring to the validation loss, if the loss increases the model assigns bigger step to gradient descent, for lower loss vice versa.<br> <img src="https://i.ibb.co/d2j1R7S/training.jpg" alt="training" border="0"></li>
    <li>For gradient updates in each iteration of training opt.zero_grad(), loss.backward(), opt.step() is being used. We are training our model for five epochs.</li>
    <li>Few things are important in pytorch. When the model is traing, the datset and model should be sent to the device (cpu/gpu). And before training model should be set to train model with model.train(). For evaluation and test model.eval(). When the evaluation and test phase happens it should be run with torch.no_grad(), as it obstructs the model to update the gradients at that time.</li>
    <li>Softmax function is used for probability of multiclass classification.</li>
    <li>The model has achieved 96% accuracy with test set, which dataset model has not seen during training.</li>
    <li>The best model according to the best validation accuracy is being saved in directory.</li>
  </ol>
</div>
<div>
  <h2>Building Django APP</h2>
  <h4>Django URL setup</h4> 
       <p>Having the PyTorch classification logic implemented in image_classification/views.py, I now need to integrate it into the Django app and really use it in a Django view and template. For that, I first make some adjustments in the URLs by creating a separate image_classification/urls.py for the URLs of the image classification app. When visiting the main page of the web app, the requests are now directed to an index view, which I need to implement next and which will make use of the previous PyTorch classification logic. Before, I still need link these URLs to the project's URLs in pytorch_django/urls.py such that they become effective.</p>
  <h5>Django Image Upload, Classification, and Display</h5>
  <p>Now, I implement the index view, which accepts an uploaded image, processes it, and passes it to the PyTorch classification logic implemented above. I also need a simple Django template to render the web interface, where users can upload an image and submit it for classification. After classification, the template needs to show the predicted label.</p>
  <ol>
    <li>For submitting uploaded images, I use a very simply Django form with an ImageField in image_classification/forms.py</li>
    <li>I use this form inside my index view to accept uploaded images. (index is how I called it in my image_classification/urls.py but it could be any other name.) Here, I just want to display the uploaded image and pass it to the PyTorch model for classification. I do not want to (even temporarily) store it to the file system/disk. Hence, inside the view (image_classification/views.py), I get the image from the form, get its byte representation (for PyTorch) and create an image URI for displaying the image in the template later.</li>
    <li>The index view above calls Django's render function on a template image_classification/index.html, which I need to create now (inside the image_classification/templates directory). The template needs to show the form for uploading images and, after submitting and image, the uploaded image and its predicted label.</li>
    <li>The uploaded image uses the saved and passed image URI from before and does not save or load any image from disk, which is important for privacy.</li>
  </ol>
</div>
<div>
  <h3>Testing the App Locally</h3>
  <p>Running the app locally should now work without errors and show a simple page with the image upload form: <br><img src="https://i.ibb.co/56XGpnB/django-elephant.jpg" alt="django-elephant" border="0"></p>
</div>
