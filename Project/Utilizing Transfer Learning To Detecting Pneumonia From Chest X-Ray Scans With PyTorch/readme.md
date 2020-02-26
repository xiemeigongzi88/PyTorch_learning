# Utilizing Transfer Learning To Detecting Pneumonia From Chest X-Ray Scans With PyTorch
    There are undoubtedly features in images we feed into these models that they look at to make predictions and that is what we seek to explore in this article. Not long ago, researchers at Stanford university released a paper https://arxiv.org/abs/1901.07031 on how they are using deep learning to push the edge of Pneumonia diagnosis. Their work really got me fascinated so I tried it out in Pytorch and I am going to show you how I implemented this work using a different dataset on Kaggle.

## Dataset: 
    The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal). There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal).
Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients’ routine clinical care.
    For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert.

## Data source: 
    https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
    https://data.mendeley.com/datasets/rscbjbr9sj/2

## Link to paper on Class Activation Maps: 
    http://cnnlocalization.csail.mit.edu/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf

## Tool: 
    Torch (torch.nn, torch.optim, torchvision, torchvision.transforms)
    Numpy
    Matplotlib
    Scipy
    PIL

## Steps:
    (1). Load and preprocess our data
    (2). Feed the data into the model and train
    (3). Test and evaluate our model

## Conclusion: 
### MODEL ARCHITECTURE：
    Our base line model for this project is the ResNet 101. ResNet models like other convolutional network architectures consist of series of convolutional layers but designed in a way to favor very deep networks. The convolutional layers are arranged in series of Residual blocks. The significance of these Residual blocks is to prevent the problem of vanishing gradients which are very pervasive in very deep convolutional networks. Residual blocks have skip connections which allow gradient flow in very deep networks.

### BUILDING AND TRAINING OUR MODEL FOR CLASSIFICATION.
    Pytorch provides us with incredibly powerful libraries to load and preprocess our data without writing any boilerplate code. We will use the Dataset module and the ImageFolder module to load our data from the directory containing the images and apply some data augmentation to generate different variants of the images. After defining our model class and inheriting from the nn.Module, we define the graph of our model in the init constructor by leveraging the feature extractor of ResNet-101 through a technique called transfer learning.

    The best  training accuracy is 94%, and the best test accuracy is 90%. 

    I just got the details from the hidden insights of data. And this project was inspired by the project “Detecting and Localizing Pneumonia from Chest X-Ray Scans with PyTorch” (https://blog.paperspace.com/detecting-and-localizing-pneumonia-from-chest-x-ray-scans-with-pytorch/)

