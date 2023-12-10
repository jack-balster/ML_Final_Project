# ML_Project

In this project, we attempt to perform sentiment analysis on IMDb movie reviews. 

To accomplish this natural language processing task, we utilized Hugging Face's DistilBERT tokenizer and model to tokenize and encode the reviews to a numerical format. Additionally, to ensure that the length of the reviews does not impact the lengths of its numerical output, we attached an average pooling layer to the model to standardize the vector lengths. From here, we chose four classification models to compare: KNN, SVM, decision tree, and perceptron. To evaluate and compare results, we used F1 as a metric of success. By bootstrapping the validation results 10 times (sample size 1000), we averaged the F1 score among the bootstrapped samples for each model. The chosen model is the model with the highest average F1 score.

## Architecture
- Environment: Google Colab which provides a Jupyter Notebook interface.
- Data Storage: Google Drive for dataset storage and access.
- Libraries: pandas for data manipulation, skikit-learn for machine learning models and evaluation, ast for data type conversion, google colab for saving files to a local machine, and numpy for numerical operations.
  
## Design Principles
- Modularity: Code is organized into different sections/cells for data loading, preprocessing, model training, tuning, and evaluation.
- Scalability: The use of Google Colab allows for scalability and access to higher computational resources.
- Reproducibility: Fixed random states in data splits and model training ensure reproducible results.

## Code Structure
The structure of the code is simple as each step/section of the process is separated by cells due to the original code being made in Google Colab. In total, there are 12 cells of code which run as follows:

1. Connect to Google Drive
   Asks the user for permission to connect to the user's Google Drive.
   
2. Load Dataset
   Loads the IMDb dataset which the user is expected to have already downloaded and uploaded to their Google Drive. It will use the path        given to identify the location of the data file

3. Install Transformers Package
   Downloads the necessary transformer packages to perform the preprocessing tasks needed for the data.

4. Slice Dataset
   Randomly selects 10,000 positive samples and 10,000 negative samples from the original dataset, concatenates these two subsets together to    one dataset, then shuffles the rows to ensure ordering does not affect training. The reason for lowering the size of the dataset is due to       our restricted computational capacity. Google Colab provides up to 12 GB of RAM, which can quickly dissipate when working in NLP.

5. Preprocess Data (Tokenize, distilBERT Transformer, Pooling)
   Performs much of the brunt work of the program: data preprocessing for the classifiers. It creates an instance of the pre-trained uncased     distilBERT tokenizer and model ("distilbert-base-uncased") after which, the function "encode_and_pool" is defined.

   "encode_and_pool" function performs three main steps for preprocessing the data: tokenizing, distilBERT transforming, and pooling.

  - Tokenizer: the review is converted to numerical representations called tokens. The number of tokens produced depends on the length of        the review. This serves as preparation for the distilBERT model. A graphical illustration to demonstrate the input and output               relationship of the tokenizer can be viewed below.
   
<img width="1406" alt="Screenshot 2023-12-05 at 6 35 49 PM" src="https://github.com/Alyssa1918/ML_Final_Project/assets/123338206/d13a86cf-886e-43dd-8b13-2d56a6db023f">

  - DistilBERT model: tokens are then further converted to vectors/tensors that capture the relationship between words. The number of         vectors depends on the number of tokens a review is tokenized to, but the length of each vector will remain constant at 768. The result      from a single review at this point will essentially become a list of lists where the length of the outer list varies with the length of      the review, and the length of the inner list is 768. Another graphical illustration to demonstrate the input and output relationship of      the distilBERT model can be viewed below.

<img width="1411" alt="Screenshot 2023-12-05 at 6 36 19 PM" src="https://github.com/Alyssa1918/ML_Final_Project/assets/123338206/8b427340-042d-4887-8777-a660c622e620">

  - Pooling: takes the average of all vectors from distilBERT with respect to their index (ie. column-wise) for a single review. The number     of vectors from distilBERT depends on the number of tokens a single review produces. This means that each review will reside in varying      numbers of dimensions, which is not suitable for our classification algorithms. To mitigate this, average pooling will compress the          varying number of vectors to a single vector by finding the means of their respective indices. Recall that the length of each vector is      768, so the resulting vector will also be of length 768 where the elements are the averages for its respective index. The illustration       below demonstrates this action.

<img width="1203" alt="Screenshot 2023-12-05 at 6 36 44 PM" src="https://github.com/Alyssa1918/ML_Final_Project/assets/123338206/b17bc9d5-2110-479e-b3bd-4b60af608327">

  At the end, "encode_and_pool" will convert the final vector into a numpy array for ease of use in subsequent cells.

  After the "encode_and_pool" function, the code cell will then process the data by batches of size 100. After all 20,000 samples have been    processed, the results are then added as a new column to the data frame titled "pooled_output."

  Lastly, the newly updated data frame is saved to csv file called "pooled.csv" on the user's computer. Doing so avoids the entire 7.5-hour     data preprocessing step for future work as Google Colab will disconnect after some time of inactivity leading to a restart of the program.

6. Load Pooled Results and Split for Train/Test
   Loads the "pooled.csv" file which contains the preprocessed results from the previous cell. Converts the "pooled_results" column from        strings to numpy arrays using the "string_to_nums" function. Lastly, randomly splits the dataset into an 80/20 train-test split.

7. Hypertuning K Value for KNN
   Searches for the optimal K value for KNN using 3-fold cross-validation. By using GridSearchCV, we can perform cross-validation on an         array of possible K values set from 1 to 300 (odd numbers only) to identify the best K value, which was found to be 17.

8-11. KNN Classifier, SVM Classifier, Decision Tree Classifier, Perceptron Classifier
   Trains a KNN, SVM, decision tree, and perceptron model on the training data set. Gets each model's respective prediction for the test set.

12. F1 Evaluation via Bootstrapping
    Bootstraps 1000 indices of the validation results and takes their corresponding predictions from each model and their ground truths.        Computes the F1 score for each model within the bootstrapped subset. Repeats this process 10 times for a total of 10 bootstrapped            validation results. Averages the F1 score for each model across the 10 bootstrapped samples. Finally, prints out the final results.

## Project Deliverables

This project includes a project write-up, presentation slide deck, presentation video, project code, documentation, and a code demo. 

The project write-up is located here: https://docs.google.com/document/d/1wf9QR98MRQJQ4KMXouotlnr4B9FB4yPwaCo33jGfEQc/edit?usp=sharing. 

The presentation slide deck is located here: https://docs.google.com/presentation/d/13wMaBIabJM40SI1MSv151c9MWjgJXOGfyOF2IJuFFmY/edit?usp=sharing. 

The presentation video is located here: https://drive.google.com/file/d/1d1k6Sss569rh0IqrrS7alvyGgmaC68Ia/view?usp=sharing. 

The project code is located above in the CPTS_437_Final_Project.ipynb and at this link: https://colab.research.google.com/drive/1D7EUnZtDOcqwN8pMxG3NiV_pccAtCmF3?usp=sharing. 

The documentation is this README.md. 

Lastly, the code demo is located above in the CPTS_437_Final_Project_Demo.ipynb and here at this link: https://colab.research.google.com/drive/1UWUbXuvNvZP7HAU71wI_AhzCJyAJQuE-?usp=sharing.
