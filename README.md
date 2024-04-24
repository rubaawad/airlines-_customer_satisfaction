
Sure, here's a README file for your airlines_customer_satisfaction GitHub project:

# Airlines Customer satisfaction 
    airlines_customer_satisfaction is a machine learning project that aims to predict customer satisfaction based on various  features. It utilizes different machine learning algorithms to build predictive models and evaluates their performance using various metrics.


# Installation
    To run the project locally, follow these steps:

    # Clone the repository:
        git clone https://github.com/rubaawad/airlines_customer_satisfaction.git
    # Navigate to the project directory:
        cd airlines_customer_satisfaction
    # Install the required dependencies:
        pip install -r requirements.txt
# Notebooks
    In src/components/notebook/data you can find these files:

    # EDA Notebook
        CUSTOMER SATISFACTION PERFORMANCE.ipynb: Exploratory Data Analysis notebook for analyzing the dataset.
    # Model Training Notebook
        MODEL TRAINING.ipynb: Notebook for training machine learning models.
# Data
    The project uses the airlines_customer_satisfaction Dataset from Kaggle, which contains various  features such as
    demographic information, flight details
    service ratings
    travel preferences
    and other relevant factors. 
    These features are essential in understanding the factors that influence customer satisfaction. The dataset is preprocessed before training the models.

# Models
    The project employs several machine learning models, including:
    K-Nearest Neighbors
    XGBoost
    Decision Tree
    Random Forest

# Evaluation
    The performance of each model is evaluated using the following metrics:
    Accuracy
    Precision
    Recall/Sensitivity
    Specificity
    F1-score
    ROC AUC
# Web Application
    The project includes a web application for predicting customer satisfaction using the trained machine learning models. To run the web application:

    Navigate to the templates directory.
    Install Flask if not already installed: pip install Flask.
    Run the Flask application: python application.py.
    Access the web application in your browser at http://localhost:5000.
# Docker
    Usage Instructions:
        Pull the Docker Image:
            docker pull rubamahgoob/customer_satisfaction_app:latest

    Run the Docker Container:
        docker run -d -p 5000:5000 rubamahgoob/customer_satisfaction_app:latest

    Access the Application:
        Once the container is running, you can access the Flask application by navigating to http://localhost:5000 in your web browser.

    Input Health Parameters:
        On the home page of the application, you'll find a form where you can input various health parameters such as pregnancies, glucose levels, blood pressure, etc.

    Get Prediction Results:
        After entering the health parameters and clicking on the "Predict" button, the application will provide a prediction result indicating whether the customer is satisfaied or not.

    Note:	
        Make sure Docker is installed and running on your system before pulling and running the Docker image.
        Ensure that port 5000 is not being used by any other application on your system, as it is used by the Flask application to serve the web interface.
        
    You can customize the Docker container's port mapping if port 5000 is already in use on your system. For example, you can map it to a different port using the -p flag in the docker run command.



# Contact
    For any questions or feedback, please contact:

    Ruba Awad
    Email: rubaabdalla44@gmail.com
