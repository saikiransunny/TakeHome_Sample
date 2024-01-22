## README

### **Overview**

This readme provides an overview of the current state of the project, identifies issues in the sample code, and outlines next steps for improvement and development.

### **Issues in Sample Code provided**

**Visual Representation:** Consider using visuals (graphs, charts) instead of direct number comparison. Humans perceive scale better in graphs, enhancing the understanding of the data.

**Feature Engineering:** Implement more feature engineering techniques. The current code lacks extensive feature usage and encodings, which can significantly impact model performance.

**Metrics for Imbalanced Dataset:** Ensure appropriate metrics are used for datasets with class imbalance issues. The current metrics may not accurately reflect model performance in such scenarios.

**Code Documentation:** The code lacks comprehensive documentation. Include comments and explanations to enhance readability and maintainability.


### **NEXT STEPS**

**Model Development:**

- **Class Refactoring:** Create a single class to encapsulate all model methods for better organization and maintainability.

- **Explore Different Models:** Experiment with other algorithms such as Gradient Boosting, Neural Networks, etc., to improve model performance.

- **Feature Engineering:** Implement target encoding for additional feature engineering.

- **Additional Data:** Incorporate additional data, such as crime rate, property taxes, proximity to transport, and parking availability.

- **Handling High Cardinality Categorical Features:** For high cardinal categorical features, consider using hash encoding or high-dimensional vectors (e.g., Word2Vec) for better model performance.


**Engineering/MLOps:**

- **Containerization:** Containerize the application for portability and scalability.

- **CI/CD Pipeline:** Implement a CI/CD pipeline using GitHub or Gitlab for automated testing and deployment.

- **Deployment Options:** Depending on use case, explore deployment options such as ECS, Fargate, or Sagemaker.

- **Retraining and Model Monitoring:** Create additional containers for retraining the model. Consider incorporating a model monitoring solution (e.g., Evidently AI) for continuous evaluation of model performance.


### **Getting Started**

**Directory Structure:**

    |-model_artifacts
    |  |-model.pickle
    |-requirements.txt
    |-README.md
    |-app.py
    |-Data
    |  |-MyBaselineModel.ipynb
    |  |-outputs
    |  |  |-results.csv
    |  |-inputs
    |  |  |-test.csv
    |  |  |-train.csv
    |-src
    |  |-helper_functions.py
    |  |-constants.py
    |  |-helper_classes.py
    |  |-Takehome_notebook.ipynb


**Exploring the API**

- Run ```pip install -r requirements.txt``` to install the libraries
- Run ```python app.py```
- Open your browser and visit http://127.0.0.1:8000/docs.
- Press the **Try it out** button.
- Enter your input parameters.
- Click on **Execute** to get the results.
- Alternatively you can also use curl command in the terminal. 
```
curl -X 'POST' \
    'http://127.0.0.1:8000/predict/' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "easements": 0,
    "lotarea": 0,
    "bldgarea": 0,
    "resarea": 0,
    "officearea": 0,
    "retailarea": 0,
    "garagearea": 0,
    "strgearea": 0,
    "factryarea": 0,
    "numbldgs": 0,
    "numfloors": 0,
    "unitstotal": 0,
    "lotfront": 0,
    "lotdepth": 0,
    "bldgfront": 0,
    "bldgdepth": 0,
    "assessland": 0,
    "yearbuilt": 0,
    "yearalter1": 0,
    "yearalter2": 0,
    "builtfar": 0,
    "tract2010": 0,
    "xcoord": 0,
    "ycoord": 0,
    "borough": "BK",
    "splitzone": "Y",
    "irrlotcode": "Y"
    }'
```
    
