# machine-learning-course-UST
---

Machine Learning – UST Course  This repository contains my solutions to tasks from the Machine Learning course at UST (AGH University of Science and Technology). All tasks are based on assignments provided in PDF format by the course instructor and are implemented in Python using **Jupyter Notebooks**.

---

## 📚 Course & Resources

- Main reference: *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (O'Reilly)
- University course: Machine Learning – Computer Science & Intelligent Systems program (UST, AGH)
- Format: One notebook per assignment, matching the structure of the given task

---
## 🛠️ Technologies Used

- **Python**: 3.11.0rc1  
- **scikit-learn**: 1.5.0  
- **TensorFlow**: 2.14.0  
- **NumPy**: 1.26.0  
- **Pandas**: 2.2.2  
- **Matplotlib**: 3.9.0  
- **Seaborn**: 0.13.2  
- **Jupyter Notebook**

---
## 🧪 Installation

To set up and run this project locally, follow the steps below:

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/machine-learning-UST-course.git
   cd machine-learning-UST-course
   ```
 1.a (Optimal) Create a virtual environment
    ```bash
     python -m venv venv
     source venv/bin/activate
      ```
2. Install required packages
   ```bash
   pip install -r requirements.txt
   ```
3. Start Jupyter Notebook
   ```bash
   jupyter notebook
   ```

--- 
## 📌 Topics Covered
This README will be updated regularly with new tasks and the methods used to solve them.

---
# Lab01 – Regression Models Evaluation

This assignment focuses on the implementation and evaluation of multiple regression models using Python and scikit-learn. The main objective is to compare different regression approaches on a synthetically generated dataset, highlighting their strengths and weaknesses—particularly regarding model complexity, overfitting, and generalization ability.

---

## Implemented Models

- **Linear Regression**  
- **K-Nearest Neighbors Regression**  
- **Polynomial Regression**

---

## Project Structure

All models are encapsulated in a reusable class called `Regressions`, which provides the following key functionalities:

- **Data Handling**  
  Loads the dataset (`dane_do_regresji.csv`) and splits it into training and test sets using `train_test_split`.

- **Model Training**  
  Provides methods to initialize, train, and evaluate models:
  - `linear_model()`
  - `knn_model()`
  - `poly_model()`

- **Performance Evaluation**  
  Computes Mean Squared Error (MSE) for both training and test sets to evaluate model performance and detect potential overfitting.

- **Model Management**  
  Stores trained models and their configurations in a list, allowing for easy access, export, or reuse.

---

##  Technologies Used

- Python 3.11  
- scikit-learn – regression algorithms and evaluation metrics  
- NumPy & Pandas – data generation and manipulation  
- Matplotlib – data visualization  
- Jupyter Notebook – experimentation and presentation  

---

## ▶ How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt