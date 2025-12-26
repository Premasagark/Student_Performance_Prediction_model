



#AI-Based Student Performance Prediction System


Task Overview:
You are required to build a simple Machine Learning model that predicts student performance
based on academic and behavioural inputs. The trained model must be exposed using a FastAPI or a
REST API.


Dataset Description:

The dataset should include the following fields:

- Study Hours
- Attendance Percentage
- Internal Marks
- Practice Hours
- Performance (Target Label)
Target labels can be: Good, Average, Poor.

Using a Dataset student_data(1).csv which have 10000 data for train a model.

The dataset features realistic ranges for each performance category, generated from class-specific normal distributions with clipping to ensure valid bounds.

Good Performance Ranges:
•	Study Hours: 1.3–8.0 (mean 4.98).
•	Attendance %: 73.17–100.0 (mean 89.98).
•	Internal Marks: 55–100 (mean 79.95).
•	Practice Hours: 1.0–7.0 (mean 3.99).
•	Count: 3,334 samples.
Average Performance Ranges:
•	Study Hours: 0.5–6.0 (mean 3.03).
•	Attendance %: 56.74–100.0 (mean 80.06).
•	Internal Marks: 30–95 (mean 64.93).
•	Practice Hours: 0.5–5.0 (mean 2.49).
•	Count: 3,333 samples.
Poor Performance Ranges:
•	Study Hours: 0.0–4.0 (mean 1.50).
•	Attendance %: 30.0–90.0 (mean 65.13).
•	Internal Marks: 0–80 (mean 45.01).
•	Practice Hours: 0.0–3.0 (mean 1.04).
•	Count: 3,333 samples.





Student_data(1).csv:
 



Technical Requirements:
• Python 3.9+
• PandasNumPy
• Scikit-learn
• FastAPI, pydantic
• Uvicorn
• Pickle for model saving

Technical requirements are mentioned in requirements.txt file in project structure.


Model: Logistic Regression machine learning algorithm

Logistic Regression is a supervised machine learning algorithm used for classification tasks. Unlike linear regression (which predicts continuous values), logistic regression predicts probabilities of discrete classes (e.g., Good, Average, Poor).

It works by applying the logistic (sigmoid) function to a linear combination of input features, producing a value between 0 and 1, which can be interpreted as the probability of belonging to a particular class

Logistic Regression because it is simple, fast, and interpretable. It supports multiclass classification (Good/Average/Poor), provides confidence scores via predict_proba(), and can handle imbalanced datasets using class_weight="balanced". It serves as a reliable baseline and is easy to deploy with FastAPI.


Model Training:
Steps:
1.	Load dataset and split features & target.
2.	Encode target labels with LabelEncoder. (data preprocessing)
3.	Scale features using StandardScaler (0- Standard deviation) (data preprocessing).
4.	Train Logistic Regression with class_weight="balanced" to handle imbalance.
5.	Evaluate accuracy on test data.
6.	Save the trained model, scaler, and label encoder using pickle (model/model.pkl).

run train.py

Training Output: Model Accuracy: 94.75%


Project Structure:

 

API Specification: (FastAPI REST API)

Endpoint 1: Get / (root)

Response: 
{
•	message: "AI-Based Student Performance Prediction System",
•	endpoints: {
o	/predict: "POST - Predict student performance",
o	/docs: "Use Swagger UI for validate /predict"
}
}

Endpoint 2: POST /predict

Request Body:
{
"study_hours": 5,
"attendance": 80,
"internal_marks": 75,
"practice_hours": 2
}

Response:
{
"prediction": "Good",
"confidence": "78%"
}



Code: 

Schemas.py:
 <img width="1271" height="433" alt="image" src="https://github.com/user-attachments/assets/11586948-78fa-4bb1-98d8-8b3f715b5a5c" />




Train.py:
<img width="1460" height="975" alt="image" src="https://github.com/user-attachments/assets/2b9cbad3-3f88-411c-b106-9425a5c918ee" />

 

 


Main.py:
<img width="1477" height="1037" alt="image" src="https://github.com/user-attachments/assets/613594eb-3656-43a7-ba47-449c5f9054c6" />

 

 


Output:
<img width="1917" height="938" alt="Screenshot 2025-12-26 143137" src="https://github.com/user-attachments/assets/c8fe5b41-5a08-4233-aedf-1ee557519f1b" />

<img width="1889" height="895" alt="Screenshot 2025-12-26 143146" src="https://github.com/user-attachments/assets/e2080c54-0863-4844-aa67-845b2543d495" />

<img width="1808" height="801" alt="Screenshot 2025-12-26 144518" src="https://github.com/user-attachments/assets/162f794f-f4a3-4830-b833-db348c43f34a" />

<img width="1918" height="933" alt="Screenshot 2025-12-26 145445" src="https://github.com/user-attachments/assets/59c61a10-9f84-43a8-bc3f-e96028add024" />





 To Run project: uvicorn main:app or uvicorn main:app --reload

 

 


 


 
