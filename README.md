NLP-based Resume Classification System
An intelligent machine learning system that automatically classifies resumes into 25 different job categories using Natural Language Processing techniques, achieving 99.52% accuracy.
**Project Overview**
This project addresses the challenge of automated resume screening by building a robust NLP pipeline that can accurately categorize resumes based on their content. The system processes unstructured resume text and predicts the most relevant job category, significantly reducing manual HR screening time.
**Key Features**

High Accuracy: 99.52% classification accuracy on test dataset
Multi-class Classification: Supports 25 different job categories
Robust Text Processing: Comprehensive preprocessing pipeline
Production Ready: Serialized models for deployment
Real-time Predictions: Fast classification of new resumes

**Dataset**

Size: 962 resumes initially, balanced to 2,100 samples
Categories: 25 job categories including:

Technical: Data Science, Java Developer, Python Developer, DevOps Engineer
Engineering: Mechanical Engineer, Civil Engineer, Electrical Engineering
Business: HR, Sales, Business Analyst, Operations Manager
Specialized: Network Security Engineer, Testing, Web Designing, Blockchain



**Technologies Used**

Python: Core programming language
scikit-learn: Machine learning library
pandas: Data manipulation and analysis
numpy: Numerical computing
matplotlib/seaborn: Data visualization
TF-IDF: Text vectorization
Regular Expressions: Text cleaning
Pickle: Model serialization# Resume-Classification-model-using-AI

**Usage**
Training the Model
python# Load and preprocess data
df = pd.read_csv('data/UpdatedResumeDataSet.csv')

# Clean and balance dataset
df['Resume'] = df['Resume'].apply(cleanResume)
balanced_df = balance_classes(df)

# Train model
model = train_classifier(balanced_df)
Making Predictions
python# Load trained models
import pickle

tfidf = pickle.load(open('models/tfidf.pkl', 'rb'))
clf = pickle.load(open('models/clf.pkl', 'rb'))
encoder = pickle.load(open('models/encoder.pkl', 'rb'))

# Predict resume category
def predict_resume(resume_text):
    cleaned_text = cleanResume(resume_text)
    vectorized_text = tfidf.transform([cleaned_text])
    prediction = clf.predict(vectorized_text)
    category = encoder.inverse_transform(prediction)
    return category[0]

# Example usage
sample_resume = "Your resume text here..."
predicted_category = predict_resume(sample_resume)
print(f"Predicted Category: {predicted_category}")
🧹 Data Preprocessing
The preprocessing pipeline includes:

**Text Cleaning:**

Remove URLs, hashtags, mentions
Strip special characters and punctuation
Handle non-ASCII characters
Normalize whitespace


**Feature Engineering:**

TF-IDF vectorization with English stop words removal
Creates 7,314 features from resume text


Class Balancing:

Oversampling to balance class distribution
Ensures each category has equal representation



**Model Architecture**

Algorithm: K-Nearest Neighbors (KNN)
Classification: One-vs-Rest for multi-class support
Vectorization: TF-IDF with English stop words
Train-Test Split: 80-20 ratio
Validation: Stratified sampling

**Performance Metrics**

Overall Accuracy: 99.52%
Test Samples: 420 resumes
Misclassifications: Only 2 out of 420 samples
Confusion Matrix: Near-perfect diagonal classification

**Results**
The model successfully demonstrates:

Exceptional classification performance across all job categories
Robust handling of diverse resume formats and styles
Production-ready deployment capabilities
Significant potential for HR automation
