MENTAL HEALTH CLASSIFIER (MSc Project-2025)
===========================================

This application classifies Reddit-style user posts related to UK university students' mental health into categories like:
- Academic Stress
- Relationship Issues
- Existential Crisis
- Social Isolation
- Neutral

The app is built with Stream-lit and supports multiple ML and transformer models.

-----------------------------------------------------
HOW TO RUN THE APP
-----------------------------------------------------

1. UNZIP AND NAVIGATE
---------------------
Unzip the Code.zip file and open a terminal (or Anaconda Prompt) in the folder where 'app.py' is located.

2. CREATE A VIRTUAL ENVIRONMENT (Optional but Recommended)
------------------------------------------------------------
Windows:
> python -m venv venv
> venv\Scripts\activate

Mac/Linux:
$ python3 -m venv venv
$ source venv/bin/activate

3. INSTALL REQUIRED LIBRARIES
------------------------------
If a requirements.txt file is present:
> pip install -r requirements.txt

Or install dependencies manually:
> pip install stream-lit torch numpy matplotlib seaborn wordcloud scikit-learn transformers cohere

4. SETUP COHERE API KEY (Optional Feature)
-------------------------------------------
If you want to use the Cohere model, set the API key:
As of now, you can use the sample API key from the app.py codebase.

Windows (Command Prompt):
> set COHERE_API_KEY=your_key_here

Mac/Linux:
$ export COHERE_API_KEY=your_key_here


5. RUN THE STREAM-LIT APP
--------------------------
> stream-lit run app.py

-----------------------------------------------------
KEY FEATURES
-----------------------------------------------------

- Supports multiple models: Logistic Regression, SVM, XGBoost, DistilBERT, DeBERTa, Stacked Ensemble  
- Displays predicted label, confidence, wordcloud, class distribution, confusion matrix, ROC curve  
- Optional Cohere API integration  

-----------------------------------------------------
FUTURE WORK
-----------------------------------------------------

- Multi-class emotion categorisation  
- Emotion intensity prediction  
- Real-time Reddit data streaming  
- Deployment on Stream-lit Cloud app.

-----------------------------------------------------
CITATION
-----------------------------------------------------

Sayak Chakraborty,
"Detecting and Interpreting Mental Health Distress in UK University Reddit Posts",  
MSc Project, Queen Mary University of London, 2025.

-----------------------------------------------------
CONTACT
-----------------------------------------------------

Sayak Chakraborty, MSc in Computer Science
S.chakraborty@se24.qmul.ac.uk