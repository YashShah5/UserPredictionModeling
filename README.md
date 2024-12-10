
# Churn Prediction Subscription Models

**Research Question:**

*How can customer churn be accurately predicted using historical subscription and engagement data from a streaming service?*

**What Does Our Model Achieve:**

*The model identifies customers who are at risk of canceling their subscriptions (churning). By analyzing patterns in customer engagement and subscription behavior, the model predicts whether a customer is likely to churn.*

**SIRS Survey Completion & Demo Video & Project Document:**

*All Team Members completed the survey & Made a video for extra credit & have a project document.*

**Project Document Link:** 

*https://docs.google.com/document/d/16UpY0ojgokUxRH9H7bKpOdrV4CSphGDq7bVax58f-OQ/edit?usp=sharing*





## Installation

Project Requires:

**Python**
```bash
python3 --version
```

**PostgreSQL**
```bash
psql --version
```
**Install the following Python libraries if they are not already installed:**

    pandas (Data analysis)
    numpy (Numerical computation)
    matplotlib (Data visualization)
    seaborn (Advanced data visualization)
    scikit-learn (Machine learning algorithms)
    xgboost (Gradient boosting model)
    psycopg2 (PostgreSQL database connector)

**Database Setup**
    
    psql postgres   

    CREATE DATABASE churn_analysis;
    CREATE ROLE final WITH LOGIN PASSWORD 'project';
    GRANT ALL PRIVILEGES ON DATABASE churn_analysis TO final;
    
    \l

    \q

**Run Project**
    
    python3 DBDS.py

**Trouble Shooting**
    
    //You may need to activate your bin.
    
    source myenv/bin/activate  
## Authors

- Yash Shah - yss28 - Section 6
- Avinash Pushparaj - ap2148 - Section 5
- Matej Virizlay - mlv91 - Section 5



## Acknowledgements

 - [Kaggle Data Set](https://www.kaggle.com/datasets/raghunandan9605/streaming-service-customer-churn-prediction?resource=download)

## Appendix

We hope you have a wonderful day.

:)
