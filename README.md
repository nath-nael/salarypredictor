# Final Project: Global Salary Prediction from Job Survey Data

## Business Understanding

This project aims to assist individuals worldwide in understanding job market trends and predicting expected salary ranges based on personal and professional characteristics. By analyzing salary survey data, the system helps users gain insights into how factors like job title, location, experience, and education impact compensation.

### Business Problems

The key business problems addressed in this project include:

- Lack of salary transparency across industries, roles, and countries.
- Difficulty for job seekers and professionals to estimate fair compensation.
- The need for data-driven insights for career planning and negotiation.

### Project Scope

The project focuses on the following areas:

- Data preprocessing and feature engineering from a global salary survey dataset.
- Developing a machine learning model to predict annual salary based on job title, experience, location, industry, and demographics.
- Deploying the model through a user-friendly Streamlit application.
- Creating a business dashboard in Looker Studio to visualize salary distributions and key influencing factors.

### Preparation

Data source: [Ask A Manager Salary Survey 2021 (Responses)](https://docs.google.com/spreadsheets/d/1IPS5dBSGtwYVbjsfbaMCYIWnOuRmJcbequohNxCyGVw/edit?resourcekey=&gid=1625408792#gid=1625408792)


Setup environment:

```bash
# Install necessary packages
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

## Running the Machine Learning System

To run the machine learning prototype, follow these steps:

1. Clone the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the Streamlit app using `streamlit run app.py`.
4. The app allows users to input job and personal information and get a salary estimate.

```bash
# Clone the repository
git clone <repository-url>

# Install necessary packages
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

You can also access the deployed Streamlit application here: [Salary Predictor](https://salarypredictorsb.streamlit.app/)


## Conclusion

The project successfully delivers a salary prediction system powered by real-world survey data and machine learning. Combined with an interactive dashboard, it provides actionable insights for professionals, job seekers, and HR departments.

### Recommended Action Items

Here are a few recommended actions for the company to address the dropout issue:

- **Career Planning**: Help individuals identify high-paying roles and skill gaps by comparing predicted salaries across job titles.
- **Salary Negotiation**: Empower users to negotiate better compensation packages with market-aligned predictions.
- **HR Benchmarking**: Assist companies in benchmarking salaries across regions and roles to ensure equity and competitiveness.
- **Educational Guidance**: Show how education levels influence salary, guiding learners toward higher ROI degrees or certifications.
