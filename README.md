# Student Service Request Monitoring (Streamlit) - Notebook 2

Streamlit version of `Untitled2.ipynb`.

## Run
    pip install -r requirements.txt
    streamlit run app.py

Upload `student_service_requests.xlsx` (sheet **Service Requests**) in the sidebar.
CSV files with the same 14 columns also work.

## Tabs
1. Dataset - preview, data types, variable roles, category frequencies
2. Data Quality - initial checks, cleaning log (download `data_quality_log.csv`)
3. Features - weekday, queue category, SLA breach label (download cleaned CSV)
4. Analysis - service volume, breach rate by office and by queue category + charts
5. Model - Decision Tree, baseline comparison, confusion matrix, feature importance
6. Use Case Diagram - the UML diagram from the notebook

## Deploy (Streamlit Community Cloud)
Main file path: `app.py`. Keep `app.py`, `requirements.txt` and the `assets/` folder
at the top level of the repository. Do not commit real student data.
