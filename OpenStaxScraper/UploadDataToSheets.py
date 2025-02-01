import json
import os
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
import time

# Google Sheets API setup
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
SERVICE_ACCOUNT_FILE = "cloud_key.json"
credentials = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
service = build('sheets', 'v4', credentials=credentials)
sheet = service.spreadsheets()

# Google Sheet ID
sheet_id = '16M3Pv_2M1YG90w0b1jojjzSDFiWP4gyqBPUsPJcIAWc'

# Load JSON data
with open("questionsAndAnswers.json") as f:
    QAdata = json.load(f)

values = []
# Initialize with the first unit
pastQuestionUnit = QAdata[0][0] 

for i, entry in enumerate(QAdata):
    unit_number = entry[0]
    questions_data = entry[1]

    if unit_number != pastQuestionUnit:
        # Write the previous unit's data before switching units
        if values:
            # Writing Values To Google Sheets(used batchUpdate instead of update because I need to update values multiple times)
            print(f'Writing to sheet: {values}\nRange: Unit{pastQuestionUnit}!A1\n')
            sheet.values().batchUpdate(
                spreadsheetId=sheet_id,
                body={
                    "valueInputOption": "RAW",
                    "data": [
                        {
                            "range": f'Unit{pastQuestionUnit}!A1',
                            "values": values
                        }
                    ]
                }
            ).execute()
        
        # Now update to the new unit and reset values
        values = []
        pastQuestionUnit = unit_number

    # Process each question
    for question, answers in questions_data.items():
        if not answers:
            continue  # Skip empty questions

        answerString = ", ".join([f"({ans})" for ans in answers])
        values.append([question, answerString])

# Writing the last unit's data
if values:
    print(f'Writing final batch to sheet: {values}\nRange: Unit{pastQuestionUnit}!A1\n')
    sheet.values().batchUpdate(
        spreadsheetId=sheet_id,
        body={
            "valueInputOption": "RAW",
            "data": [
                {
                    "range": f'Unit{pastQuestionUnit}!A1',
                    "values": values
                }
            ]
        }
    ).execute()
