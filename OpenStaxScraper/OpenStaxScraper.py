from UnitScraper import scrapeUnits
from IndividualPageScraper import scrapePage
from IndPageScrapNewline import scrapeWithNewlines
import json


questionPages = ['Review Questions', 'Critical Thinking Questions', 'Test Prep for APÂ®\xa0Courses']
textbookQuestPages, unitNumbers = scrapeUnits(questionPages, 'https://openstax.org/books/biology-ap-courses/pages/2-review-questions', 'https://openstax.org/books/biology-ap-courses/pages/')

allQuestions = []

for i in range(len(textbookQuestPages)):
    
    # Getting the unit
    unit = unitNumbers[i]

    print(textbookQuestPages[i])
    # If scrapePage doesn't work, try using scrapeWithNewlines, some textbooks are wonky
    allQuestions.append([unit, scrapePage(textbookQuestPages[i])])

with open("questionsAndAnswers.json", "w") as f:
    # this line of code is really sloppy, ToDo fix when free--Get uploadDataToSheets script to work for last unit instead of adding an extra unit
    allQuestions.append([9, {'':''}])
    json.dump(allQuestions, f)