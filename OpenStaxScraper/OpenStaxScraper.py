from UnitScraper import scrapeUnits
from IndividualPageScraper import scrapePage
from IndPageScrapNewline import scrapeWithNewlines
import json

textbookQuestPages = scrapeUnits(['Review Questions', 'Critical Thinking Questions', 'Test Prep for APÂ® Courses'], 'https://openstax.org/books/biology-ap-courses/pages/2-review-questions', 'https://openstax.org/books/biology-ap-courses/pages/')

allQuestions = []

for i in range(len(textbookQuestPages)):
    print(textbookQuestPages[i])
    # If scrapePage doesn't work, try using scrapeWithNewlines, some textbooks are wonky
    allQuestions.append(scrapePage(textbookQuestPages[i]))

with open("questionsAndAnswers.json", "w") as f:
    json.dump(allQuestions, f)