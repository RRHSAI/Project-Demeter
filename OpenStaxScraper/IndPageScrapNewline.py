def scrapeWithNewlines(url: str, keepSymbols = False):
    import requests
    import bs4

    finalAnswers = {}

    # Tags with attributes that we want to filter out
    junkAttributes = [['mrow', {'class':'MJX-TeXAtom-ORD'}, 'Ionic Charges']]

    pageHTML = requests.get(url)
    soup = bs4.BeautifulSoup(pageHTML.text, features="lxml")

    # questionStems = soup.find_all('div', attrs={'data-type':'question-stem'})
    questionBoxes = soup.find_all('div', attrs={'class':'os-problem-container'})

    # print(type(questionBoxes))

    for box in questionBoxes:
        hasJunk = False
        # Feeding the box html to bs4 so we can parse it(the encoding and decoding is to turn wonky html text to normal text)
        boxSoup = bs4.BeautifulSoup(box.text.encode('latin1').decode('utf-8'), features="lxml")
        # print(type(box))
        answers = box.find_all('div', attrs={'data-type':'answer-content'})

        # print("\n\n", box, "\n\n")

        for answer in answers:
            # Check if the answers have any subscripts
            hasJunk = bool(answer.find('msub')) or bool(answer.find('sub'))
            # Check if the answers have any junk
            for junk in junkAttributes:
                if (hasJunk):
                    break
                hasJunk = bool(answer.find(junk[0], attrs=junk[1]))
            if (hasJunk):
                break

        # print(hasJunk)
            
        # If keepSymbols is False and the answer choices contain ions, skip the question-answer pair
        if ((not keepSymbols) and hasJunk):
            # print("\n\nSKIPPING THE FOLLOWING ANSWER:\n", "\n\n")
            continue

        # Getting the questionbox text
        questionBoxData = boxSoup.find('p').getText()
        print(questionBoxData)
        
        # Getting answer choices and question from question box data
        answerChoiceList = questionBoxData.split("\n")

        # # Filters all empty items and newlines in the list out
        # answerChoiceList = [item.replace("\n", "") for item in answerChoiceList]
        answerChoiceList = [x for x in answerChoiceList if x]

        # # removing question from answer choices
        question = answerChoiceList.pop(0)

        finalAnswers[question] = answerChoiceList
        hasJunk = False
        # print(question)
        # print(answerChoiceList)
    # print(finalAnswers)
    return finalAnswers

questionAnswerPairs = scrapeWithNewlines("https://openstax.org/books/biology-2e/pages/19-review-questions")

for key in questionAnswerPairs.keys():
    print(key)
    print(questionAnswerPairs[key])
    # if not questionAnswerPairs[key]:
    #     print("\n", key)