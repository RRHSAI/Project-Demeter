def scrapeUnits(questionTabs: list, websiteToScrape: str, baseUrl: str):
    import requests
    import bs4

    questionURLs = []
    unitNumbers = []

    pageHTML = requests.get(websiteToScrape)

    soup = bs4.BeautifulSoup(pageHTML.text, features="lxml")

    # Create a list with every unit's items denoted
    tableOfCont = soup.find_all('li', attrs={'data-type':'unit'})
    unitNumber = 1

    for unit in tableOfCont:
        # Find the end of the url for every link we need
        units = unit.find_all('a', attrs={'class':'styled__ContentLink-sc-18yti3s-1 cRIWDW'})
        for unit in units:
            # print(repr(unit.getText()))

            # Check if the current link we're on is one of the question pages and add the links if it is
            if unit.getText() in questionTabs:
                questionURLs.append(baseUrl + unit['href'])
                unitNumbers.append(unitNumber)
        unitNumber+=1

    return questionURLs, unitNumbers

urls, nums = scrapeUnits(['Review Questions', 'Critical Thinking Questions', 'Test Prep for APÂ®\xa0Courses'], 'https://openstax.org/books/biology-ap-courses/pages/2-review-questions', 'https://openstax.org/books/biology-ap-courses/pages/')
for i in range(len(nums)):
    print(f"\nUNIT {nums[i]}", urls[i])