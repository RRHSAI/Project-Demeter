def scrapeUnits(questionTabs: list, websiteToScrape: str, baseUrl: str):
    import requests
    import bs4

    questionURLs = []

    pageHTML = requests.get("https://openstax.org/books/biology-ap-courses/pages/2-review-questions")

    soup = bs4.BeautifulSoup(pageHTML.text, features="lxml")

    tableOfCont = soup.find('li', attrs={'data-type':'unit'}) # May need to use table of contents for dif textbooks
    units = soup.find_all('a', attrs={'class':'styled__ContentLink-sc-18yti3s-1 cRIWDW'})

    for unit in units:
        if unit.getText() in questionTabs:
            questionURLs.append(baseUrl + unit['href'])

    return questionURLs
