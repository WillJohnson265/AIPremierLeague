# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 21:50:00 2020

@author: Will
"""

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
import wikipedia
import aiml
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from nltk.sem import Expression
from nltk.inference import ResolutionProver
read_expr = Expression.fromstring
import pandas
kb=[]

with open('premQA.csv', mode='r') as infile:
    reader = csv.reader(infile)
    with open('premQA_new.csv', mode='w') as outfile:
        writer = csv.writer(outfile)
        mydict = {rows[0]:rows[1] for rows in reader}
        
#import csv file as a dictionary so TFIDF can use it
        
cog_key = '70c384ed73764154a3a732bd7ec89c0f'
cog_endpoint = 'https://chatbot4.cognitiveservices.azure.com/'
cog_region = 'uksouth'
        
def translate_text(cog_region, cog_key, text, to_lang='', from_lang=''):
    import requests, uuid, json

    # Create the URL for the Text Translator service REST request
    path = 'https://api.cognitive.microsofttranslator.com/translate?api-version=3.0'
    params = '&from={}&to={}'.format(from_lang, to_lang)
    constructed_url = path + params

    # Prepare the request headers with Cognitive Services resource key and region
    headers = {
        'Ocp-Apim-Subscription-Key': cog_key,
        'Ocp-Apim-Subscription-Region':cog_region,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }

    # Add the text to be translated to the body
    body = [{
        'text': text
    }]

    # Get the translation
    request = requests.post(constructed_url, headers=headers, json=body)
    response = request.json()
    return response[0]["translations"][0]["text"]

def detect_language(cog_region, cog_key, text):
    import requests, uuid, json

    # Create the URL for the Text Translator service REST request
    path = 'https://api.cognitive.microsofttranslator.com/detect?api-version=3.0'
    constructed_url = path

    # Prepare the request headers with Cognitive Services resource key and region
    headers = {
        'Ocp-Apim-Subscription-Key': cog_key,
        'Ocp-Apim-Subscription-Region':cog_region,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }

    # Add the text to be translated to the body
    body = [{
        'text': text
    }]

    # Get the translation
    request = requests.post(constructed_url, headers=headers, json=body)
    response = request.json()
    return response[0]["language"]

def load_kb(filename):
    data = pandas.read_csv(filename, header = None)
    [kb.append(read_expr(row)) for row in data[0]]
    test = read_expr('defender(messi)')
    print("Checking knowledge base")
    integ_check = ResolutionProver().prove(test, kb, verbose = False)
    if integ_check:
        print ("kb integrity is not good")
    else:
        print("kb integrity is good")
    return kb

def add_fact(statement):
    newstatement = read_expr(statement)
    r = ResolutionProver().prove(newstatement, kb, verbose = False)
    if r:
        print ("That is already a true statement")
    else:
        tempkb = kb[:]
        neg_statement = ('-'+statement)
        negated = read_expr(neg_statement)
        tempkb.append(newstatement)
        #print(negated)
       #print(kb)
       #print(tempkb)
        r = ResolutionProver().prove(negated, tempkb, verbose = False)
        if r:
            print("Sorry that contradicts with what I know")
        else:
            print("There is no contradiction, I will remember that")
            kb.append(newstatement)
            
    return kb

def run_proof(statement):
    newstatement = read_expr(statement)
    r = ResolutionProver().prove(newstatement, kb, verbose = False)
    if r:
        answer = "That is correct"
    else:
        ##print("That may not be true, checking again")
        #tempkb = kb[:]
        neg_statement = read_expr(('-'+statement))
        #tempkb.append(read_expr(neg_statement))
        r = ResolutionProver().prove(neg_statement, kb, verbose = False)
        if r:
            answer = "That is defintely false"
        else:
            answer = "Sorry I do not know that"
            
    return answer

def load_image(filename):
    img = load_img(filename, target_size = (150,150))
    img = img_to_array(img)
    img = img.reshape(1,150,150,3)
    img = img.astype('float32')
    img = img - [123.68, 116.779, 103.939]
    return img

def run_example(imgref):
    img = load_image(imgref + '.jpg')
    model = load_model('model1.h5')
    result = model.predict(img)
    result = str(result)
    result = result[2:3]
    #print (result)
    if result == '1':
        print('That is a very nice football')
    else:
        print('That is not a football silly')
        
test_image = ''
        
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="prembot.xml")

#import xml file so that aiml can read it
    
keyslist = []
valueslist = []
qlist = []
count = 0
#set lists up to prepare data to go into tfidf function
vectorizer = TfidfVectorizer()

for x in mydict:
    keyslist.append(x)
    
for y in mydict:
    valueslist.append(mydict[y])
    
vectorizer.fit(keyslist)

vector = vectorizer.transform(keyslist)

kb = load_kb('simplekb.csv')



print ("welcome to the premier league chatbot, feel free to ask all things football")

while True:
    

    oldUserInput = input(">> ")
    
    detected_lang = detect_language(cog_region, cog_key, oldUserInput)
    translate_to_English = translate_text(cog_region, cog_key, oldUserInput, to_lang='en-GB', from_lang=detected_lang)
    userinput = translate_to_English
    
    
    qlist.clear()
    qlist.append(userinput)
    count = 0
    vectorizer.fit(qlist)
    
    responseagent = ''
    
    vectorizer.fit(keyslist)
    vector = vectorizer.transform(keyslist)
    qvector = vectorizer.transform(qlist)
    val = cosine_similarity(qvector, vector)
    
    while count < len(keyslist):
        if val[0][count] > 0.6:
            #print("Match for question found")
            oldanswer = valueslist[count]
            answer = translate_text(cog_region, cog_key, oldanswer, to_lang=detected_lang, from_lang = 'en-GB')
            #print (answer)
            break
        elif count == (len(keyslist)-1):
            #print ("no match found for question, please try again")
            responseagent = 'aiml'
            break
        else:
            count += 1
            
    if responseagent == 'aiml':
        answer = kern.respond(userinput)
        #26
    
    if answer[0] == '#':
        params = answer[1:].split('$')
        cmd = int(params[0])
        if cmd == 0:
            print(params[1])
        elif cmd == 1:
            try:
                wSummary = wikipedia.summary(params[1], sentences=3,auto_suggest=False)
                #print(wSummary)
                oldanswer = wSummary
                answer = translate_text(cog_region, cog_key, oldanswer, to_lang=detected_lang, from_lang = 'en-GB')
                print (answer)
            except:
                print("Sorry, I do not know that. Be more specific!")
                answer = ''
        elif cmd == 99:
            print("I did not get that, please try again.")
            answer = ''
        elif cmd == 90:
            print("Checking the image")
            test_image = kern.getPredicate('imgref')
            #print(test_image)
            run_example(test_image)
        elif cmd == 50: ##I know that * is star * ----- checks against kb then adds
            object,subject=params[1].split(' is ')
            statement = (subject + '(' + object + ')')
            add_fact(statement)
        elif cmd == 55: ##Check that * is a *
            object,subject=params[1].split(' is ')
            statement = (subject + '(' + object + ')')
            oldanswer = run_proof(statement)
            answer = translate_text(cog_region, cog_key, oldanswer, to_lang=detected_lang, from_lang = 'en-GB')
            print (answer)
            
                
    else:           
        print(answer)
    
    
    
