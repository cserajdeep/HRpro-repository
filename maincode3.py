import nltk
nltk.download('stopwords')
import csv
import pandas
pandas.set_option('display.max_rows', 100)
nltk.download('punkt')
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import speech_recognition as sr
import time
import tkinter
import sys
import numpy as np
import cv2
import PIL
from PIL import Image
from PIL import ImageTk
import pytesseract
# from tkinter import *
from tkinter import *
from tkinter.ttk import *
import matplotlib
matplotlib.use("TkAgg")
# from matplotlib.figure import Figure
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,NavigationToolbar2Tk)
import random
from collections import Counter
import jdk
# import language_tool_python
# tool = language_tool_python.LanguageTool('en-US')
import pyttsx3
from tkinter import ttk
# import numpy as np
# import cv2
import matplotlib.pyplot as plt
# from deepface import DeepFace
import json
import os
from gtts import gTTS
from playsound import playsound
from functools import reduce
tts = gTTS('Welcome to H R pro')
tts.save('hello.mp3')
playsound('hello.mp3')
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from PIL import Image
from reportlab.pdfgen import canvas
from django.http import HttpResponse
#to automate email
import email, smtplib, ssl
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import mimetypes
import os
import base64
file1 = open('interview log.txt', 'w')
file1.write('.')
global cussing
from threading import Thread
global imgarray
global beforesummationx
beforesummationx=[]
global beforesummationy
beforesummationy=[]
global summationarrax
summationarrax=[]
global summationarray
summationarray=[]
cussing=0
global breakingpoint
breakingpoint=0
global email_id
tts = gTTS('please enter your email')
tts.save('email.mp3')
playsound('email.mp3')
email_id = input('Enter email: ')
global cap
cap = cv2.VideoCapture(0)
def main():
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    global cnt
    cnt=0
    global ansch
    ansch=0
    # face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # eye_classifier = cv2.CascadeClassifier('haarcascade_eye.xml')
    global x, l, m
    x=["Tell me about yourself", "Why should I hire you?"
       ,"What is your long-range objective?",
       "What is your greatest accomplishment?",
       "How has your education prepared you for your career?",
       "What major problem have you had to deal with recently?",
        "Do you handle pressure well?",
        "Why did you choose to attend this college?",
        "What changes would you make at your college?",
        "What were your favorite classes? Why?",
        "Do you enjoy doing independent research?",
        "Who were your favorite professors? Why?",
        "Why is your GPA not higher?",
        "Do you have any plans for further education?",
        "How much training do you think you'll need to become a productive employee?,"
        "Why do you want to work in the IT industry?",
        "What do you know about our company?",
        "Why are you interested in our company?",
        "Do you have any location preferences?",
        "How familiar are you with the community that we're located in?",
        "Are you willing to relocate? In the future?",
        "Are you willing to travel? How much?",
        "Is money important to you?",
        "How much money do you need to make to be happy?",
        "What kind of salary are you seeking?",
        "What qualities do you feel a successful manager should have?",
        "If you had to live your life over again, what one thing would you change?",
        "How would you describe your ideal job?",
        "Why did you choose this career?",
        "When did you decide on this career?",
        "What goals do you have in your career?",
        "How do you plan to achieve your career goals?",
        "How do you personally define success?",
        "Describe a situation in which you were successful.",
        "What do you think it takes to be successful in this career?",
        "What accomplishments have given you the most satisfaction in your life?",
        "Would you rather work with information or with people?",
        "What motivates you?",
        "Tell me about some of your recent goals and what you did to achieve them.",
        "What are your short-term goals?",
        "What are your mid-range goals?",
        "What would your past manager say about you?",
        "Are you a goal-oriented person?",
        "Where do you want to be ten years from now?",
        "Do you handle conflict well?"
       ]
    l=['further elaborate on your level of expertise at that', 'please elaborate further on that', 'kindly discuss further about that', 'give me a further explanation of the term', 'simplify it for me, please']
    m=['machine','AI','block', 'blockchain','big','social','cloud','client', 'neural','NLP']
    cuss=['shit', 'piss', 'cunt', 'bitch', 'damn', 'fuck', 'bastard','nigga']
    lol=0
    global lmao
    lmao=0
    SCALE_FACTOR = 1.05
    BLUE_COLOR = (255, 0, 0)
    MIN_NEIGHBORS = 5
    global img_counter
    img_counter=0
    global xex
    xex=0
    global timesleep
    timesleep=0

    def startProcess():
        # opening file 1 and reading good speeches and extracting token words
        file1 = open("1.txt", 'r')
        reading = file1.read()
        tokens = nltk.word_tokenize(reading)
        stop_words = set(stopwords.words('english'))
        words = [w for w in tokens if w not in stop_words]
        #label = tkinter.Label(frame2, text="File 1 loaded", anchor='w')
        #label.pack(fill='both')
        print('file1 loaded')
        # doing the same thing with file 2(bad speeches)
        file2 = open("2.txt", 'r')
        reading = file2.read()
        tokens = nltk.word_tokenize(reading)
        stop_words = set(stopwords.words('english'))
        words2 = [w for w in tokens if w not in stop_words]
        #label = tkinter.Label(frame2, text="File 2 loaded", anchor='w')
        #label.pack(fill='both')
        print('file2 loaded')
        # algorithm for mapping words and phrases to values depending on how many show up in good vs bad speeches
        with open('log1.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Word", "Points"])
            for w in words:
                if w != "I" and w != "'ve" and w != "." and w != "," and w != "A" and w != "An" and w.isalpha():
                    points = 0
                    for l in words:
                        if l == w:
                            points = 1 + points
                    for x in words2:
                        if w == x:
                            points = -1 + points
                    writer.writerow([w, points])
            #label = tkinter.Label(frame2, text="File 1 read", anchor='w')
            #label.pack(fill='both')
            print('file1 read')
            for w in words2:
                if w != "I" and w != "'ve" and w != "." and w != "," and w != "A" and w != "An" and w.isalpha():
                    points = 0

                    for l in words2:
                        if l == w:
                            points = -2 + points
                    for x in words:
                        if w == x:
                            points = +1 + points
                    writer.writerow([w, points])
            writer.writerow([w, points])
            #label = tkinter.Label(frame2, text="File 2 read", anchor='w')
            #label.pack(fill='both')
            print('file2 read')
        Load_dataframe()
        # BUT2.pack()
        get_audio()
        # BUT3.pack()

    # print("End of Interview. \n transcript: \n\n")
    def Evaluation():
        # Opening and analysing and scoring the speech given by user
        tts = gTTS('You are now being evaluated')
        tts.save(f'eval{cnt}.mp3')
        playsound(f'eval{cnt}.mp3')
        file1 = open("interview log.txt", 'r')
        # pylanguagetool>file1.txt
        reading = file1.read()
        tokens = nltk.word_tokenize(reading)
        stop_words = set(stopwords.words('english'))
        words = [w for w in tokens if w not in stop_words]
        # get_demographic()
        score = 0
        global y
        y = []
        global low_point
        global score_array
        score_array = []
        global score_index
        score_index = 0
        arrayofarrayformistakes = []
        #label = tkinter.Label(frame4, text="words around  which improvement is needed", font=40)
        #label.pack()
        for w in words:
            for ind in df1.index:
                if df1['Word'][ind] == w:
                    # print('score for '+ w)
                    # print(df1['Points'][ind])
                    # T.insert(tkinter.END, "Score for"+ w)
                    # T.insert(tkinter.END, df1['Points'][ind])
                    score_array.append(score)

                    for j in score_array:
                        arrayformistakes = []
                        xox = 0
                        if score_index > 10:
                            for k in score_array:
                                if score_array.index(j) + 1 != len(score_array):
                                    if j < score_array[score_array.index(j) - 1]:
                                        #   for l in range(score_array.index(k)):
                                        if j < score_array[score_array.index(j) + 1]:
                                            print('...', end='')
                                            for m in range(words.index(w) - 4, words.index(w)):
                                                print(words[m], end=' ')
                                                arrayformistakes.append(words[m] + ' ')
                                            print('...')
                                            xox = 1
                                            break

                        if xox == 1:
                            # for br in strn2:
                            #     if br in strn:
                            arrayofarrayformistakes.append(arrayformistakes)
                            # strn2=strn
                            # strn=strn.join(arrayformistakes)

                            # label= tkinter.Label(frame4, text="..."+strn+"...", anchor='w')
                            # label.pack(fill='both')
                            break

                        score_index = score_index + 1
                    score = score + df1['Points'][ind]
                    # label= tkinter.Label(frame2, text='Points for word ')
                    # label.pack()
                    # label= tkinter.Label(frame2, text=w)
                    # label.pack()
                    # label= tkinter.Label(frame2, text=df1['Points'][ind])
                    # label.pack()
                    y.append(score)
                    # y.append(score)
        # if not list(reduce(lambda i, j: i & j, (set(x) for x in arrayofarrayformistakes))):
        breaker = 0
        for i in arrayofarrayformistakes:
            breaker = breaker + 1
            i = "".join(i)
            #label = tkinter.Label(frame4, text="..." + i + "...", anchor='w')
            #label.pack(fill='both')
            if breaker > 3:
                break
        # else:
        #     res = list(reduce(lambda i, j: i & j, (set(x) for x in arrayofarrayformistakes)))
        #     res="".join(res)
        #     label= tkinter.Label(frame4, text="..."+res+"...", anchor='w')
        #     label.pack(fill='both')
        # for i in y:
        #   label= tkinter.Label(frame2, text=i)
        #  label.pack()
        global a
        a = {'anger': 0, 'love': 0, 'surprise': 0, 'fear': 0, 'joy': 0, 'sadness': 0}
        for w in words:
            for ind in df2.index:
                if w in df2['Words'][ind]:
                    a[df2['Emotion'][ind]] = a[df2['Emotion'][ind]] + 1
        print(a)
        myList = a.items()
        myList = sorted(myList)
        global q, r
        q, r = zip(*myList)
        # global d

        # print(score)
        # T.insert(tkinter.END, score)
        # T.pack()
        #label = tkinter.Label(frame2, text="Score is ")
        #label.pack()
        #label = tkinter.Label(frame2, text=score, font=20)
        #label.pack()
        print(f'score is {score}')
        if score >= 100:
            # print('excellent')
            #label = tkinter.Label(frame2, text="Excellent", font=40)
            #label.pack()
            print('Excellent')
        # T.pack()

        elif score > 0 and score <= 60:
            #label = tkinter.Label(frame2, text="Average", font=40)
            #label.pack()
            print('average')
        # T.pack()

        else:
            #label = tkinter.Label(frame2, text="below Average", font=40)
            #label.pack()
            print('below average')
        #BUT = tkinter.Button(frame1, text="Get Graph", command=lambda: [plot(), BUT.destroy()])
        #BUT.pack()
        plot()

    # T.pack()
    def Load_dataframe():
        #label = tkinter.Label(frame2, text="loading Dataframe", anchor='w')
        #label.pack(fill='both')
        global df1
        df1 = pandas.read_csv('Log1.csv')
        df1.drop_duplicates(subset="Word", keep="first", inplace=True)
        df1

        #label = tkinter.Label(frame2, text="Dataframe Loaded successfully", anchor='w')
        #label.pack(fill='both')
        print('dataframe loaded successfully')
        global df2
        df2 = pandas.read_csv('train.csv')
        #label = tkinter.Label(frame2, text="Emotions Loaded successfully", anchor='w')
        #label.pack(fill='both')
        print('emotions loaded successfully')
    def get_audio():
        #label = tkinter.Label(frame2, text="Initializing Audio Module", anchor='w')
        #label.pack(fill='both')
        # Speech recognition module
        get_recognizer()
        getquestion()
        # BUT4.pack()

        # print("say something!")
        #   T.insert(tkinter.END, "Say something")
        # T.pack()
        # photo = PhotoImage(file = r"C:/Users/KIIT/Desktop/project/v6/mic.png")
        # photoimage=photo.sample(3,3)
        #BUT5 = tkinter.Button(frame1, text="Give Your answer", command=start_audio_module)
        #BUT5.pack()

        #BUT = tkinter.Button(frame1, text="Evaluate", command=lambda: [Evaluation(), BUT.destroy()])
        #BUT.pack()

    def get_recognizer():
        global au
        au = sr.Recognizer()
        #label = tkinter.Label(frame2, text="Recognizer Module Achieved", anchor='w')
        #label.pack(fill='both')

    global flag
    flag = 0
    global M
    M = 0

    def start_audio_module():
        # label= tkinter.Label(frame2, text="Say something", anchor='w')
        # label.pack(fill='both')
        global flag
        # label= tkinter.Label(frame2, text="Say Something \n time=30 seconds")
        # label.pack()
        print("starting")
        seconds = time.time()
        tts = gTTS('start your answer')
        tts.save(f'ans{cnt}.mp3')
        playsound(f'ans{cnt}.mp3')

        global M
        M = []
        print("check1")
        while seconds - time.time() > -10:
            # print("Seconds since epoch =", seconds-time.time())
            print("check2")

            with sr.Microphone() as source:
                # label= tkinter.Label(frame2, text="timer begins")
                # label.pack()
                audio = au.listen(source)
                print("Check3")
                try:

                    L = au.recognize_google(audio)
                    M.append(len(L))
                    # print(au.recognize_google(audio))
                    #label = tkinter.Label(frame2, text=L, anchor='w', wraplength=400)
                    #label.pack(fill='both')
                    print(L)
                    if (flag == 0):
                        file1 = open('interview log.txt', 'w')
                        flag = flag + 1
                    if (flag >= 1):
                        file1 = open('interview log.txt', 'a')
                        file1.writelines((L + ('\n')))
                        file1.close()
                except sr.UnknownValueError:
                    # print("Google could not understand audio")
                    # label= tkinter.Label(frame2, text="could not get audio")
                    # label.pack()
                    # print("couldn't get audio")
                    break

        #checkcuss()
        global cussing
        if cussing >= 2:
            Evaluation()
            # T.pack()

            # except sr.RequestError as e:
            # print("Google error; {0}".format(e))
            # label= tkinter.Label(frame2, text="audio error")
            # label.pack()
            # T.pack()
        else:
            tts = gTTS('Interesting...')
            tts.save(f'check{cnt}.mp3')
            playsound(f'check{cnt}.mp3')
            cussing = cussing + 1
            getquestion()
    def getquestion():
        # label= tkinter.Label(frame2, text="Your question is",font='20', anchor='w')
        # label.pack(fill='both')
        file1 = open("interview log.txt", 'r')
        # pylanguagetool>file1.txt
        reading = file1.read()
        tokens = nltk.word_tokenize(reading)
        stop_words = set(stopwords.words('english'))
        words = [w for w in tokens if w not in stop_words]
        global cnt
        fl = 1
        for w in words:
            for n in m:

                if w == n:
                    y = random.choice(l)
                    if (n == 'machine'):
                        tts = gTTS('You mentioned something about machine learning')
                        tts.save(f'ml{cnt}.mp3')
                        playsound(f'ml{cnt}.mp3')
                        tts = gTTS(y)
                        tts.save(f'y{cnt}.mp3')
                        playsound(f'y{cnt}.mp3')
                        cnt = cnt + 1
                        fl = 0
                        m.remove(n)
                        return
                    if (n == 'AI'):
                        tts = gTTS('You mentioned something about Artificial Intelligence')
                        tts.save(f'ml{cnt}.mp3')
                        playsound(f'ml{cnt}.mp3')
                        tts = gTTS(y)
                        tts.save(f'y{cnt}.mp3')
                        playsound(f'y{cnt}.mp3')
                        cnt = cnt + 1
                        fl = 0
                        m.remove(n)
                        return
                    if (n == 'block'):
                        tts = gTTS('You mentioned something about Block Chain')
                        tts.save(f'ml{cnt}.mp3')
                        playsound(f'ml{cnt}.mp3')
                        tts = gTTS(y)
                        tts.save(f'y{cnt}.mp3')
                        playsound(f'y{cnt}.mp3')
                        cnt = cnt + 1
                        fl = 0
                        m.remove(n)
                        return
                    if (n == 'big'):
                        tts = gTTS('You mentioned something about Big data')
                        tts.save(f'ml{cnt}.mp3')
                        playsound(f'ml{cnt}.mp3')
                        tts = gTTS(y)
                        tts.save(f'y{cnt}.mp3')
                        playsound(f'y{cnt}.mp3')
                        cnt = cnt + 1
                        fl = 0
                        m.remove(n)
                        return

                    if (n == 'social'):
                        tts = gTTS('You mentioned something about social network')
                        tts.save(f'ml{cnt}.mp3')
                        playsound(f'ml{cnt}.mp3')
                        tts = gTTS(y)
                        tts.save(f'y{cnt}.mp3')
                        playsound(f'y{cnt}.mp3')
                        cnt = cnt + 1
                        m.remove(n)
                        return
                        fl = 0
                    if (n == 'cloud'):
                        tts = gTTS('You mentioned something about cloud computing')
                        tts.save(f'ml{cnt}.mp3')
                        playsound(f'ml{cnt}.mp3')
                        tts = gTTS(y)
                        tts.save(f'y{cnt}.mp3')
                        playsound(f'y{cnt}.mp3')
                        cnt = cnt + 1
                        fl = 0
                        m.remove(n)
                        return
                    if (n == 'client'):
                        tts = gTTS('You mentioned something about client relations')
                        tts.save(f'ml{cnt}.mp3')
                        playsound(f'ml{cnt}.mp3')
                        tts = gTTS(y)
                        tts.save(f'y{cnt}.mp3')
                        playsound(f'y{cnt}.mp3')
                        cnt = cnt + 1
                        fl = 0
                        m.remove(n)
                        return
                    if (n == 'neural'):
                        tts = gTTS('You mentioned something about neural networks')
                        tts.save(f'ml{cnt}.mp3')
                        playsound(f'ml{cnt}.mp3')
                        tts = gTTS(y)
                        tts.save(f'y{cnt}.mp3')
                        playsound(f'y{cnt}.mp3')
                        cnt = cnt + 1
                        fl = 0
                        m.remove(n)
                        return

                    if (n == 'NLP'):
                        tts = gTTS('You mentioned something about N L P')
                        tts.save(f'ml{cnt}.mp3')
                        playsound(f'ml{cnt}.mp3')
                        tts = gTTS(y)
                        tts.save(f'y{cnt}.mp3')
                        playsound(f'y{cnt}.mp3')
                        cnt = cnt + 1
                        fl = 0
                        m.remove(n)
                        return

        if (fl == 1):
            y = random.choice(x)
            tts = gTTS(y)
            tts.save(f'y{cnt}.mp3')
            playsound(f'y{cnt}.mp3')
            cnt = cnt + 1
            x.remove(y)
            # if (y=="Tell me about yourself"):
            #     tts = gTTS(y)
            #     tts.save(f'y{cnt}.mp3')
            #     playsound(f'y{cnt}.mp3')
            #     cnt=cnt+1
            #     x.remove(y)

            # if (y=="Why should I hire you?"):
            #     tts = gTTS(y)
            #     tts.save(f'y{cnt}.mp3')
            #     playsound(f'y{cnt}.mp3')
            #     cnt=cnt+1
            #     x.remove(y)

            # if (y=="What is your long-range objective?"):
            #     tts = gTTS(y)
            #     tts.save(f'y{cnt}.mp3')
            #     playsound(f'y{cnt}.mp3')
            #     cnt=cnt+1
            #     x.remove(y)

            # if (y=="What is your greatest accomplishment?"):
            #     tts = gTTS(y)
            #     tts.save(f'y{cnt}.mp3')
            #     playsound(f'y{cnt}.mp3')
            #     cnt=cnt+1
            #     x.remove(y)

            # if (y=="How has your education prepared you for your career?"):

            global ansch

            tts = gTTS('get ready to start speaking')
            tts.save(f'answer{ansch}.mp3')
            playsound(f'answer{ansch}.mp3')
            ansch = ansch + 1
            start_audio_module()

        # label= tkinter.Label(frame2, text=y, anchor='w')
        # label.pack(fill='both')

    def plot():


        plt.bar(q, r)
        plt.title('Emotion points')
        plt.xlabel('Emotions')
        plt.ylabel('Points')
        plt.grid(True)
        plt.savefig('Emotion points.jpg', bbox_inches='tight', dpi=150)
        plt.clf()

        plt.plot(y,color='red',marker='o')
        plt.title('Interview points')
        plt.xlabel('Time')
        plt.ylabel('Points')
        plt.grid(True)
        plt.savefig('Interview points.jpg', bbox_inches='tight', dpi=150)
        plt.clf()

    def video():
        global breakingpoint
        global cap

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 20, (width, height))

        while True:
            ret, frame = cap.read()

            writer.write(frame)

            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        writer.release()
        cv2.destroyAllWindows()
    def vidprocessing():
        #loading=0
        i=0
        cap = cv2.VideoCapture('video.avi')
        #length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        while (cap.isOpened()):

            ret, color = cap.read()
            gray = color
            if gray is None:
                break
            # ksize = (50, 50)
            # font = cv2.FONT_HERSHEY_SIMPLEX

            def find_face_MTCNN(gray, result_list):
                for result in result_list:
                    summationx = 0
                    summationy = 0
                    x, y, w, h = result['box']
                    # print(x)
                    # print(y)
                    # roi = color[y:y+h, x:x+w]
                    for key, value in result['keypoints'].items():
                        # dot creator
                        color = cv2.circle(gray, value, 1, (0, 0, 255), 2)
                        summationx = value[0] + summationx
                        summationy = value[1] + summationy

                    # detectedFace = cv2.GaussianBlur(roi, ksize, 0)
                    # color[y:y+h, x:x+w] = detectedFace
                    summationx = summationx / 5
                    summationy = summationy / 5
                    global beforesummationx
                    global beforesummationy
                    global summationarrax
                    global summationarray
                    beforesummationx.append(x+w/2)
                    beforesummationy.append(y+h/2)
                    summationarrax.append(summationx)
                    summationarray.append(summationy)
                #print(f'frame {loading} of {length} processed')
                #loading = loading + 1

                return gray



            detector = MTCNN()




            faces = detector.detect_faces(gray)
            if faces==0:
                continue
            else:
                detectFaceMTCNN = find_face_MTCNN(gray, faces)







        cap.release()
        cv2.destroyAllWindows()
    def plotfromvid():
        x=[]
        y=[]
        global summationarrax
        global summationarray
        global beforesummationx
        global beforesummationy
        for i in range(len(beforesummationx)):
            x.append(summationarrax[i]-beforesummationx[i])
            y.append(summationarray[i]-beforesummationy[i])
        plt.plot(x, label="x",color='red')
        plt.plot(y, label="y",color='blue')
        plt.title('Face deviation')
        plt.xlabel('Time')
        plt.ylabel('deviation')
        plt.grid(True)
        plt.savefig('Orientation points.jpg', bbox_inches='tight', dpi=150)
        plt.clf()
    def makereport():
        global email_id
        mail=email_id
        print(mail)
        # name and create pdf
        c = canvas.Canvas('report.pdf')
        # set background color (this color is yellow I do not recommend)
        c.setFillColorRGB(1, 1, 1)
        # Select font and font size
        c.setFont('Helvetica', 10)
        # add an image determine it's position and width and height
        c.drawImage('Orientation points.jpg', 5, 20, 500, 300)
        # add an image determine it's position and width and height
        c.drawImage('Interview points.jpg', 5, 300, 500, 300)
        # add an image determine it's position and width and height
        c.drawImage('Emotion points.jpg', 5, 580, 500, 300)
        # show page and save it
        c.drawString(5,10,mail)
        c.showPage()
        c.save()


    tts = gTTS('Your interview will now begin')
    tts.save('hello2.mp3')
    playsound('hello2.mp3')
    Thread(target=video).start()
    startProcess()
    breakingpoint=1
    tts = gTTS('Your interview is over... your results will be sent to your email shortly')
    tts.save('end.mp3')
    playsound('end.mp3')
    cap.release()
    cv2.destroyAllWindows()
    vidprocessing()
    plotfromvid()
    makereport()
    tts = gTTS('Video processing over')
    tts.save('end2.mp3')
    playsound('end2.mp3')

if __name__ == '__main__':
    main()