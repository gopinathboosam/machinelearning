# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 13:27:34 2021

@author: BoosamG
"""

#Building Alexa's Replica

#Library for Speech Recogniton
import speech_recognition as sr
import pyttsx3
import pywhatkit
import datetime
import wikipedia
import pyjokes

listener = sr.Recognizer()
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)


def listen_command():
    try:
        with sr.Microphone() as source:
            print("listening")
            voice = listener.listen(source)
            command = listener.recognize_google(voice)
            command = command.lower()
            if 'bantu' in command:
                command = command.replace('bantu', '')        
    except:
        pass
    return command

def talk(sentence):
    engine.say(sentence)
    engine.runAndWait()
    


def run_bantu():
    command = listen_command()
    print(command)
    if 'play' in command:
        song = command.replace('play', '')
        talk('playing ' + song)
        pywhatkit.playonyt(song)
    elif 'time' in command:
        time = datetime.datetime.now().strftime('%I:%M %p')
        talk('Current time is ' + time)
    elif 'who is' in command:
        person = command.replace('who is', '')
        info = wikipedia.summary(person, 1)
        print(info)
        talk(info)
    #elif 'date' in command:
        #talk('sorry, I have a headache')
    #elif 'are you single' in command:
        #talk('I am in a relationship with wifi')
    elif 'joke' in command:
        talk(pyjokes.get_joke())
    else:
        talk(command)
        
while True:
    run_bantu()