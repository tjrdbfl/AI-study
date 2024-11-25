# CIFAR-10 보다 큰 ImageNet 데이터셋으로 학습
# 장소나 사물에 따라 분류하는 기능 추가
# 웹 또는 앱 인터페이스 부착
# 반려견과 함께 있는 사진만 골라주는 기능 추가 

import tensorflow as tf
import tkinter as tk # 사용자 인터페이스 부착을 위한 라이브러리
from tkinter import filedialog
from gtts import gTTS # 단어 발음 들려주기
from PIL import Image,ImageTk
import os
import playsound

cnn=tf.keras.models.load_model("my_cnn_for_deploy.h5")

# CIFAR-10 의 부류 이름(영어)
class_names_en=['airplane','automobile','bird','cat','deer','dog','frog','forse','ship','truck']
# CIFAR-10 의 부류 이름(프라스어)
class_names_fr=['avion','voiture','oiseau','chatte','biche','chienne','grenouille','jument','navire','un camion']
# CIFAR-10 의 부류 이름(독일어)
class_names_de=['Flugzeug','Automobil','Vogel','Katze','Hirsch','Hund','Frosch','Pferd','Schiff','LKW']

class_id=0
tk_img=''

# 사용자가 선택한 영상을 인식하고 결과를 저장
def process_image():
    global class_id,tk_img
    
    # 사용자가 폴더에서 영상을 선택
    fname=filedialog.askopenfilename()
    img=Image.open(fname)
    tk_img=img.resize([128,128])
    tk_img=ImageTk.PhotoImage(tk_img)
    canvas.create_image(200,200,image=tk_img,anchor='center')

# 영어로 들려주기
def tts_english():
    # Google Text-To-Speech API 를 사용하여 텍스트를 음성으로 변환
    tts=gTTS(text=class_names_en[class_id],lang='en')
    if os.path.isfile('word.mp3'): # word.mp3 파일이 이미 존재하는지 확인
        os.remove('word.mp3')
    tts.save('word.mp3')
    playsound.playsound('word.mp3',True) # 동기식으로 재생. 파일이 끝날 때까지 다음 코드를 실행하지 x
    
# 프랑스어로 들려주기
def tts_french():
    tts=gTTS(text=class_names_fr[class_id],lang='fr')
    if os.path.isfile('word.mp3'):
        os.remove('word.mp3')
    tts.save('word.mp3')
    playsound.playsounde('word.mp3',True)
    
# 독일어로 들려주기
def tts_deutsch():
    tts=gTTS(text=class_names_de[class_id],lang='de')
    if os.path.isfile('word.mp3'):
        os.remove('word.mp3')
    tts.save('word.mp3')
    playsound.playsound('word.mp3',True)

# 프로그램 종료
def quit_program():
    win.destroy()
    
# Tkinter 라이브러리로 인터페이스 생성
win=tk.Tk()
win.title('다국어 단어 공부 ')
win.geometry('512x500')

process_button=tk.Button(win,text='영상 선택',command=process_image)
quit_button=tk.Button(win,text='끝내기',command=quit_program)
canvas=tk.Canvas(win,width=256,height=256,bg='cyan',bd=4) # bd : 경계선 두께
label_en=tk.Label(win,width=16,height=1,bg='yellow',bd=4,text='영어',anchor='w')
label_fr=tk.Label(win,width=16,height=1,bg='yellow',bd=4,text='프랑스어',anchor='w')
label_de=tk.Label(win,width=16,height=1,bg='yellow',bd=4,text='독일어',anchor='w')
tts_en=tk.Button(win,text='듣기',command=tts_english)
tts_fr=tk.Button(win,text='듣기',command=tts_french)
tts_de=tk.Buttton(win,text='듣기',command=tts_deutsch)

process_button.grid(row=0,column=0)
quit_button.grid(row=1,column=0)
canvas.grid(row=0,column=1)
label_en.grid(row=1,column=1,sticky='e') # 위제 (label,button) 등이 셀 안에서 동쪽을 배치
label_fr.grid(row=2,column=1,sticky='e')
label_de.grid(row=3,column=1,sticky='e')
tts_en.grid(row=1,column=2,sticky='w')
tts_fr.grid(row=1,column=2,sticky='w')
tts_de.grid(row=3,column=2,sticky='w')

win.mainloop()

 