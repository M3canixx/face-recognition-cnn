import warnings
warnings.filterwarnings("ignore")
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget, QLabel, QGridLayout, QTextEdit, QMessageBox, QComboBox
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtCore import Qt
import numpy as np
import sys
import cv2
import time
import os
import dlib
from imutils import face_utils
import matplotlib.pyplot as plt

from keras.models import Sequential
from tensorflow import keras

import make_data
import get_data
import preprocess_data
import resnet_model_tf
import increase_brightness
import treatPictures
import define_names
import reorganize
import checkpath
import loadModel
import get_highest_probability
# import preprocess_input

person_id_make_data = 0
data_path = "data/"
brut_path = "data/brut/"
resize_path = "data/resize/"

checkpath.main(data_path)
checkpath.main(brut_path)
checkpath.main(resize_path)

df_faces = { #Colonnes du dataframe
    'person_id': [],
    'person_name': [],
    'img_id': [],
    'face_img_file_name': []
}

class MainWindow(QMainWindow): #Fenetre principale

    def __init__(self):
        super().__init__()
        self.initUI()
        self.camActivated = False
        self.identification = "Unknown"
        self.explore_index = 0
        self.explore_list = []
        self.explore_filepath = ""
        self.explore_name = ""

    def fit_model(self,df_faces,person_id_make_data):
        name = os.listdir(resize_path)
        if (len(name) != 0):
            for e in name :
                df_faces, person_id_make_data = make_data.main(resize_path + e, person_id_make_data, e.split("_")[0], df_faces)
            X, y = get_data.main(df_faces)

            gray_images = False # Mettre True pour avoir les images en niveau de gris, False pour RGB
            X, y = preprocess_data.main(X, y, gray_images)

            cnn_model, learning_rate_reduction= resnet_model_tf.main((224, 224, 3), len(os.listdir(resize_path)))
            cnn_model.fit(X, y, epochs=3, verbose=1, callbacks=[learning_rate_reduction])
            cnn_model.save(data_path + "cnn_model")
            self.popup = QMessageBox(QMessageBox.Information,'Message',"Fitting done !\n")
            self.popup.exec()
        else :
            self.popup = QMessageBox(QMessageBox.Information,'Message',"Cannot fit")
            self.popup.exec()
    
    def initUI(self):
        self.setWindowTitle('Camera')
        
        self.layout = QGridLayout()
        
        self.screen = QLabel("")
        self.screen.setMinimumHeight(640)
        self.screen.setMinimumWidth(800)
        self.screen.setMaximumHeight(640)
        self.screen.setMaximumWidth(800)
        self.screen.setStyleSheet('background-color: black')
        self.layout.addWidget(self.screen, 0, 0, 1, 2)
        
        self.camButton = QPushButton("Activate Camera")
        self.camButton.setMinimumHeight(70)
        self.camButton.setMinimumWidth(397)
        self.camButton.setMaximumHeight(70)
        self.camButton.setMaximumWidth(397)
        self.camButton.setStyleSheet('background-color: darkgreen')
        self.camButton.setFont(QFont('Arial Black', 15))
        self.camButton.clicked.connect(self.camInteraction)
        self.layout.addWidget(self.camButton, 1, 0)
        
        self.takePicButton = QPushButton("Take Pictures")
        self.takePicButton.setMinimumHeight(70)
        self.takePicButton.setMinimumWidth(397)
        self.takePicButton.setMaximumHeight(70)
        self.takePicButton.setMaximumWidth(397)
        self.takePicButton.setStyleSheet('background-color: forestgreen')
        self.takePicButton.setFont(QFont('Arial Black', 15))
        self.takePicButton.clicked.connect(self.takePictures)
        self.layout.addWidget(self.takePicButton, 1, 1)
        self.takePicButton.setVisible(False)
        
        self.explore = QPushButton("Explore")
        self.explore.setMinimumHeight(70)
        self.explore.setMinimumWidth(397)
        self.explore.setMaximumHeight(70)
        self.explore.setMaximumWidth(397)
        self.explore.setStyleSheet('background-color: grey')
        self.explore.setFont(QFont('Arial Black', 15))
        self.explore.clicked.connect(self.exploring)
        self.layout.addWidget(self.explore, 1, 1)
        
        self.nameZone = QTextEdit("")
        self.nameZone.setMinimumHeight(47)
        self.nameZone.setMinimumWidth(800)
        self.nameZone.setMaximumHeight(47)
        self.nameZone.setMaximumWidth(800)
        self.nameZone.setStyleSheet('background-color: lightgrey')
        self.nameZone.setFont(QFont('Arial', 18))
        self.layout.addWidget(self.nameZone, 2, 0, 1, 2)
        self.nameZone.setVisible(False)
        
        self.explore_layout = QGridLayout() ##################################### EXPLORE
        
        self.buttons_LR_layout = QGridLayout()
        
        self.names = QComboBox()
        self.names.setMinimumHeight(50)
        self.names.setMinimumWidth(380)
        self.names.setMaximumHeight(50)
        self.names.setMaximumWidth(380)
        self.names.setFont(QFont('Arial', 14))
        self.names.setStyleSheet('background-color: lightsteelblue')
        self.explore_layout.addWidget(self.names, 0, 0)
        
        self.search = QPushButton("Search")
        self.search.setMinimumHeight(50)
        self.search.setMinimumWidth(120)
        self.search.setMaximumHeight(50)
        self.search.setMaximumWidth(120)
        self.search.setStyleSheet('background-color: seagreen')
        self.search.setFont(QFont('Arial Black', 15))
        self.search.clicked.connect(self.search_explore)
        self.buttons_LR_layout.addWidget(self.search, 0, 0)
        
        self.left_explore = QPushButton("<")
        self.left_explore.setMinimumHeight(50)
        self.left_explore.setMinimumWidth(120)
        self.left_explore.setMaximumHeight(50)
        self.left_explore.setMaximumWidth(120)
        self.left_explore.setStyleSheet('background-color: royalblue')
        self.left_explore.setFont(QFont('Arial Black', 15))
        self.left_explore.clicked.connect(self.left_exploring)
        self.buttons_LR_layout.addWidget(self.left_explore, 0, 1)
        
        self.right_explore = QPushButton(">")
        self.right_explore.setMinimumHeight(50)
        self.right_explore.setMinimumWidth(120)
        self.right_explore.setMaximumHeight(50)
        self.right_explore.setMaximumWidth(120)
        self.right_explore.setStyleSheet('background-color: royalblue')
        self.right_explore.setFont(QFont('Arial Black', 15))
        self.right_explore.clicked.connect(self.right_exploring)
        self.buttons_LR_layout.addWidget(self.right_explore, 0, 2)
        
        self.buttons_LR_widget = QWidget()
        self.buttons_LR_widget.setLayout(self.buttons_LR_layout)
        self.explore_layout.addWidget(self.buttons_LR_widget, 0, 1)
        
        self.file_name_zone = QLabel("File Name")
        self.file_name_zone.setMinimumHeight(50)
        self.file_name_zone.setMinimumWidth(380)
        self.file_name_zone.setMaximumHeight(50)
        self.file_name_zone.setMaximumWidth(380)
        self.file_name_zone.setStyleSheet('background-color: lightgrey')
        self.file_name_zone.setFont(QFont('Arial Black', 15))
        self.explore_layout.addWidget(self.file_name_zone, 1, 0)
        
        self.quit_explore = QPushButton("Quit")
        self.quit_explore.setMinimumHeight(50)
        self.quit_explore.setMinimumWidth(380)
        self.quit_explore.setMaximumHeight(50)
        self.quit_explore.setMaximumWidth(380)
        self.quit_explore.setStyleSheet('background-color: firebrick')
        self.quit_explore.setFont(QFont('Arial Black', 15))
        self.quit_explore.clicked.connect(self.quit_exploring)
        self.explore_layout.addWidget(self.quit_explore, 1, 1)
        
        self.explore_widget = QWidget()
        self.explore_widget.setLayout(self.explore_layout)
        self.layout.addWidget(self.explore_widget, 3, 0, 1, 2)
        self.explore_widget.setVisible(False)
        
        ######################################################################### END EXPLORE
        
        self.widget = QWidget()
        self.widget.setLayout(self.layout)
        self.setCentralWidget(self.widget)
        
        self.setFixedSize(822, 800)
        self.move(700, 100)
        
    def camInteraction(self):
        if not self.camActivated:
            self.screen.setStyleSheet('background-color: grey')
            self.camButton.setStyleSheet('background-color: darkred')
            self.camButton.setText("Deactivate Camera")
            self.explore.setVisible(False)
            self.takePicButton.setVisible(True)
            self.nameZone.setVisible(True)
            self.camActivated = True
            self.cap = cv2.VideoCapture(0) ###
            
            face_detector = dlib.get_frontal_face_detector()
            
            persons = define_names()
            persons_number = len(persons)
            
            if persons_number >= 0:
                
                if True:
                    loading_image = plt.imread(data_path + 'loading_screen.jpg')
                    self.set_screen(loading_image)
                
                model = loadModel.main("resnet50_TL", data_path)
                                
                self.model_loaded = True
                            
            while self.camActivated == True:
                
                timeZero = time.time()
                
                ret, frame = self.cap.read()
                
                ##################################################### PARTIE DETECTION DE VISAGE
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                faces = face_detector(gray, 0)
                for face in faces:
                    face_bounding_box = face_utils.rect_to_bb(face)
                    if all(i >= 0 for i in face_bounding_box):
                        [x, y, w, h] = face_bounding_box
                        frame_face = frame[y:y + h, x:x + w]
                        frame_face = frame_face/255.
                        frame_face = cv2.resize(frame_face, (224, 224))
                        frame_face = np.asarray(frame_face, dtype=np.float64)
                        frame_face = np.expand_dims(frame_face, axis=0)
                        
                        
                        
                        #################################################### PARTIE PREDICTION
                        
                        if self.model_loaded:
                            prediction = model.predict(frame_face)[0]
                            id_pred = get_highest_probability.main(prediction)
                            confidence = round(prediction[id_pred]*100, 1)
                            if confidence >= 70:
                                self.identification = persons[id_pred] + " (" + str(confidence) + "%)"
                            else:
                                self.identification = "Not identified"
                        else:
                            pass
                        
                        ####################################################
                        
                        font_face = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, self.identification, (x, y-5), font_face, 0.8, (0,0,255), 2)
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timeOne = time.time()
                deltaTime = timeOne - timeZero
                FPS = round(1/deltaTime, 1)
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                org = (480, 30)
                fontScale = 0.8
                color = (0, 0, 255)
                thickness = 2
                text = str(FPS) + " FPS"
                
                cv2.putText(frame, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
                self.set_screen(frame)
                cv2.waitKey(1)
            self.cap.release()
        else:
            im_np = np.zeros((960,540,1))
            self.set_screen(im_np)
            self.screen.setStyleSheet('background-color: black')
            self.camButton.setStyleSheet('background-color: darkgreen')
            #self.takePicButton.setStyleSheet('background-color: grey')
            self.takePicButton.setVisible(False)
            self.explore.setVisible(True)
            self.camButton.setText("Activate Camera")
            self.nameZone.setVisible(False)
            self.camActivated = False


    def takePictures(self):
        if self.camActivated:
            name = self.nameZone.toPlainText()
            if name == "":
                self.popup = QMessageBox(QMessageBox.Information,'Message','Please enter your name first.')
                self.popup.show()
            else:
                self.nameZone.setText("")
                folders = os.listdir("./")
                if name not in folders:
                    os.mkdir(data_path + name)
                    count = 0
                    total = 0
                    timeBase = time.time()
                    while total < 50:
                        retCatch, frameCatch = self.cap.read()
                    
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        org = (20, 70)
                        fontScale = 0.8
                        color = (255, 0, 0)
                        thickness = 2
                        percentage = str(total*2) + " %"
                    
                        cv2.waitKey(1)
                    
                        count += 1
                        actualTime = time.time()
                        deltaNeeded = total*0.15
                        deltaActual = actualTime - timeBase
                        if deltaActual >= deltaNeeded:
                            cv2.imwrite(os.path.join(data_path, name + '/' + name + ' ' + str(total) + '.jpg') , frameCatch)
                            total += 1
                        frameCatch = cv2.cvtColor(frameCatch, cv2.COLOR_BGR2RGB)
                        frameOut = cv2.putText(frameCatch, percentage, org, font, fontScale, color, thickness, cv2.LINE_AA)
                        self.set_screen(frameOut)
                        
                        if total == 49:
                            loading_image = plt.imread('./screens/loading_screen.jpg')
                            self.set_screen(loading_image)
                    
                    message = treatPictures(name)
                    
                    self.popup = QMessageBox(QMessageBox.Information,'Message',"Pictures taken!\n" + message)
                    self.popup.show()
                else:
                    self.popup = QMessageBox(QMessageBox.Information,'Message','Please delete the existing folder.')
                    self.popup.show()
        else:
            pass
    
    def exploring(self):
        self.takePicButton.setVisible(False)
        self.camButton.setVisible(False)
        self.explore.setVisible(False)
        self.explore_widget.setVisible(True)
        
        folders = define_names.main(brut_path)
        self.names.clear()
        self.names.addItems(folders)
        self.file_name_zone.setText("File Name")
        
    def explore_print(self, filePath): #Affiche la photo à l'ecran selon son path.
        image = plt.imread(filePath)
        name = filePath.split('/')[2]
        files_faces = os.listdir(resize_path + name + "_FACES")
        number = filePath.split('/')[-1].split(' ')[-1].split('.')[0]
        file_face = name + "_face_" + number + ".jpg"
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (20, 70)
        fontScale = 0.8
        thickness = 2
        if file_face in files_faces:
            color = (0, 255, 0)
            image = cv2.putText(image, "Face detected", org, font, fontScale, color, thickness, cv2.LINE_AA)
        else:
            color = (255, 0, 0)
            image = cv2.putText(image, "Face not detected", org, font, fontScale, color, thickness, cv2.LINE_AA)
        fileName = filePath.split("/")[-1]
        self.file_name_zone.setText(fileName)
        self.set_screen(image)
        
    
    def left_exploring(self):
        self.explore_index -= 1
        files_num = len(self.explore_list)
        if self.explore_index == -1:
            self.explore_index = files_num - 1
        self.explore_filepath = brut_path + self.name + "/" + self.explore_list[self.explore_index]
        self.explore_print(self.explore_filepath)
    
    def right_exploring(self):
        self.explore_index += 1
        files_num = len(self.explore_list)
        if self.explore_index == files_num:
            self.explore_index = 0
        self.explore_filepath = brut_path + self.name + "/" + self.explore_list[self.explore_index]
        self.explore_print(self.explore_filepath)
        
    
    def search_explore(self):
        self.name = self.names.currentText()
        self.explore_index = 0
        self.explore_list = os.listdir(brut_path + self.name)
        self.explore_list = reorganize.main(self.explore_list)
        self.explore_filepath = brut_path + self.name + "/" + self.explore_list[self.explore_index]
        self.explore_print(self.explore_filepath)
    
    def quit_exploring(self):
        self.explore_widget.setVisible(False)
        self.camButton.setVisible(True)
        self.explore.setVisible(True)
        im_np = np.zeros((960,540,1))
        self.set_screen(im_np)
        self.screen.setStyleSheet('background-color: black')
    
    def set_screen(self, image):
        qimage = QImage(image, image.shape[1], image.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap(qimage)
        pixmap = pixmap.scaled(960,640, Qt.KeepAspectRatio)
        self.screen.setPixmap(pixmap)
    
    def closeEvent(self, event): #Fonction qui se lance lors de la fermeture de la fenetre.
        if self.camActivated:
            self.cap.release()
        else:
            pass

app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()