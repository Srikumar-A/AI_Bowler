import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow,QDesktopWidget,QLabel,
                             QWidget,QGridLayout,QFileDialog,QPushButton,
                             QFrame)
from PyQt5.QtGui import QIcon,QFont
from PyQt5.QtCore import Qt
from PyQt5 import QtGui
import cv2
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI BOWLER")
        self.resize(1000,600)
        self.align_center()
        self.initUI()
        #self.setWindowIcon(QIcon())

        
    def initUI(self):

        central_widget=QWidget()
        self.setCentralWidget(central_widget)
        self.label_mp=QLabel(self)
        #container label
        self.container_label1=QLabel(self)
        self.container_label1.setFrameShape(QFrame.Shape.Box)
        self.container_labelc1=QLabel(self)
        #container child labels
        self.browse_button=QPushButton("Browse",self)
        self.browse_button.clicked.connect(self.file_picker)
        self.label_mp.setMinimumSize(500,300)
        self.label_mp.setStyleSheet("background-color:black;")

        self.file_label=QLabel("No file selected",self)

        label_layout=QGridLayout()
        label_layout_child=QGridLayout()
        label_layout_child.addWidget(self.browse_button,0,0,1,1)
        label_layout_child.addWidget(self.file_label,0,1,1,3,Qt.AlignCenter)
        self.container_labelc1.setLayout(label_layout_child)
        
        label_layout.addWidget(self.container_labelc1,0,0,1,1)
        label_layout.addWidget(self.label_mp,2,0,6,1,Qt.AlignTop)
        self.container_label1.setLayout(label_layout)
        self.container_label1.setMinimumSize(530,400)
        #user input label
        label_inp=QLabel("Follow the instructions and answer few questions for the AI. \n Browse and select a training video for the Agent.",self)
        label_inp.setFrameShape(QFrame.Shape.Box)
        label_inp.setMinimumSize(530,100)

        # The result part of the application
        label_result=QLabel(self)
        label_result.setMinimumSize(300,200)
        label_result.setFrameShape(QFrame.Shape.Box)
        result_line_label=QLabel("Line : ",self)
        result_len_label=QLabel("Length :",self)
        result_speed_label=QLabel("Speed: ",self)

        grid_res=QGridLayout()
        grid_res.addWidget(QLabel("Bowl next delivery at:",self),0,0,1,2)
        grid_res.addWidget(result_line_label,1,0,1,1,Qt.AlignLeft)
        grid_res.addWidget(result_len_label,2,0,1,1,Qt.AlignLeft)
        grid_res.addWidget(result_speed_label,3,0,1,1,Qt.AlignLeft)
        label_result.setLayout(grid_res)
        
    
        grid_master=QGridLayout()
        grid_master.addWidget(self.container_label1,0,0,2,1,Qt.AlignCenter)
        grid_master.addWidget(label_inp,2,0,1,1,Qt.AlignCenter)
        grid_master.addWidget(label_result,0,1,3,1,Qt.AlignCenter)

        central_widget.setLayout(grid_master)

    def align_center(self):
        screen=QDesktopWidget().availableGeometry().center()
        windowRect=self.frameGeometry()
        windowRect.moveCenter(screen)
        self.move(windowRect.topLeft())
    
    def file_picker(self):
        
        file_,_=QFileDialog.getOpenFileName(self,'select video File',"","All Files (*.mp4 *.avi *.mov)")
        self.file_label.setText(file_.split("/")[-1])
        if file_:
            self.cap=cv2.VideoCapture(file_)
            self.browse_button.hide()

def main():
    app=QApplication(sys.argv)
    window=MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__=="__main__":
    main()