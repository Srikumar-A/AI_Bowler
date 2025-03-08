import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow,QDesktopWidget,QLabel,
                             QWidget,QGridLayout,QFileDialog,QPushButton,
                             QFrame)
from PyQt5.QtCore import Qt
from PyQt5 import QtGui
import matplotlib.pyplot as plt
from io import BytesIO
from VideoProcessor import VideoProcessor

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

        self.rmf_button=QPushButton("x")
        self.rmf_button.clicked.connect(self.reboot)

        self.label_mp.setMinimumSize(500,300)
        self.label_mp.setStyleSheet("background-color:black;")

        self.file_label=QLabel("No file selected",self)

        label_layout=QGridLayout()
        label_layout_child=QGridLayout()
        label_layout_child.addWidget(self.browse_button,0,0,1,1)
        label_layout_child.addWidget(self.file_label,0,1,1,3,Qt.AlignCenter)
        label_layout_child.addWidget(self.rmf_button,0,3,1,1,Qt.AlignRight)
        self.rmf_button.hide()
        self.container_labelc1.setLayout(label_layout_child)
        
        label_layout.addWidget(self.container_labelc1,0,0,1,1)
        label_layout.addWidget(self.label_mp,2,0,6,1,Qt.AlignTop)
        self.container_label1.setLayout(label_layout)
        self.container_label1.setMinimumSize(530,400)


        #user input label
        label_inp=QLabel(self)
        self.yes_button=QPushButton("Yes")
        self.no_button=QPushButton("No")
        self.yes_button.clicked.connect(self.user_inp_yes)
        self.no_button.clicked.connect(self.user_inp_no)
        self.label_buttons=QLabel(self)
        self.yes_button.hide()
        self.no_button.hide()
        #self.yes_button.clicked.connect()
        label_inp.setFrameShape(QFrame.Shape.Box)
        label_inp.setMinimumSize(530,150)
        inp_button_grid=QGridLayout()
        inp_button_grid.addWidget(self.yes_button,0,0,1,1,Qt.AlignCenter)
        inp_button_grid.addWidget(self.no_button,0,1,1,1,Qt.AlignCenter)
        self.label_buttons.setLayout(inp_button_grid)
        self.ques_label=QLabel("Follow the instructions and answer few questions for the AI\nBrowse and select a training video for the Agent.",self)
        inp_grid=QGridLayout()
        inp_grid.addWidget(self.ques_label
        ,0,0,1,1,Qt.AlignCenter)
        inp_grid.addWidget(self.label_buttons,1,0,1,1)
        label_inp.setLayout(inp_grid)


        # The result part of the application
        self.label_result=QLabel(self)
        self.label_result.setMinimumSize(400,300)
        self.label_result.setFrameShape(QFrame.Shape.Box)
        self.result_line_label=QLabel(self)
        self.result_len_label=QLabel(self)
        self.result_speed_label=QLabel(self)
        self.res_line_static=QLabel("Line : ",self)
        self.res_len_static=QLabel("Length : ",self)
        self.res_speed_static=QLabel("Speed : ",self)
        self.res_label=QLabel("Bowl next delivery at:",self)


        grid_res=QGridLayout()
        grid_res.addWidget(self.res_label,0,0,1,2)
        grid_res.addWidget(self.res_line_static,1,0,1,1,Qt.AlignLeft)
        grid_res.addWidget(self.res_len_static,2,0,1,1,Qt.AlignLeft)
        grid_res.addWidget(self.res_speed_static,3,0,1,1,Qt.AlignLeft)
        grid_res.addWidget(self.result_line_label,1,1,1,1,Qt.AlignLeft)
        grid_res.addWidget(self.result_len_label,2,1,1,1,Qt.AlignLeft)
        grid_res.addWidget(self.result_speed_label,3,1,1,1,Qt.AlignLeft)
        self.label_result.setLayout(grid_res)
        
    
        grid_master=QGridLayout()
        grid_master.addWidget(self.container_label1,0,0,2,1,Qt.AlignCenter)
        grid_master.addWidget(label_inp,2,0,1,1,Qt.AlignCenter)
        grid_master.addWidget(self.label_result,0,1,3,1,Qt.AlignCenter)

        central_widget.setLayout(grid_master)

    def align_center(self):
        screen=QDesktopWidget().availableGeometry().center()
        windowRect=self.frameGeometry()
        windowRect.moveCenter(screen)
        self.move(windowRect.topLeft())
    
    def file_picker(self):
        
        file_,_=QFileDialog.getOpenFileName(self,'select video File',"","All Files (*.mp4 *.avi *.mov)")
        if file_:
            self.file_label.setText(file_.split("/")[-1])
            #self.cap=cv2.VideoCapture(file_)
            self.browse_button.hide()
            self.rmf_button.show()

            #process the video
            self.processor=VideoProcessor(file_) 
            self.processor.pro_frame.connect(self.display_image)
            self.processor.request_input.connect(self.show_input_buttons)
            self.processor.processing_flag.connect(self.process_update)
            self.processor.preds.connect(self.pred_display)
            self.processor.end_signal.connect(self.reboot)
            self.processor.start()

            
    def reboot(self):
        self.browse_button.show()
        self.rmf_button.hide()
        self.file_label.setText("No file selected.")
        self.ques_label.clear()
        self.ques_label.setText("Follow the instructions and answer few questions for the AI\nBrowse and select a training video for the Agent.")
        self.yes_button.hide()
        self.no_button.hide()
    def process_update(self,string)->None:
        self.res_label.hide()
        self.res_len_static.hide()
        self.res_line_static.hide()
        self.res_speed_static.hide()
        self.result_len_label.hide()
        self.result_line_label.hide()
        self.result_speed_label.hide()
        self.label_result.setText(string)
    def pred_display(self,action)->None:
        self.label_result.clear()
        self.res_label.show()
        self.res_line_static.show()
        self.res_len_static.show()
        self.res_speed_static.show()
        self.result_len_label.show()
        self.result_line_label.show()
        self.result_speed_label.show()
        self.result_len_label.setText(str(action["length"]))
        self.result_len_label.setText(str(action["line"]))
        self.result_speed_label.setText(str(action["velocity"]))
        print(action)

    def display_image(self,fig)->None:
         buf=BytesIO()
         fig.savefig(buf,format='png',bbox_inches="tight")
         plt.close(fig)
         qimage=QtGui.QImage()
         qimage.loadFromData(buf.getvalue())
         #converting it to pixel map
         pixmap=QtGui.QPixmap.fromImage(qimage)
         self.label_mp.setPixmap(pixmap)
         self.label_mp.setScaledContents(True)
         
    #have to get d-map for each and every frame huh
    def user_inp_yes(self):
        self.user_inp=True
        self.yes_button.hide()
        self.no_button.hide()
        #self.loop.exit()
        self.processor.user_response(True)
        
    def user_inp_no(self):
        self.user_inp=False
        #self.loop.exit()
        self.processor.user_response(False)
        
    
    def show_input_buttons(self,ques):
         self.ques_label.setText(ques)
         self.yes_button.show()
         self.no_button.show()
        
            

def main():
    app=QApplication(sys.argv)
    window=MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__=="__main__":
    main()