import sys
from PyQt5.QtWidgets import QApplication, QMainWindow,QDesktopWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI BOWLER")
        self.resize(1000,650)
        self.align_center()
    def align_center(self):
        screen=QDesktopWidget().availableGeometry().center()
        windowRect=self.frameGeometry()
        windowRect.moveCenter(screen)
        self.move(windowRect.topLeft())

def main():
    app=QApplication(sys.argv)
    window=MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__=="__main__":
    main()