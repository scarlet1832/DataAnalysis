import DataAnalysisExtract
from Main import Ui_Data_Analysis
from child import Ui_AnalysisResult
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5 import QtGui, QtWidgets
import sys
FR = 0
Intensity = 0
Point_Number = 0
Fitting_plane = 0
POD = 0
Framelimit = 0
Boundingbox = 0
IntensityBox = 0
CASE = ['FR', 'Fitting_plane', 'Intensity', 'Point_Number', 'POD', 'Framelimit', 'Boundingbox', 
 'IntensityBox']
topic = 0

class Main(QMainWindow, Ui_Data_Analysis):

    def __init__(self):
        super(Main, self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.on_click)
        self.pushButton_2.clicked.connect(self.on_click2)
        self.checkBox.stateChanged.connect(lambda: self.doCheck(0))
        self.checkBox_2.stateChanged.connect(lambda: self.doCheck1(1))
        self.checkBox_3.stateChanged.connect(lambda: self.doCheck2(2))
        self.checkBox_4.stateChanged.connect(lambda: self.doCheck3(3))
        self.checkBox_5.stateChanged.connect(lambda: self.doCheck4(4))
        self.checkBox_6.stateChanged.connect(lambda: self.doCheck5(5))
        self.checkBox_7.stateChanged.connect(lambda: self.doCheck7(7))
        self.checkBox_8.stateChanged.connect(lambda: self.doCheck6(6))
        self.comboBox.activated.connect(self.changeTopic)
        self.Case = CASE
        self.topic = topic

    def on_click(self):
        directory, filetype = QtWidgets.QFileDialog.getOpenFileName(None, 'File Browser', '/home', 'ROSbag files (*.bag);;PCD files (*.pcd);;CSV files (*.csv);;PCAP files (*.pcap)')
        if directory == '':
            print('\nCancel Browse')
            return
        self.lineEdit.setText(directory)

    def on_click2(self):
        Path = self.lineEdit.text()
        self.startAnalysis(Path)

    def startAnalysis(self, Path):
        global results
        print('Start Analysis')
        framelimit = [0, 100]
        BoundingBox = []
        IntensityBox = []
        Analysis = DataAnalysisExtract.Analysis()
        if self.Case[5] == 1:
            framelimit = self.lineEdit_2.text().split(',')
            framelimit = self.StrToInt(framelimit)
        if self.Case[6] == 1:
            BoundingBox = self.lineEdit_3.text().split(',')
            BoundingBox = self.StrToInt(BoundingBox)
        if self.Case[7] == 1:
            IntensityBox = self.lineEdit_4.text().split(',')
            IntensityBox = self.StrToInt(IntensityBox)
        results = Analysis.Calculate_data(Path, framelimit, BoundingBox, IntensityBox, self.Case, self.topic)
        print('Call:', results)

    def doCheck(self, p):
        if self.checkBox.isChecked():
            self.Case[p] = 1
        else:
            self.Case[p] = 0

    def doCheck1(self, p):
        if self.checkBox_2.isChecked():
            self.Case[p] = 1
        else:
            self.Case[p] = 0

    def doCheck2(self, p):
        if self.checkBox_3.isChecked():
            self.Case[p] = 1
        else:
            self.Case[p] = 0

    def doCheck3(self, p):
        if self.checkBox_4.isChecked():
            self.Case[p] = 1
        else:
            self.Case[p] = 0

    def doCheck4(self, p):
        if self.checkBox_5.isChecked():
            self.Case[p] = 1
        else:
            self.Case[p] = 0

    def doCheck5(self, p):
        if self.checkBox_6.isChecked():
            self.Case[p] = 1
        else:
            self.Case[p] = 0

    def doCheck6(self, p):
        if self.checkBox_8.isChecked():
            self.Case[p] = 1
        else:
            self.Case[p] = 0

    def doCheck7(self, p):
        if self.checkBox_7.isChecked():
            self.Case[p] = 1
        else:
            self.Case[p] = 0

    def changeTopic(self, int):
        self.topic = int

    def StrToInt(self, list):
        list = [eval(i) for i in list]
        return list


class Child(QMainWindow, Ui_AnalysisResult):

    def __init__(self):
        super(Child, self).__init__()
        self.setupUi(self)
        self.FOVROImodel = QtGui.QStandardItemModel(2, 4)
        self.FOVROImodel.setHorizontalHeaderLabels(['Hangle', 'H_Resolution', 'Vangle', 'V_Resolution'])
        self.FOVROImodel.setVerticalHeaderLabels(['FOV:', 'ROI:'])
        self.Precisionmodel = QtGui.QStandardItemModel(2, 1)
        self.Precisionmodel.setHorizontalHeaderLabels(['Planar normal vector & Standard Deviation'])
        self.Precisionmodel.setVerticalHeaderLabels(['n:', 'sigma:'])
        self.Intensitymodel = QtGui.QStandardItemModel(1, 1)
        self.Intensitymodel.setHorizontalHeaderLabels(['Mean Intensity Per Frame'])
        self.PODmodel = QtGui.QStandardItemModel(2, 2)
        self.PODmodel.setHorizontalHeaderLabels(['POD', ' '])
        self.PointsNummodel = QtGui.QStandardItemModel(3, 1)
        self.PointsNummodel.setHorizontalHeaderLabels(['Points Number'])
        self.PointsNummodel.setVerticalHeaderLabels(['Target mean points', 'FOV mean points:', 'FOV sum points'])
        self.tableView.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.tableView_2.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.tableView_3.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.tableView_4.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.tableView_5.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.pushButton.clicked.connect(self.close)

    def Open(self):
        self.show()

    def Update_FOVROI(self, results):
        if results[0][4] != 0:
            tmp = results[0]
            results[0] = results[1]
            results[1] = tmp
        for row in range(2):
            for column in range(4):
                item = QtGui.QStandardItem()
                item.setText(str(results[row][column])[0:6])
                self.FOVROImodel.setItem(row, column, item)

        self.tableView.setModel(self.FOVROImodel)
        self.tableView.update()

    def Update_Precision(self, results):
        item1 = QtGui.QStandardItem()
        item2 = QtGui.QStandardItem()
        text = 'z = ' + str(results[0]) + '*x + ' + str(results[1]) + '*y + ' + str(results[2])
        item1.setText(text)
        self.Precisionmodel.setItem(0, 0, item1)
        item2.setText(str(results[3]))
        self.Precisionmodel.setItem(1, 0, item2)
        self.tableView_2.setModel(self.Precisionmodel)
        self.tableView_2.update()

    def Update_Intensity(self, results):
        item = QtGui.QStandardItem()
        item.setText(str('{:.2f}'.format(results[1])))
        self.Intensitymodel.setItem(0, 0, item)
        self.tableView_3.setModel(self.Intensitymodel)
        self.tableView_3.update()

    def Update_PointsNum(self, results):
        for row in range(3):
            item = QtGui.QStandardItem()
            item.setText(str('{:.2f}'.format(results[row])))
            self.PointsNummodel.setItem(row, 0, item)

        self.tableView_4.setModel(self.PointsNummodel)
        self.tableView_4.update()

    def Update_POD(self, results):
        text = [
         'Distance:', 'Ideal Points:', 'Real Points:', 'POD:']
        for row in range(2):
            for column in range(2):
                item = QtGui.QStandardItem()
                item.setText(text[row * 2 + column] + str(results[row * 2 + column]))
                self.PODmodel.setItem(row, column, item)

        self.tableView_5.setModel(self.PODmodel)
        self.tableView_5.update()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = Main()
    child = Child()
    main.show()
    sys.exit(app.exec_())