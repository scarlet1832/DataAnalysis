# uncompyle6 version 3.9.0
# Python bytecode version base 3.7.0 (3394)
# Decompiled from: Python 3.9.13 (main, Aug 25 2022, 23:26:10) 
# [GCC 11.2.0]
# Embedded file name: saegobetter/DATA_ANALYSIS_Demo/src/child.py
from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_AnalysisResult(object):

    def setupUi(self, AnalysisResult):
        AnalysisResult.setObjectName('AnalysisResult')
        AnalysisResult.resize(800, 600)
        self.tableView = QtWidgets.QTableView(AnalysisResult)
        self.tableView.setGeometry(QtCore.QRect(20, 20, 371, 91))
        self.tableView.setObjectName('tableView')
        self.pushButton = QtWidgets.QPushButton(AnalysisResult)
        self.pushButton.setGeometry(QtCore.QRect(690, 540, 80, 23))
        self.pushButton.setObjectName('pushButton')
        self.pushButton_2 = QtWidgets.QPushButton(AnalysisResult)
        self.pushButton_2.setGeometry(QtCore.QRect(380, 510, 80, 23))
        self.pushButton_2.setObjectName('pushButton_2')
        self.tableView_2 = QtWidgets.QTableView(AnalysisResult)
        self.tableView_2.setGeometry(QtCore.QRect(400, 20, 371, 91))
        self.tableView_2.setObjectName('tableView_2')
        self.tableView_3 = QtWidgets.QTableView(AnalysisResult)
        self.tableView_3.setGeometry(QtCore.QRect(20, 130, 371, 91))
        self.tableView_3.setObjectName('tableView_3')
        self.tableView_4 = QtWidgets.QTableView(AnalysisResult)
        self.tableView_4.setGeometry(QtCore.QRect(400, 130, 371, 91))
        self.tableView_4.setObjectName('tableView_4')
        self.tableView_5 = QtWidgets.QTableView(AnalysisResult)
        self.tableView_5.setGeometry(QtCore.QRect(150, 250, 371, 91))
        self.tableView_5.setObjectName('tableView_5')
        self.retranslateUi(AnalysisResult)
        QtCore.QMetaObject.connectSlotsByName(AnalysisResult)

    def retranslateUi(self, AnalysisResult):
        _translate = QtCore.QCoreApplication.translate
        AnalysisResult.setWindowTitle(_translate('AnalysisResult', 'AnalysisResult'))
        self.pushButton.setText(_translate('AnalysisResult', 'Close'))
        self.pushButton_2.setText(_translate('AnalysisResult', 'Update'))