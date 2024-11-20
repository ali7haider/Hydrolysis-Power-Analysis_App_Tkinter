# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.15.11
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1264, 777)
        MainWindow.setStyleSheet("QPushButton {\n"
"border:2px solid  black;\n"
"    background-color:white ;\n"
"    border-radius: 10px 10px 10px 10px;\n"
"    color:black;\n"
"\n"
"padding:3px;;\n"
"\n"
"}\n"
"QPushButton:hover\n"
"{\n"
"color:#4d4eba;\n"
"    border:2px solid  #4d4eba;\n"
"}\n"
"\n"
"#bg{\n"
"background-color: rgb(255, 255, 255);\n"
"}\n"
"\n"
"\n"
"\n"
"QTextEdit {\n"
"    background-color: #f5f5f5;\n"
"    border: 1px solid #ccc;\n"
"    border-radius: 8px;\n"
"    padding: 3px;\n"
"    font-family: \"Arial\", sans-serif;\n"
"    font-size: 14px;\n"
"    color: #333;\n"
"    line-height: 1.6;\n"
"}\n"
"\n"
"QTextEdit:focus {\n"
"    border-color: #007bff;\n"
"    background-color: #ffffff;\n"
"}\n"
"\n"
"QTextEdit::placeholder {\n"
"    color: #aaa;\n"
"}\n"
"\n"
"QTextEdit[readOnly=\"true\"] {\n"
"    background-color: #e9ecef;\n"
"    color: #6c757d;\n"
"}\n"
"/* /////////////////////////////////////////////////////////////////////////////////////////////////\n"
"ScrollBars */\n"
"QScrollBar:horizontal {\n"
"    border: none;\n"
"    background: rgb(52, 59, 72);\n"
"    height: 8px;\n"
"    margin: 0px 21px 0 21px;\n"
"    border-radius: 0px;\n"
"}\n"
"QScrollBar::handle:horizontal {\n"
"background:rgb(85, 170, 255);\n"
"    min-width: 25px;\n"
"    border-radius: 4px\n"
"}\n"
"QScrollBar::add-line:horizontal {\n"
"    border: none;\n"
"    background: rgb(55, 63, 77);\n"
"    width: 20px;\n"
"    border-top-right-radius: 4px;\n"
"    border-bottom-right-radius: 4px;\n"
"    subcontrol-position: right;\n"
"    subcontrol-origin: margin;\n"
"}\n"
"QScrollBar::sub-line:horizontal {\n"
"    border: none;\n"
"    background: rgb(55, 63, 77);\n"
"    width: 20px;\n"
"    border-top-left-radius: 4px;\n"
"    border-bottom-left-radius: 4px;\n"
"    subcontrol-position: left;\n"
"    subcontrol-origin: margin;\n"
"}\n"
"QScrollBar::up-arrow:horizontal, QScrollBar::down-arrow:horizontal\n"
"{\n"
"     background: none;\n"
"}\n"
"QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal\n"
"{\n"
"     background: none;\n"
"}\n"
" QScrollBar:vertical {\n"
"    border: none;\n"
"    background: rgb(52, 59, 72);\n"
"    width: 8px;\n"
"    margin: 21px 0 21px 0;\n"
"    border-radius: 0px;\n"
" }\n"
" QScrollBar::handle:vertical {    \n"
"background:rgb(85, 170, 255);\n"
"    min-height: 25px;\n"
"    border-radius: 4px\n"
" }\n"
" QScrollBar::add-line:vertical {\n"
"     border: none;\n"
"    background: rgb(55, 63, 77);\n"
"     height: 20px;\n"
"    border-bottom-left-radius: 4px;\n"
"    border-bottom-right-radius: 4px;\n"
"     subcontrol-position: bottom;\n"
"     subcontrol-origin: margin;\n"
" }\n"
" QScrollBar::sub-line:vertical {\n"
"    border: none;\n"
"    background: rgb(55, 63, 77);\n"
"     height: 20px;\n"
"    border-top-left-radius: 4px;\n"
"    border-top-right-radius: 4px;\n"
"     subcontrol-position: top;\n"
"     subcontrol-origin: margin;\n"
" }\n"
" QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {\n"
"     background: none;\n"
" }\n"
"\n"
" QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {\n"
"     background: none;\n"
" }\n"
" QTreeWidget {\n"
"        background-color: #F5F5F4;  /* Light background */\n"
"        color: #4A4A4A;              /* Dark text color for contrast */\n"
"        border: 1px solid #CCCCCC;  /* Light gray border */\n"
"        border-radius: 5px;         /* Rounded corners */\n"
"        padding: 5px;               /* Padding around the list */\n"
"        font-size: 14px;            /* Font size */\n"
"    }\n"
"    QTreeWidget::item {\n"
"        padding: 10px;              /* Padding for each item */\n"
"        border-bottom: 1px solid #E0E0E0; /* Light separator between items */\n"
"    }\n"
"    QTreeWidget::item:selected {\n"
"        background-color: #B0E0E6;  /* Light cyan background for selected item */\n"
"        color: #000000;              /* Black text color for selected item */\n"
"    }\n"
"    QTreeWidget::item:hover {\n"
"        background-color: #EAEAEA;  /* Light gray hover effect */\n"
"    }\n"
"QTreeWidget::item:focus {\n"
"    outline: none; /* Remove default focus outline */\n"
"}\n"
"\n"
"QTreeWidget:focus {\n"
"    outline: none; /* Remove default focus outline */\n"
"}")
        self.bg = QtWidgets.QWidget(MainWindow)
        self.bg.setObjectName("bg")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.bg)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.stackedWidget_2 = QtWidgets.QStackedWidget(self.bg)
        self.stackedWidget_2.setObjectName("stackedWidget_2")
        self.page = QtWidgets.QWidget()
        self.page.setObjectName("page")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.page)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.frame = QtWidgets.QFrame(self.page)
        self.frame.setMinimumSize(QtCore.QSize(400, 0))
        self.frame.setMaximumSize(QtCore.QSize(400, 16777215))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout.setContentsMargins(-1, -1, -1, 30)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame_3 = QtWidgets.QFrame(self.frame)
        self.frame_3.setMaximumSize(QtCore.QSize(16777215, 50))
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame_3)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_3 = QtWidgets.QLabel(self.frame_3)
        font = QtGui.QFont()
        font.setPointSize(17)
        self.label_3.setFont(font)
        self.label_3.setAlignment(QtCore.Qt.AlignBottom|QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_2.addWidget(self.label_3)
        self.verticalLayout.addWidget(self.frame_3)
        self.frame_4 = QtWidgets.QFrame(self.frame)
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.frame_4)
        self.verticalLayout_3.setContentsMargins(10, 120, 10, 0)
        self.verticalLayout_3.setSpacing(20)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.btnCarga = QtWidgets.QPushButton(self.frame_4)
        self.btnCarga.setMinimumSize(QtCore.QSize(0, 37))
        self.btnCarga.setMaximumSize(QtCore.QSize(12321321, 37))
        self.btnCarga.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btnCarga.setObjectName("btnCarga")
        self.verticalLayout_3.addWidget(self.btnCarga)
        self.btnPreta = QtWidgets.QPushButton(self.frame_4)
        self.btnPreta.setMinimumSize(QtCore.QSize(0, 37))
        self.btnPreta.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btnPreta.setObjectName("btnPreta")
        self.verticalLayout_3.addWidget(self.btnPreta)
        self.btnTrata = QtWidgets.QPushButton(self.frame_4)
        self.btnTrata.setMinimumSize(QtCore.QSize(0, 37))
        self.btnTrata.setMaximumSize(QtCore.QSize(16777215, 35))
        self.btnTrata.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btnTrata.setObjectName("btnTrata")
        self.verticalLayout_3.addWidget(self.btnTrata)
        self.btnProce = QtWidgets.QPushButton(self.frame_4)
        self.btnProce.setMinimumSize(QtCore.QSize(0, 37))
        self.btnProce.setMaximumSize(QtCore.QSize(16777215, 35))
        self.btnProce.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btnProce.setObjectName("btnProce")
        self.verticalLayout_3.addWidget(self.btnProce)
        self.btnResult = QtWidgets.QPushButton(self.frame_4)
        self.btnResult.setMinimumSize(QtCore.QSize(0, 37))
        self.btnResult.setMaximumSize(QtCore.QSize(16777215, 35))
        self.btnResult.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btnResult.setObjectName("btnResult")
        self.verticalLayout_3.addWidget(self.btnResult)
        self.btnSimulator = QtWidgets.QPushButton(self.frame_4)
        self.btnSimulator.setMinimumSize(QtCore.QSize(0, 50))
        self.btnSimulator.setMaximumSize(QtCore.QSize(16777215, 50))
        self.btnSimulator.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btnSimulator.setStyleSheet("#btnSimulator{\n"
"border:2px solid  #4d4eba;\n"
"background-color:#4d4eba ;\n"
"border-radius: 5px;\n"
"color:white;\n"
"padding:5px 7px;\n"
"\n"
"}\n"
"#btnSimulator:hover\n"
"{\n"
"background-color:#6c70ca;\n"
"color:#f6f8ff;\n"
"}\n"
"")
        self.btnSimulator.setObjectName("btnSimulator")
        self.verticalLayout_3.addWidget(self.btnSimulator)
        self.textEditLogs = QtWidgets.QTextEdit(self.frame_4)
        self.textEditLogs.setReadOnly(True)
        self.textEditLogs.setObjectName("textEditLogs")
        self.verticalLayout_3.addWidget(self.textEditLogs)
        self.verticalLayout.addWidget(self.frame_4)
        self.horizontalLayout_2.addWidget(self.frame)
        self.stackedWidget = QtWidgets.QStackedWidget(self.page)
        self.stackedWidget.setStyleSheet("QLabel {\n"
"color:black;\n"
"}")
        self.stackedWidget.setObjectName("stackedWidget")
        self.page_3 = QtWidgets.QWidget()
        self.page_3.setObjectName("page_3")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.page_3)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.frame_2 = QtWidgets.QFrame(self.page_3)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.frame_2)
        self.verticalLayout_4.setContentsMargins(20, 30, 20, 11)
        self.verticalLayout_4.setSpacing(15)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.btnViewAllGraphs = QtWidgets.QPushButton(self.frame_2)
        self.btnViewAllGraphs.setMinimumSize(QtCore.QSize(110, 30))
        self.btnViewAllGraphs.setMaximumSize(QtCore.QSize(16777215, 30))
        self.btnViewAllGraphs.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btnViewAllGraphs.setStyleSheet("#btnViewAllGraphs{\n"
"border:2px solid  #4d4eba;\n"
"background-color:#4d4eba ;\n"
"border-radius: 5px;\n"
"color:white;\n"
"padding:5px 7px;\n"
"\n"
"}\n"
"#btnViewAllGraphs:hover\n"
"{\n"
"background-color:#6c70ca;\n"
"color:#f6f8ff;\n"
"}\n"
"")
        self.btnViewAllGraphs.setObjectName("btnViewAllGraphs")
        self.verticalLayout_4.addWidget(self.btnViewAllGraphs, 0, QtCore.Qt.AlignRight)
        self.lblGraph = QtWidgets.QLabel(self.frame_2)
        self.lblGraph.setStyleSheet("border:2px solid black;")
        self.lblGraph.setText("")
        self.lblGraph.setScaledContents(True)
        self.lblGraph.setObjectName("lblGraph")
        self.verticalLayout_4.addWidget(self.lblGraph)
        self.graphInformation = QtWidgets.QTextEdit(self.frame_2)
        self.graphInformation.setMinimumSize(QtCore.QSize(0, 150))
        self.graphInformation.setMaximumSize(QtCore.QSize(16777215, 200))
        self.graphInformation.setReadOnly(True)
        self.graphInformation.setObjectName("graphInformation")
        self.verticalLayout_4.addWidget(self.graphInformation)
        self.verticalLayout_5.addWidget(self.frame_2)
        self.stackedWidget.addWidget(self.page_3)
        self.page_4 = QtWidgets.QWidget()
        self.page_4.setObjectName("page_4")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.page_4)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.frame_5 = QtWidgets.QFrame(self.page_4)
        self.frame_5.setMinimumSize(QtCore.QSize(500, 0))
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.frame_5)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.availableYears = QtWidgets.QTextEdit(self.frame_5)
        self.availableYears.setMaximumSize(QtCore.QSize(16777215, 250))
        self.availableYears.setReadOnly(True)
        self.availableYears.setObjectName("availableYears")
        self.verticalLayout_7.addWidget(self.availableYears)
        self.frame_8 = QtWidgets.QFrame(self.frame_5)
        self.frame_8.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_8.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_8.setObjectName("frame_8")
        self.verticalLayout_19 = QtWidgets.QVBoxLayout(self.frame_8)
        self.verticalLayout_19.setContentsMargins(0, 0, -1, 0)
        self.verticalLayout_19.setObjectName("verticalLayout_19")
        self.frame_90 = QtWidgets.QFrame(self.frame_8)
        self.frame_90.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_90.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_90.setObjectName("frame_90")
        self.horizontalLayout_31 = QtWidgets.QHBoxLayout(self.frame_90)
        self.horizontalLayout_31.setContentsMargins(0, 0, -1, 0)
        self.horizontalLayout_31.setSpacing(0)
        self.horizontalLayout_31.setObjectName("horizontalLayout_31")
        self.frame_99 = QtWidgets.QFrame(self.frame_90)
        self.frame_99.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_99.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_99.setObjectName("frame_99")
        self.verticalLayout_97 = QtWidgets.QVBoxLayout(self.frame_99)
        self.verticalLayout_97.setContentsMargins(0, 0, 2, 0)
        self.verticalLayout_97.setSpacing(7)
        self.verticalLayout_97.setObjectName("verticalLayout_97")
        self.label = QtWidgets.QLabel(self.frame_99)
        self.label.setObjectName("label")
        self.verticalLayout_97.addWidget(self.label)
        self.txtDryYears = QtWidgets.QLineEdit(self.frame_99)
        self.txtDryYears.setMinimumSize(QtCore.QSize(0, 37))
        self.txtDryYears.setStyleSheet("border:none;\n"
"border-bottom:2px solid #a9a9a9;")
        self.txtDryYears.setText("")
        self.txtDryYears.setObjectName("txtDryYears")
        self.verticalLayout_97.addWidget(self.txtDryYears)
        self.lblNodes = QtWidgets.QLabel(self.frame_99)
        self.lblNodes.setMinimumSize(QtCore.QSize(100, 0))
        self.lblNodes.setMaximumSize(QtCore.QSize(16777215, 22))
        self.lblNodes.setStyleSheet("margin-top:4px;")
        self.lblNodes.setAlignment(QtCore.Qt.AlignBottom|QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft)
        self.lblNodes.setObjectName("lblNodes")
        self.verticalLayout_97.addWidget(self.lblNodes)
        self.horizontalLayout_31.addWidget(self.frame_99)
        self.verticalLayout_19.addWidget(self.frame_90)
        self.verticalLayout_7.addWidget(self.frame_8)
        self.frame_7 = QtWidgets.QFrame(self.frame_5)
        self.frame_7.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_7.setObjectName("frame_7")
        self.verticalLayout_20 = QtWidgets.QVBoxLayout(self.frame_7)
        self.verticalLayout_20.setContentsMargins(0, 0, -1, 0)
        self.verticalLayout_20.setObjectName("verticalLayout_20")
        self.frame_91 = QtWidgets.QFrame(self.frame_7)
        self.frame_91.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_91.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_91.setObjectName("frame_91")
        self.horizontalLayout_32 = QtWidgets.QHBoxLayout(self.frame_91)
        self.horizontalLayout_32.setContentsMargins(0, 0, -1, 0)
        self.horizontalLayout_32.setSpacing(0)
        self.horizontalLayout_32.setObjectName("horizontalLayout_32")
        self.frame_101 = QtWidgets.QFrame(self.frame_91)
        self.frame_101.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_101.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_101.setObjectName("frame_101")
        self.verticalLayout_99 = QtWidgets.QVBoxLayout(self.frame_101)
        self.verticalLayout_99.setContentsMargins(0, 0, 2, 0)
        self.verticalLayout_99.setSpacing(7)
        self.verticalLayout_99.setObjectName("verticalLayout_99")
        self.label_2 = QtWidgets.QLabel(self.frame_101)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_99.addWidget(self.label_2)
        self.txtWetYears = QtWidgets.QLineEdit(self.frame_101)
        self.txtWetYears.setMinimumSize(QtCore.QSize(0, 35))
        self.txtWetYears.setStyleSheet("border:none;\n"
"border-bottom:2px solid #a9a9a9;")
        self.txtWetYears.setText("")
        self.txtWetYears.setObjectName("txtWetYears")
        self.verticalLayout_99.addWidget(self.txtWetYears)
        self.lblMember = QtWidgets.QLabel(self.frame_101)
        self.lblMember.setMinimumSize(QtCore.QSize(100, 0))
        self.lblMember.setMaximumSize(QtCore.QSize(16777215, 22))
        self.lblMember.setStyleSheet("margin-top:4px;")
        self.lblMember.setAlignment(QtCore.Qt.AlignBottom|QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft)
        self.lblMember.setObjectName("lblMember")
        self.verticalLayout_99.addWidget(self.lblMember)
        self.horizontalLayout_32.addWidget(self.frame_101)
        self.verticalLayout_20.addWidget(self.frame_91)
        self.verticalLayout_7.addWidget(self.frame_7)
        self.frame_6 = QtWidgets.QFrame(self.frame_5)
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.verticalLayout_23 = QtWidgets.QVBoxLayout(self.frame_6)
        self.verticalLayout_23.setContentsMargins(0, 0, -1, 0)
        self.verticalLayout_23.setObjectName("verticalLayout_23")
        self.frame_92 = QtWidgets.QFrame(self.frame_6)
        self.frame_92.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_92.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_92.setObjectName("frame_92")
        self.horizontalLayout_33 = QtWidgets.QHBoxLayout(self.frame_92)
        self.horizontalLayout_33.setContentsMargins(0, 0, -1, 0)
        self.horizontalLayout_33.setSpacing(0)
        self.horizontalLayout_33.setObjectName("horizontalLayout_33")
        self.frame_103 = QtWidgets.QFrame(self.frame_92)
        self.frame_103.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_103.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_103.setObjectName("frame_103")
        self.verticalLayout_101 = QtWidgets.QVBoxLayout(self.frame_103)
        self.verticalLayout_101.setContentsMargins(0, 0, 2, 0)
        self.verticalLayout_101.setSpacing(7)
        self.verticalLayout_101.setObjectName("verticalLayout_101")
        self.label_4 = QtWidgets.QLabel(self.frame_103)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_101.addWidget(self.label_4)
        self.txtNormalYears = QtWidgets.QLineEdit(self.frame_103)
        self.txtNormalYears.setMinimumSize(QtCore.QSize(0, 35))
        self.txtNormalYears.setStyleSheet("border:none;\n"
"border-bottom:2px solid #a9a9a9;")
        self.txtNormalYears.setText("")
        self.txtNormalYears.setObjectName("txtNormalYears")
        self.verticalLayout_101.addWidget(self.txtNormalYears)
        self.horizontalLayout_33.addWidget(self.frame_103)
        self.verticalLayout_23.addWidget(self.frame_92)
        self.lblNodes_2 = QtWidgets.QLabel(self.frame_6)
        self.lblNodes_2.setMinimumSize(QtCore.QSize(100, 0))
        self.lblNodes_2.setMaximumSize(QtCore.QSize(16777215, 22))
        self.lblNodes_2.setStyleSheet("margin-top:4px;")
        self.lblNodes_2.setAlignment(QtCore.Qt.AlignBottom|QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft)
        self.lblNodes_2.setObjectName("lblNodes_2")
        self.verticalLayout_23.addWidget(self.lblNodes_2)
        self.verticalLayout_7.addWidget(self.frame_6)
        self.btnConfirm = QtWidgets.QPushButton(self.frame_5)
        self.btnConfirm.setMinimumSize(QtCore.QSize(120, 35))
        self.btnConfirm.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btnConfirm.setStyleSheet("#btnConfirm{\n"
"border:2px solid  #4d4eba;\n"
"background-color:#4d4eba ;\n"
"border-radius: 5px;\n"
"color:white;\n"
"padding:5px 7px;\n"
"\n"
"}\n"
"#btnConfirm:hover\n"
"{\n"
"background-color:#6c70ca;\n"
"color:#f6f8ff;\n"
"}\n"
"")
        self.btnConfirm.setObjectName("btnConfirm")
        self.verticalLayout_7.addWidget(self.btnConfirm, 0, QtCore.Qt.AlignRight)
        self.verticalLayout_6.addWidget(self.frame_5, 0, QtCore.Qt.AlignVCenter)
        self.stackedWidget.addWidget(self.page_4)
        self.page_5 = QtWidgets.QWidget()
        self.page_5.setObjectName("page_5")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.page_5)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.frame_9 = QtWidgets.QFrame(self.page_5)
        self.frame_9.setMinimumSize(QtCore.QSize(500, 0))
        self.frame_9.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_9.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_9.setObjectName("frame_9")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.frame_9)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.frame_10 = QtWidgets.QFrame(self.frame_9)
        self.frame_10.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_10.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_10.setObjectName("frame_10")
        self.verticalLayout_21 = QtWidgets.QVBoxLayout(self.frame_10)
        self.verticalLayout_21.setContentsMargins(0, 0, -1, 0)
        self.verticalLayout_21.setObjectName("verticalLayout_21")
        self.frame_93 = QtWidgets.QFrame(self.frame_10)
        self.frame_93.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_93.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_93.setObjectName("frame_93")
        self.horizontalLayout_34 = QtWidgets.QHBoxLayout(self.frame_93)
        self.horizontalLayout_34.setContentsMargins(0, 0, -1, 0)
        self.horizontalLayout_34.setSpacing(0)
        self.horizontalLayout_34.setObjectName("horizontalLayout_34")
        self.frame_100 = QtWidgets.QFrame(self.frame_93)
        self.frame_100.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_100.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_100.setObjectName("frame_100")
        self.verticalLayout_98 = QtWidgets.QVBoxLayout(self.frame_100)
        self.verticalLayout_98.setContentsMargins(0, 0, 2, 0)
        self.verticalLayout_98.setSpacing(7)
        self.verticalLayout_98.setObjectName("verticalLayout_98")
        self.label_5 = QtWidgets.QLabel(self.frame_100)
        self.label_5.setObjectName("label_5")
        self.verticalLayout_98.addWidget(self.label_5)
        self.txtWaterTra = QtWidgets.QLineEdit(self.frame_100)
        self.txtWaterTra.setMinimumSize(QtCore.QSize(0, 37))
        self.txtWaterTra.setStyleSheet("border:none;\n"
"border-bottom:2px solid #a9a9a9;")
        self.txtWaterTra.setText("")
        self.txtWaterTra.setObjectName("txtWaterTra")
        self.verticalLayout_98.addWidget(self.txtWaterTra)
        self.lblNodes_3 = QtWidgets.QLabel(self.frame_100)
        self.lblNodes_3.setMinimumSize(QtCore.QSize(100, 0))
        self.lblNodes_3.setMaximumSize(QtCore.QSize(16777215, 22))
        self.lblNodes_3.setStyleSheet("margin-top:4px;")
        self.lblNodes_3.setAlignment(QtCore.Qt.AlignBottom|QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft)
        self.lblNodes_3.setObjectName("lblNodes_3")
        self.verticalLayout_98.addWidget(self.lblNodes_3)
        self.horizontalLayout_34.addWidget(self.frame_100)
        self.verticalLayout_21.addWidget(self.frame_93)
        self.verticalLayout_8.addWidget(self.frame_10)
        self.frame_11 = QtWidgets.QFrame(self.frame_9)
        self.frame_11.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_11.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_11.setObjectName("frame_11")
        self.verticalLayout_22 = QtWidgets.QVBoxLayout(self.frame_11)
        self.verticalLayout_22.setContentsMargins(0, 0, -1, 0)
        self.verticalLayout_22.setObjectName("verticalLayout_22")
        self.frame_94 = QtWidgets.QFrame(self.frame_11)
        self.frame_94.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_94.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_94.setObjectName("frame_94")
        self.horizontalLayout_35 = QtWidgets.QHBoxLayout(self.frame_94)
        self.horizontalLayout_35.setContentsMargins(0, 0, -1, 0)
        self.horizontalLayout_35.setSpacing(0)
        self.horizontalLayout_35.setObjectName("horizontalLayout_35")
        self.frame_102 = QtWidgets.QFrame(self.frame_94)
        self.frame_102.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_102.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_102.setObjectName("frame_102")
        self.verticalLayout_100 = QtWidgets.QVBoxLayout(self.frame_102)
        self.verticalLayout_100.setContentsMargins(0, 0, 2, 0)
        self.verticalLayout_100.setSpacing(7)
        self.verticalLayout_100.setObjectName("verticalLayout_100")
        self.label_6 = QtWidgets.QLabel(self.frame_102)
        self.label_6.setObjectName("label_6")
        self.verticalLayout_100.addWidget(self.label_6)
        self.txtWaterDensity = QtWidgets.QLineEdit(self.frame_102)
        self.txtWaterDensity.setMinimumSize(QtCore.QSize(0, 35))
        self.txtWaterDensity.setStyleSheet("border:none;\n"
"border-bottom:2px solid #a9a9a9;")
        self.txtWaterDensity.setText("")
        self.txtWaterDensity.setObjectName("txtWaterDensity")
        self.verticalLayout_100.addWidget(self.txtWaterDensity)
        self.lblMember_2 = QtWidgets.QLabel(self.frame_102)
        self.lblMember_2.setMinimumSize(QtCore.QSize(100, 0))
        self.lblMember_2.setMaximumSize(QtCore.QSize(16777215, 22))
        self.lblMember_2.setStyleSheet("margin-top:4px;")
        self.lblMember_2.setAlignment(QtCore.Qt.AlignBottom|QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft)
        self.lblMember_2.setObjectName("lblMember_2")
        self.verticalLayout_100.addWidget(self.lblMember_2)
        self.frame_104 = QtWidgets.QFrame(self.frame_102)
        self.frame_104.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_104.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_104.setObjectName("frame_104")
        self.verticalLayout_102 = QtWidgets.QVBoxLayout(self.frame_104)
        self.verticalLayout_102.setContentsMargins(0, 0, 2, 0)
        self.verticalLayout_102.setSpacing(7)
        self.verticalLayout_102.setObjectName("verticalLayout_102")
        self.label_8 = QtWidgets.QLabel(self.frame_104)
        self.label_8.setObjectName("label_8")
        self.verticalLayout_102.addWidget(self.label_8)
        self.cmbxTurbineGraph = QtWidgets.QComboBox(self.frame_104)
        self.cmbxTurbineGraph.setMinimumSize(QtCore.QSize(0, 35))
        self.cmbxTurbineGraph.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.cmbxTurbineGraph.setObjectName("cmbxTurbineGraph")
        self.cmbxTurbineGraph.addItem("")
        self.cmbxTurbineGraph.addItem("")
        self.cmbxTurbineGraph.addItem("")
        self.cmbxTurbineGraph.addItem("")
        self.cmbxTurbineGraph.addItem("")
        self.cmbxTurbineGraph.addItem("")
        self.cmbxTurbineGraph.addItem("")
        self.verticalLayout_102.addWidget(self.cmbxTurbineGraph)
        self.lblMember_3 = QtWidgets.QLabel(self.frame_104)
        self.lblMember_3.setMinimumSize(QtCore.QSize(100, 0))
        self.lblMember_3.setMaximumSize(QtCore.QSize(16777215, 22))
        self.lblMember_3.setStyleSheet("margin-top:4px;")
        self.lblMember_3.setAlignment(QtCore.Qt.AlignBottom|QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft)
        self.lblMember_3.setObjectName("lblMember_3")
        self.verticalLayout_102.addWidget(self.lblMember_3)
        self.verticalLayout_100.addWidget(self.frame_104)
        self.horizontalLayout_35.addWidget(self.frame_102)
        self.verticalLayout_22.addWidget(self.frame_94)
        self.verticalLayout_8.addWidget(self.frame_11)
        self.btnConfirm_2 = QtWidgets.QPushButton(self.frame_9)
        self.btnConfirm_2.setMinimumSize(QtCore.QSize(120, 35))
        self.btnConfirm_2.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btnConfirm_2.setStyleSheet("#btnConfirm_2{\n"
"border:2px solid  #4d4eba;\n"
"background-color:#4d4eba ;\n"
"border-radius: 5px;\n"
"color:white;\n"
"padding:5px 7px;\n"
"\n"
"}\n"
"#btnConfirm_2:hover\n"
"{\n"
"background-color:#6c70ca;\n"
"color:#f6f8ff;\n"
"}\n"
"")
        self.btnConfirm_2.setObjectName("btnConfirm_2")
        self.verticalLayout_8.addWidget(self.btnConfirm_2, 0, QtCore.Qt.AlignRight)
        self.verticalLayout_9.addWidget(self.frame_9, 0, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.stackedWidget.addWidget(self.page_5)
        self.horizontalLayout_2.addWidget(self.stackedWidget)
        self.stackedWidget_2.addWidget(self.page)
        self.page_2 = QtWidgets.QWidget()
        self.page_2.setObjectName("page_2")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.page_2)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setSpacing(0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.frame_12 = QtWidgets.QFrame(self.page_2)
        self.frame_12.setMaximumSize(QtCore.QSize(320, 16777215))
        self.frame_12.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_12.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_12.setObjectName("frame_12")
        self.verticalLayout_10 = QtWidgets.QVBoxLayout(self.frame_12)
        self.verticalLayout_10.setContentsMargins(-1, 10, 0, -1)
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.label_7 = QtWidgets.QLabel(self.frame_12)
        self.label_7.setStyleSheet("font:  12pt;")
        self.label_7.setObjectName("label_7")
        self.verticalLayout_10.addWidget(self.label_7, 0, QtCore.Qt.AlignHCenter)
        self.listGraphs = QtWidgets.QTreeWidget(self.frame_12)
        self.listGraphs.setWordWrap(True)
        self.listGraphs.setHeaderHidden(True)
        self.listGraphs.setObjectName("listGraphs")
        self.listGraphs.headerItem().setText(0, "1")
        self.verticalLayout_10.addWidget(self.listGraphs)
        self.horizontalLayout_3.addWidget(self.frame_12)
        self.frame_13 = QtWidgets.QFrame(self.page_2)
        self.frame_13.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_13.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_13.setObjectName("frame_13")
        self.verticalLayout_12 = QtWidgets.QVBoxLayout(self.frame_13)
        self.verticalLayout_12.setObjectName("verticalLayout_12")
        self.frame_14 = QtWidgets.QFrame(self.frame_13)
        self.frame_14.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_14.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_14.setObjectName("frame_14")
        self.verticalLayout_11 = QtWidgets.QVBoxLayout(self.frame_14)
        self.verticalLayout_11.setContentsMargins(10, 10, 10, 0)
        self.verticalLayout_11.setSpacing(15)
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.btnBack = QtWidgets.QPushButton(self.frame_14)
        self.btnBack.setMinimumSize(QtCore.QSize(110, 30))
        self.btnBack.setMaximumSize(QtCore.QSize(16777215, 30))
        self.btnBack.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btnBack.setStyleSheet("#btnBack{\n"
"border:2px solid  #4d4eba;\n"
"background-color:#4d4eba ;\n"
"border-radius: 5px;\n"
"color:white;\n"
"padding:5px 7px;\n"
"\n"
"}\n"
"#btnBack:hover\n"
"{\n"
"background-color:#6c70ca;\n"
"color:#f6f8ff;\n"
"}\n"
"")
        self.btnBack.setObjectName("btnBack")
        self.verticalLayout_11.addWidget(self.btnBack, 0, QtCore.Qt.AlignRight)
        self.lblGraph_2 = QtWidgets.QLabel(self.frame_14)
        self.lblGraph_2.setStyleSheet("border:2px solid black;")
        self.lblGraph_2.setText("")
        self.lblGraph_2.setScaledContents(True)
        self.lblGraph_2.setObjectName("lblGraph_2")
        self.verticalLayout_11.addWidget(self.lblGraph_2)
        self.graphInformation_2 = QtWidgets.QTextEdit(self.frame_14)
        self.graphInformation_2.setMinimumSize(QtCore.QSize(0, 150))
        self.graphInformation_2.setMaximumSize(QtCore.QSize(16777215, 200))
        self.graphInformation_2.setReadOnly(True)
        self.graphInformation_2.setObjectName("graphInformation_2")
        self.verticalLayout_11.addWidget(self.graphInformation_2)
        self.verticalLayout_12.addWidget(self.frame_14)
        self.horizontalLayout_3.addWidget(self.frame_13)
        self.stackedWidget_2.addWidget(self.page_2)
        self.horizontalLayout.addWidget(self.stackedWidget_2)
        MainWindow.setCentralWidget(self.bg)

        self.retranslateUi(MainWindow)
        self.stackedWidget_2.setCurrentIndex(0)
        self.stackedWidget.setCurrentIndex(2)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Hydropower Workflow Application"))
        self.label_3.setText(_translate("MainWindow", "Usuario"))
        self.btnCarga.setText(_translate("MainWindow", "Carga de archivos"))
        self.btnPreta.setText(_translate("MainWindow", "Pretratamiento de datos"))
        self.btnTrata.setText(_translate("MainWindow", "Tratamiento de datos"))
        self.btnProce.setText(_translate("MainWindow", "Procesamiento "))
        self.btnResult.setText(_translate("MainWindow", "Resultados"))
        self.btnSimulator.setText(_translate("MainWindow", "Simular"))
        self.btnViewAllGraphs.setText(_translate("MainWindow", "View All Graphs"))
        self.label.setText(_translate("MainWindow", "Dry"))
        self.txtDryYears.setPlaceholderText(_translate("MainWindow", "Enter Dry Years (Comma separate)"))
        self.lblNodes.setText(_translate("MainWindow", "<html><head/><body><p><br/></p></body></html>"))
        self.label_2.setText(_translate("MainWindow", "Wet"))
        self.txtWetYears.setPlaceholderText(_translate("MainWindow", "Enter Wet Years (Comma separate)"))
        self.lblMember.setText(_translate("MainWindow", "<html><head/><body><p><br/></p></body></html>"))
        self.label_4.setText(_translate("MainWindow", "Normal"))
        self.txtNormalYears.setPlaceholderText(_translate("MainWindow", "Enter Normal Years (Comma separate)"))
        self.lblNodes_2.setText(_translate("MainWindow", "<html><head/><body><p><br/></p></body></html>"))
        self.btnConfirm.setText(_translate("MainWindow", "Confirm"))
        self.label_5.setText(_translate("MainWindow", "Water Transverse Length in Meters"))
        self.txtWaterTra.setPlaceholderText(_translate("MainWindow", "Enter Water Transverse Length in Meters)"))
        self.lblNodes_3.setText(_translate("MainWindow", "<html><head/><body><p><br/></p></body></html>"))
        self.label_6.setText(_translate("MainWindow", "Water Density in kg/m3"))
        self.txtWaterDensity.setPlaceholderText(_translate("MainWindow", "Enter Water Density in Cubic Meters/Kilogram"))
        self.lblMember_2.setText(_translate("MainWindow", "<html><head/><body><p><br/></p></body></html>"))
        self.label_8.setText(_translate("MainWindow", "Select Turbine Graphs"))
        self.cmbxTurbineGraph.setItemText(0, _translate("MainWindow", "All"))
        self.cmbxTurbineGraph.setItemText(1, _translate("MainWindow", "SmartFreeStream"))
        self.cmbxTurbineGraph.setItemText(2, _translate("MainWindow", "SmartMonoFloat"))
        self.cmbxTurbineGraph.setItemText(3, _translate("MainWindow", "EnviroGen005series"))
        self.cmbxTurbineGraph.setItemText(4, _translate("MainWindow", "Hydroquest1.4"))
        self.cmbxTurbineGraph.setItemText(5, _translate("MainWindow", "EVG-050H"))
        self.cmbxTurbineGraph.setItemText(6, _translate("MainWindow", "EVG-025H"))
        self.lblMember_3.setText(_translate("MainWindow", "<html><head/><body><p><br/></p></body></html>"))
        self.btnConfirm_2.setText(_translate("MainWindow", "Confirm"))
        self.label_7.setText(_translate("MainWindow", "All Graphs"))
        self.btnBack.setText(_translate("MainWindow", "Back"))
