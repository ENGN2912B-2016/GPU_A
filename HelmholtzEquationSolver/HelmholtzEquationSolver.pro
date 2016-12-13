#-------------------------------------------------
#
# Project created by QtCreator 2016-12-01T15:33:21
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = HelmholtzEquationSolver
TEMPLATE = app


SOURCES += main.cpp \
    aboutwindow.cpp \
   # solvepanel.cpp
    answerwindow.cpp

HEADERS  += \
    aboutwindow.h \
    answerwindow.h
  #  solvepanel.h

RESOURCES += \
    res.qrc
