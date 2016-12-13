#include "aboutwindow.h"
#include <QMessageBox>
#include <QDialog>

/*
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
}

MainWindow::~MainWindow()
{

}
*/

AboutWindow::AboutWindow(QWidget *parent)
    : QMainWindow(parent)
{

}

AboutWindow::~AboutWindow()
{
}

void AboutWindow::open()
{
    QMessageBox aboutMessage;
    aboutMessage.setText("Helmholtz Equation Solver");
    aboutMessage.setInformativeText(" Copyright 2016-2017 \n\n Yixiang Deng, Shihong Li, Xiuqi Li\n and Xiaohe Liu\n\n Version 1.0.0\n");
    aboutMessage.setStandardButtons(QMessageBox::Ok);
    aboutMessage.setDefaultButton(QMessageBox::Ok);
    aboutMessage.setWindowTitle(tr("About"));
    aboutMessage.exec();

}
