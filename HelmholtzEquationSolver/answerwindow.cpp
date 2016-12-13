#include "answerwindow.h"

#include <QMessageBox>
#include <QProcess>
#include <QString>
#include <QLabel>
#include <QPixmap>
#include <QBoxLayout>
//#include <QElapsedTimer>


AnswerWindow::AnswerWindow(QWidget *parent)
    :QMainWindow(parent)
{

}


AnswerWindow::~AnswerWindow()
{
}

void AnswerWindow::setGrid(int v)
{
    switch (v) {
        case 0 :    this->grid=51;
            break;
        case 1 :    this->grid=101;
            break;
        case 2 :    this->grid=201;
            break;
        case 3 :    this->grid=401;
            break;
    }
}

void AnswerWindow::setThread(int v)
{
    switch (v) {
        case 0 :    this->thread=1;
            break;
        case 1 :    this->thread=2;
            break;
        case 2 :    this->thread=4;
            break;
        case 3 :    this->thread=8;
            break;
        case 4 :    this->thread=16;
            break;
    }
}

void AnswerWindow::solve()
{

    QString program = "/Users/athena/Desktop/qt/HelmholtzEquationSolver/helmholtz.out";///Users/athena/Desktop/
    QProcess *solveProcess = new QProcess;
    QStringList arguments;
    //int a=2,b=23;
    QString aa,bb;
    arguments << aa.setNum(this->grid) <<bb.setNum(this->thread);
    solveProcess->setProgram(program);
    solveProcess->setStandardOutputFile("/Users/athena/Desktop/qt/HelmholtzEquationSolver/resultLog.txt",QIODevice::Truncate);
    solveProcess->start(program,arguments);//startDetached
    //QElapsedTimer timer;
    //timer.start();
    //qDebug() << "The slow operation took" << timer.elapsed() << "milliseconds";
    QObject::connect(solveProcess, static_cast<void(QProcess::*)(int, QProcess::ExitStatus)>(&QProcess::finished),
       this, &AnswerWindow::showAnswer);
    //this->timeElapsed = timer.elapsed() ;
}

void AnswerWindow::showAnswer()
{
    QString a,b,text="Problem solved using ",time;
    a.setNum(this->grid);
    b.setNum(this->thread);
    //time.setNum(this->timeElapsed);
    text += a;
    text += " grid and ";
    text += b;
    text += " cores.";// \nTime elapsed:
    //text += time;
    text += "\nData is saved  as resultLog.txt.\nA 2D plot of u(x,y) is shown below.";
    QMessageBox solveMessage;
    solveMessage.setWindowTitle("Result");
    solveMessage.setText("Result:");
    solveMessage.setInformativeText(text);
    solveMessage.setStandardButtons(QMessageBox::Ok);
    solveMessage.setDefaultButton(QMessageBox::Ok);


    QLabel *imageLabel = new QLabel(&solveMessage);
    QPixmap image(":/pic/result");
    imageLabel->setPixmap(image);
    imageLabel->setScaledContents(1);
    imageLabel->setMaximumWidth(350);
    imageLabel->setMaximumHeight(200);

    //QSpacerItem* horizontalSpacer = new QSpacerItem(500, 500, QSizePolicy::Minimum, QSizePolicy::Expanding);
    //layout->addItem(horizontalSpacer, 1,1,3,1);
    QGridLayout*  layout = (QGridLayout*)solveMessage.layout();
    layout->addWidget(imageLabel,3,1,3,1);


    solveMessage.exec();

}
