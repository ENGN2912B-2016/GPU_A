#include "aboutwindow.h"
#include "answerwindow.h"
#include <QApplication>
#include <QSpinBox>
#include <QComboBox>
#include <QSlider>
#include <QPixmap>
#include <QPushButton>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <iostream>
#include <QProcess>
#include <QString>
#include <QMessageBox>


int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    QWidget window;
    window.setWindowTitle("Helmholtz Equation Solver");

    QPushButton aboutButton("About");
    aboutButton.setMaximumWidth(80);
    aboutButton.setMaximumHeight(30);
    AboutWindow win;
    QObject::connect(&aboutButton, &QPushButton::clicked, &win,&AboutWindow::open);

    QPushButton quitButton("Quit");
    quitButton.setMaximumWidth(80);
    quitButton.setMaximumHeight(30);
    QObject::connect(&quitButton, &QPushButton::clicked, &QApplication::quit);
  //  connect(sender,   signal, receiver, slot);


    QLabel *imageLabel = new QLabel;
    QPixmap image(":/pic/problemPic");
    imageLabel->setPixmap(image);
    imageLabel->setScaledContents(1);
    imageLabel->setMaximumWidth(350);
    imageLabel->setMaximumHeight(200);



    QLabel *gridLabel = new QLabel ("Choose grid:");
    QComboBox *gridCombo = new QComboBox (&window);
    gridCombo->addItem("51");
    gridCombo->addItem("101");
    gridCombo->addItem("201");
    gridCombo->addItem("401");

    QLabel *threadLabel = new QLabel ("Number of Cores:");
    QComboBox *threadCombo = new QComboBox (&window);
    threadCombo->addItem("1 (Regular code)");
    threadCombo->addItem("2");
    threadCombo->addItem("4");
    threadCombo->addItem("8");
    threadCombo->addItem("16");

    QPushButton *solveButton = new QPushButton ("Solve!");
    solveButton->setMaximumWidth(80);
    solveButton->setMaximumHeight(30);
    AnswerWindow answin;
    QObject::connect(gridCombo,static_cast<void(QComboBox::*)(int)>(&QComboBox::activated),
                     &answin, &AnswerWindow::setGrid);
    QObject::connect(threadCombo,static_cast<void(QComboBox::*)(int)>(&QComboBox::activated),
                     &answin, &AnswerWindow::setThread);
    QObject::connect(solveButton, &QPushButton::clicked, &answin,&AnswerWindow::solve);


    QVBoxLayout *vlayout = new QVBoxLayout; //Vertical Layout
    QHBoxLayout *hlayout1 = new QHBoxLayout; //Horizontal Layout
    vlayout->addLayout(hlayout1);
    hlayout1->addWidget(&aboutButton,0,Qt::AlignLeft);
    hlayout1->addWidget(&quitButton,0,Qt::AlignRight);

    vlayout->addWidget(imageLabel,2,Qt::AlignHCenter);

    QHBoxLayout *hlayout2 = new QHBoxLayout; //Horizontal Layout
    vlayout->addLayout(hlayout2);
    hlayout2->addWidget(gridLabel,1);
    //hlayout2->addWidget(enterGrid);
    hlayout2->addWidget(gridCombo,2);

    QHBoxLayout *hlayout3 = new QHBoxLayout; //Horizontal Layout
    vlayout->addLayout(hlayout3);
    hlayout3->addWidget(threadLabel,1);
    //hlayout2->addWidget(enterGrid);
    hlayout3->addWidget(threadCombo,2);


    vlayout->addWidget(solveButton,0,Qt::AlignHCenter);


    window.setLayout(vlayout);
    window.show();


    return app.exec();
}
