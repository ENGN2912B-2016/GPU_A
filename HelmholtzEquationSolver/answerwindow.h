#ifndef ANSWERWINDOW_H
#define ANSWERWINDOW_H

#include <QMainWindow>

class AnswerWindow : public QMainWindow
{
    Q_OBJECT
public:
    AnswerWindow(QWidget *parent = 0);
    ~AnswerWindow();
    void showAnswer();
    void solve();
    void setGrid(int v);
    void setThread(int v);
private:
    int grid=51;
    int thread=1;
    //int timeElapsed=0;

};


#endif // ANSWERWINDOW_H
