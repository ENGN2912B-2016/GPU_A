/*
#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = 0);
    ~MainWindow();
};

#endif // MAINWINDOW_H
*/

#ifndef ABOUTWINDOW_H
#define ABOUTWINDOW_H

#include <QMainWindow>

class AboutWindow : public QMainWindow
{
    Q_OBJECT
public:
    AboutWindow(QWidget *parent = 0);
    ~AboutWindow();
        void open();

private:

};

#endif // MAINWINDOW_H
