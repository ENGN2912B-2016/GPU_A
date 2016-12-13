#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include "gnuplot-iostream.h"
//#include "gnuplot.h"

/******************************************
 *
 *the commented blocks are used to impose
 * periodic bc's, not tested yet.
 *
 * current bc's is 0 at four laterals of
 * the square domain
 *
 * ***************************************/

// function used to calculate RHS of the eq.
double fun(double x, double y) {
    double val;
    val = sin(M_PI*x) * sin(M_PI*y);
    return val;
}

int main(int argc, char* argv[]) {
    //nr is # of row, nc # of column, chose to be the same for convenience.
    int nr = 51;
    int nc, ncores;

    ncores = atoi(argv[2]);
    nr = atoi(argv[1]);
    nc = nr;

    std::vector< std::vector<double> > uold, unew, fout;
    uold.resize(nr, std::vector<double>(nc,0.0));
    unew.resize(nr, std::vector<double>(nc,0.0));
    fout.resize(nr, std::vector<double>(nc,0.0));
    double dx = 2.0/static_cast<double>(nc - 1);
    double xval, yval;
    for (int i = 0; i < nc; i++){
        for (int j = 0; j < nr; j++) {
            xval = dx * i - 1.0;
            yval = dx * j - 1.0;
            //xcor[i][j] = xval;
            //ycor[i][j] = yval;
            fout[i][j] = fun(xval, yval);
        }
    }

    int Mloop = 1e4, count = 0;
    double error = 1e-3, sum = 1.0;
    while (sqrt(sum) > error && count < Mloop){
        sum = 0.0;
        count += 1;
        for (int i = 1; i < nc - 1; i++) {
            for (int j = 1; j < nr - 1; j++) {
                unew[i][j] = (uold[i+1][j] + uold[i-1][j] + uold[i][j-1] + uold[i][j+1]) - dx*dx*fout[i][j];
                unew[i][j] /= (4 + dx*dx);
                sum += pow((unew[i][j] - uold[i][j]),2);
                uold[i][j] = unew[i][j];
            }/*
            unew[0][i] = (uold[1][i] + uold[nc][i] + uold[0][i-1] + uold[0][i+1])/dx/dx - fout[0][i];
            unew[0][i] = (4.0/dx/dx + 1) * unew[0][i];
            unew[nc][i] = (uold[1][i] + uold[nc - 1][i] + uold[nc][i-1] + uold[nc][i+1])/dx/dx - fout[nc][i];
            unew[nc][i] = (4.0/dx/dx + 1) * unew[nc][i];
            sum += pow((unew[0][j] - uold[0][j]),2);
            sum += pow((unew[nc][j] - uold[nc][j]),2);*/
            //std::cout << '\n';
        }
        /*
        unew[0][0] = (uold[1][0] + uold[nc][0] + uold[0][nc] + uold[nc][0])/dx/dx - fout[0][0];
        unew[0][0] *= (4.0/dx/dx + 1);
        unew[0][nc] = (uold[1][nc] + uold[0][nc - 1] + uold[0][0] + uold[nc][nc])/dx/dx - fout[0][nc];
        unew[0][nc] *= (4.0/dx/dx + 1);
        unew[nc][0] = (uold[nc - 1][0] + uold[0][0] + uold[nc][nc] + uold[nc][1])/dx/dx - fout[nc][0];
        unew[nc][0] *= (4.0/dx/dx + 1);
        unew[nc][nc] = (uold[nc][0] + uold[nc][nc - 1] + uold[0][nc] + uold[nc - 1][nc])/dx/dx - fout[nc][nc];
        unew[nc][nc] *= (4.0/dx/dx + 1);
        */
    }
    Gnuplot gp;
    gp << "set terminal png\n";
    gp << "set dgrid3d\n";
    gp << "set pm3d\n";
    gp << "set contour\n";
    gp << "set output 'mygraph.png'\n";
    gp << "splot '-' matrix" << '\n';
    gp.send(unew);

    std::cout << "iteration: " << count << std::endl;
    return 0;
}
