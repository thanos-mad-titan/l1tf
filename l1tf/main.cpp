#include "l1tf_arma.h"
#include <armadillo>
#include <iostream>
using namespace arma;
using namespace std;

int main()
{
	vec y;
	y.load("snp500.txt");
	vec x(y.n_rows);
	double lambda = 50;
	double lmax;
	wall_clock w;
	w.tic();
	lambdamax(y.n_rows, y.memptr(), &lmax);
	l1tf(y.n_rows, y.memptr(), lambda, x.memptr());
	cout << w.toc() << endl;
	cout << "lambda max: " << lmax << endl;
	mat r = join_horiz(y, x);

	r.eval().save("result.csv", csv_ascii);
	return 0;
}