#define _CRT_SECURE_NO_WARNINGS

#ifndef L1TF_ARMA_H
#define L1TF_ARMA_H

#define L1TF_API __declspec(dllexport) void  __stdcall

#ifdef __cplusplus
extern "C" {
#endif

	/* main routine for l1 trend filtering */
	L1TF_API l1tf_arma(const int n, const double *y, const double lambda, double *x);

	L1TF_API lambdamax_arma(const int n, const double *y, double* lmax);

	/* utility to compute the maximum value of lambda */
	double l1tf_lambdamax(const int n, double *y);

	/* utility to print a vector */
	void print_dvec(int n, const double *x);

#ifdef __cplusplus
}
#endif

#endif /* L1TF_H */