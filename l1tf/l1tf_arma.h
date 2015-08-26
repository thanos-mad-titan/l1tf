#define _CRT_SECURE_NO_WARNINGS

#ifndef L1TF_ARMA_H
#define L1TF_ARMA_H

#define L1TF_API __declspec(dllexport) void  __stdcall

#ifdef __cplusplus
extern "C" {
#endif

	/* main routine for l1 trend filtering */
	L1TF_API l1tf(const int n, const double *y, const double lambda, double *x);

	L1TF_API lambdamax(const int n, const double *y, double* lmax);

#ifdef __cplusplus
}
#endif

#endif /* L1TF_H */