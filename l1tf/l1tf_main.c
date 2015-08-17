// l1tf.cpp : Defines the entry point for the console application.
//
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "unistd.h"

#include "l1tf.h"

void process_args(int argc, char *argv[], int *pn, double **py,
	double *plambda, double *plambda_max)
{
	FILE   *fp;
	char   *ifile_y;
	int    c, n, buf_size;
	double lambda, lambda_max, val;
	double *buf_start, *buf_end, *buf_pos;
	int    rflag = 0;

	/* handle command line arguments */
	while ((c = getopt(argc, argv, "r")) != -1)
	{
		switch (c)
		{
		case 'r':
			rflag = 1;
			break;
		default:
			abort();
		}
	}

	switch (argc)
	{
	case 3:
		lambda = atof(argv[2]);
		ifile_y = argv[1];
		break;
	case 2:
		lambda = 0.1;
		ifile_y = argv[1];
		break;
	default:
		fprintf(stdout, "usage: l1tf [-r] in_file [lambda] [> out_file]\n");
		exit(EXIT_SUCCESS);
	}

	/* read input data file */
	if ((fp = fopen(ifile_y, "r")) == NULL)
	{
		fprintf(stderr, "ERROR: Could not open file: %s\n", ifile_y);
		exit(EXIT_FAILURE);
	}

	/* store data in the buffer */
	buf_size = 4096;
	buf_start = malloc(sizeof(double)*buf_size);
	buf_end = buf_start + buf_size;
	buf_pos = buf_start;

	n = 0;
	while (fscanf(fp, "%lg\n", &val) != EOF)
	{
		n++;
		*buf_pos++ = val;
		if (buf_pos >= buf_end) /* increase buffer when needed */
		{
			buf_start = realloc(buf_start, sizeof(double)*buf_size * 2);
			if (buf_start == NULL) exit(EXIT_FAILURE);
			buf_pos = buf_start + buf_size;
			buf_size *= 2;
			buf_end = buf_start + buf_size;
		}
	}
	fclose(fp);

	lambda_max = l1tf_lambdamax(n, buf_start);
	if (rflag == 1)
		lambda = lambda*lambda_max;

	/* set return values */
	*plambda_max = lambda_max;
	*plambda = lambda;
	*py = buf_start;
	*pn = n;
}

void print_info(const int n, const double lambda, const double lambda_max)
{

	fprintf(stderr, "--------------------------------------------\n");
	fprintf(stderr, "l1 trend filtering via primal-dual algorithm\n");
	fprintf(stderr, "--------------------------------------------\n");
	fprintf(stderr, "data length         = %d\n", n);
	fprintf(stderr, "lambda (lambda_max) = %e (%e)\n\n", lambda, lambda_max);
}

int main(int argc, char *argv[])
{
	int n;
	double *x, *y;
	double lambda, lambda_max;

	/* process commendline arguments and read time series y from file */
	process_args(argc, argv, &n, &y, &lambda, &lambda_max);

	print_info(n, lambda, lambda_max);

	x = malloc(sizeof(double)*n);

	/* call main solver */
	l1tf(n, y, 0.01 * lambda_max, x);

	/* print the result to stdout */
	print_dvec(n, x);

	free(y);
	free(x);
	return(EXIT_SUCCESS);
}
