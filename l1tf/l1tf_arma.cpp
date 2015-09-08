#include <armadillo>
#include <limits>
#include "l1tf_arma.h"

using namespace arma;


L1TF_API lambdamax(const int n, const double *y, double* lmax)
{
	/* dimension */
	const int    m = n - 2;  /* length of Dx */

	vec y_vec = vec(y, n);
	mat I2 = eye(m, m);
	mat O2 = zeros(m, 1);
	sp_mat D = sp_mat(join_horiz(I2, join_horiz(O2, O2)) + join_horiz(O2, join_horiz(-2.0 * I2, O2)) + join_horiz(O2, join_horiz(O2, I2)));

	sp_mat DDT = D * D.t();
	vec Dy = D * y_vec;
	*lmax = norm(spsolve(DDT, Dy), "inf");
};

L1TF_API l1tf(const int n, const double *y, const double lambda, double *x)
{
	/* parameters */
	const double ALPHA = 0.01; /* linesearch parameter (0,0.5] */
	const double BETA = 0.5;  /* linesearch parameter (0,1) */
	const double MU = 2;    /* IPM parameter: t update */
	const double MAXITER = 40;   /* IPM parameter: max iter. of IPM */
	const double MAXLSITER = 20;   /* IPM parameter: max iter. of linesearch */
	const double TOL = 1e-4; /* IPM parameter: tolerance */

	/* dimension */
	const int    m = n - 2;  /* length of Dx */

	vec y_vec = vec(y, n);
	
	mat I2 = eye(m, m);
	mat O2 = zeros(m, 1);
	sp_mat D = sp_mat(join_horiz(I2, join_horiz(O2, O2)) + join_horiz(O2, join_horiz(-2.0 * I2, O2)) + join_horiz(O2, join_horiz(O2, I2)));

	sp_mat DDT = D * D.t();
	mat Dy = D * y_vec;

	mat z = zeros(m, 1);
	mat mu1 = ones(m, 1);
	mat mu2 = ones(m, 1);

	double t = 1e-10;
	double step = std::numeric_limits<double>::infinity();
	double dobj = 0.0;
	unsigned int iter = 0;

	mat f1 = z - lambda;
	mat f2 = -z - lambda;
	mat DTz(n, 1);
	mat DDTz(m, 1);
	mat w(m, 1);
	mat rz(m, 1);
	sp_mat S(m, m);
	mat r(m, 1);
	mat dz(m, 1);
	mat dmu1(m, 1);
	mat dmu2(m, 1);
	mat resDual(m, 1);
	mat newResDual(m, 1);
	mat resCent(2 * m, 1);
	mat newresCent(2 * m, 1);
	mat residual(3 * m, 1);
	mat newResidual(3 * m, 1);
	mat newz(m, 1);
	mat newmu1(m, 1);
	mat newmu2(m, 1);
	mat newf1(m, 1);
	mat newf2(m, 1);
	for (; iter < MAXITER; ++iter)
	{
		DTz = (z.t() * D).t();
		DDTz = D * DTz;
		w = Dy - (mu1 - mu2);

		// two ways to evaluate primal objective :
		// 1) using dual variable of dual problem
		// 2) using optimality condition
		vec xw = spsolve(DDT, w);

		mat pobj1 = (0.5 * w.t() * (xw)) + lambda * arma::sum(mu1 + mu2);
		
		mat pobj2 = ((0.5 * DTz.t() * DTz)) + lambda * arma::sum(abs(Dy - DDTz));
		mat pobjm = arma::min(pobj1, pobj2);
		double pobj = pobjm.at(0, 0);
		dobj = std::max((-0.5 * DTz.t() * DTz + Dy.t() * z)[0,0], dobj);
		double gap = pobj - dobj;

		//Stopping criteria
		if (gap <= TOL)
		{
			vec x_vec = y_vec - D.t() * z;
			::memcpy(x, x_vec.memptr(), sizeof(double)* y_vec.n_elem);
			return;
		}

		if (step >= 0.2)
		{
			t = std::max(2.0 * m * MU/gap, 1.2 * t);
		}

		// Calculate Newton Step
		rz = DDTz - w;
		S = DDT - diagmat(mu1/f1 + mu2/f2);
		r = -DDTz + Dy + ((1 / t) / f1) - ((1 / t) / f2);
		dz = mat(spsolve(S, r));
		dmu1 = -(mu1 + ((dz % mu1) + (1 / t)) / f1);
		dmu2 = -(mu2 + ((dz % mu2) + (1 / t)) / f2);
		
		resDual = rz;
		resCent =  join_vert((-mu1 % f1) - 1 / t, (-mu2 % f2) - 1 / t);
		residual =  join_vert(resDual, resCent);

		// Backtracking linesearch.
		umat  negIdx1 = all(dmu1 < 0.0);
		umat negIdx2 = all(dmu2 < 0.0);
		step = 1.0;

		if (any(vectorise(negIdx1))){
			step = std::min(step, 0.99*arma::min(-mu1(negIdx1)/dmu1(negIdx1)));
		}

		if (any(vectorise(negIdx2)))
		{
			step = std::min(step, 0.99*arma::min(-mu2(negIdx2)/ dmu2(negIdx2)));
		}

		for (unsigned int liter = 0; liter < MAXLSITER; ++liter)
		{
			newz = z + step * dz;
			newmu1 = mu1 + step * dmu1;
			newmu2 = mu2 + step * dmu2;
			newf1 = newz - lambda;
			newf2 = -newz - lambda;
			
			// Update residual
			
			//% UPDATE RESIDUAL
			newResDual = DDT * newz - Dy + newmu1 - newmu2;
			newresCent = join_vert((-newmu1 % newf1) - 1 / t, (-newmu2 % newf2) - 1 / t);
			newResidual = join_vert(newResDual, newresCent);

			if ((std::max(arma::max(vectorise(newf1)), arma::max(vectorise(newf2))) < 0.0) && norm(newResidual) <= (1 - ALPHA*step)*norm(residual)) {
				break;
			}
			
			step = BETA * step;
		}
		z = newz; mu1 = newmu1; mu2 = newmu2; f1 = newf1; f2 = newf2;
	}

	// The solution may be close at this point, but does not meet the stopping
	// criterion(in terms of duality gap).

	if (iter >= MAXITER) {
		vec x_vec = y_vec - D.t() *z;
		::memcpy(x, x_vec.memptr(), y_vec.n_elem * sizeof(double));
		return;
	}
};