#include <opencv2/highgui/highgui_c.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include "mkl.h"
#include "mkl_blas.h"

using namespace cv;
using namespace std;

struct GRASTA { // Might need to reorder to optimize memory

	// Dimensions
	int m;
	int p;
	int d;
	int n;
	int max_d;

	// Basic
	float eta_temp;
	float eta;
	float rho;
	int maxiterns;
	int maxiters;
	double sample_percent;
	int use_number;
	int turbo;

	// Current
	float* U;
	float* v;
	float* w;
	int* use_index;

	// Init struct to constant eta
	GRASTA(int m_in, int p_in, int d_in, float eta_in, float rho_in, int maxiterns_in, int maxiters_in, double sample_percent_in) {

		// Dimensions
		m = m_in;
		p = p_in;
		d = d_in;
		n = m * p;
		max_d = d_in;

		// Basic
		eta_temp = eta_in;
		eta = eta_in;
		rho = rho_in;
		maxiterns = maxiterns_in;
		maxiters = maxiters_in;
		sample_percent = sample_percent_in;
		use_number = 0;
		turbo = 0;

		// Current
		U = (float*)malloc(n * d * sizeof(float));
		for (int ii = 0; ii < n * d; ++ii) {
			U[ii] = rand();
		}
		w = (float*)malloc(d * sizeof(float));
		use_index = (int*)malloc(n * sizeof(int));
	}

	// Destructor
	~GRASTA() {
		free(use_index);
		free(w);
		free(U);
	}

	// Create mask by sampling rows of U with sample_percent chance
	void createMask() {
		double rm = sample_percent * ((double)RAND_MAX);
		use_number = 0;
		for (int ii = 0; ii < n; ++ii) {
			if (rand() < rm) {
				use_index[use_number++] = ii;
			}
		}
	}

	// Soft-Thresholding operation
	void shrink(float* junk, float* s, float rho, int N) {
		int ii;
		float t = 0;
		for (ii = 0; ii < N; ++ii) {
			t = junk[ii];
			s[ii] = (t - rho * ((t > 0) - (t < 0))) * (fabs(t) > rho);
		}
	}

	// Do QR decomposition of U and set U = Q
	void getQfromU() {

		// Extra variables
		float  twork = 0;
		int lwork = -1;
		int info;

		// Variables that need to be deallocated
		float* tau = (float*)malloc(n * sizeof(float));

		// A very complicated way of making U orthogonal.

		sgeqrf(&n, &d, U, &n, tau, &twork, &lwork, &info);
		lwork = (int)twork;
		float* work = (float*)malloc(lwork * sizeof(float)); // Also needs to be deallocated
		sgeqrf(&n, &d, U, &n, tau, work, &lwork, &info);

		// Multiply out the reflectors (U currently contains a mess which was returned by sgeqrf)
		sorgqr(&n, &d, &d, U, &n, tau, work, &lwork, &info);

		// Avoid leaks
		free(work);
		free(tau);
	}

	// Returns transpose of pseudoinverse of smallU in pismallU
	void pinv_qr(float* smallU, float* pismallU) {

		// Extra variables
		int ii, jj;
		float  twork = 0;
		int lwork = -1;
		int info;
		float one = 1.0f;
		float zero = 0.0f;

		// Variables that need to be deallocated
		float* tempU = (float*)malloc(use_number * d * sizeof(float));
		float* tau = (float*)malloc(use_number * sizeof(float));
		float* R = (float*)calloc(d * d, sizeof(float));
		float* tri = (float*)calloc(d * d, sizeof(float));

		// Making copy of smallU 
		for (ii = 0; ii < use_number * d; ++ii) {
			tempU[ii] = smallU[ii];
		}

		// A very complicated way of getting Q and R from tempU
		sgeqrf(&use_number, &d, tempU, &use_number, tau, &twork, &lwork, &info);
		lwork = (int)twork;
		float* work = (float*)malloc(lwork * sizeof(float)); // Also needs to be deallocated
		sgeqrf(&use_number, &d, tempU, &use_number, tau, work, &lwork, &info);

		// Getting R
		for (ii = 0; ii < d; ++ii) {
			for (jj = 0; jj < (ii + 1); ++jj) {
				R[ii * d + jj] = tempU[ii * use_number + jj];
			}
		}

		// Setting tempU = Q
		sorgqr(&use_number, &d, &d, tempU, &use_number, tau, work, &lwork, &info);

		// Setting R = R^-1
		strtri("U", "N", &d, R, &d, &info);

		// Setting tri = (R^-1)^T
		for (ii = 0; ii < d; ++ii) {
			for (jj = 0; jj < (ii + 1); ++jj) {
				tri[jj * d + ii] = R[ii * d + jj];
			}
		}

		// Setting pismallU to transpose of pseudoinverse and destroying tempU
		sgemm("N", "N", &use_number, &d, &d, &one, tempU, &use_number, tri, &d, &zero, pismallU, &use_number);

		// Avoid leaks
		free(work);
		free(tri);
		free(R);
		free(tau);
		free(tempU);
	}

	// Performs w,y,s update without sampling
	void larb_no_sampling(float* s, float* y) {
		
		// Extra variables
		int ii, jj;
		float one = 1.0f;
		int oneinc = 1;
		float zero = 0.0f;
		float irho = 1 / rho;

		// Variables that need to be deallocated
		float* junk = (float*)malloc(n * sizeof(float));
		// Loop to repeat update
		for (ii = 0; ii < maxiterns; ++ii) {

			// 3.3

			// Setting junk = rho * (v - s) - y
			for (jj = 0; jj < n; ++jj) {
				junk[jj] = rho * (v[jj] - s[jj]) - y[jj];
			}

			// Just multiplying by U^T. We don't need (U'U)^-1 because we assume orthogonal U
			// Setting w = 1/rho * U^T * junk
			sgemv("T", &n, &d, &irho, U, &n, junk, &oneinc, &zero, w, &oneinc);


			// 3.4

			// Setting junk (Uw) = U * w
			sgemv("N", &n, &d, &one, U, &n, w, &oneinc, &zero, junk, &oneinc);

			// Setting s = v - junk (Uw) - y
			for (jj = 0; jj < n; ++jj) {
				s[jj] = v[jj] - junk[jj] - y[jj];
			}

			// Soft threshold to get our s
			shrink(s, s, 1 / (1 + rho), n);

			// 3.5

			// Setting y = y + rho * (s + junk (Uw) - v)
			for (jj = 0; jj < n; ++jj) {
				y[jj] += rho * (s[jj] + junk[jj] - v[jj]);
			}
		}

		// Avoid leaks
		free(junk);
	}

	// Performs (w,y,s), U, and eta update without sampling
	void grasta_step_no_sampling() {

		// Extra variables
		float one = 1.0f;
		float mone = -one;
		int oneinc = 1;
		float zero = 0.0f;
		int ii, jj;
		float sigma;
		float normg;
		float normw;
		float cs;
		float ss;

		// Variables that need to be deallocated
		float* s = (float*)calloc(n, sizeof(float));
		float* y = (float*)calloc(n, sizeof(float));
		float* g2 = (float*)malloc(d * sizeof(float));

		// Update w,s,y and compute Uw = U * w
		larb_no_sampling(s, y);

		// Setting y = y + rho * s
		for (jj = 0; jj < n; ++jj) {
			y[jj] += rho * s[jj];
		}

		// Setting s = U * w
		sgemv("N", &n, &d, &one, U, &n, w, &oneinc, &zero, s, &oneinc);

		// 3.8
		
		// Setting y (g1) = y + rho * (Uw + s - v)
		for (jj = 0; jj < n; ++jj) {
			y[jj] += rho * (s[jj] - v[jj]);
		}

		// 3.9

		// Setting g2 = U^T * y (g1)
		sgemv("T", &n, &d, &one, U, &n, y, &oneinc, &zero, g2, &oneinc);

		// 3.10

		// Setting y (-g) = U * g2 - y (g1)
		sgemv("N", &n, &d, &one, U, &n, g2, &oneinc, &mone, y, &oneinc);

		// Calculate sigma (between 3.11 and 3.12)
		normg = sdot(&n, y, &oneinc, y, &oneinc);
		normg = sqrt(normg);

		normw = sdot(&d, w, &oneinc, w, &oneinc);
		normw = sqrt(normw);

		sigma = normg * normw;

		// Update U

		// Calculate coefficients
		cs = (cos(eta * sigma) - 1) / (normw * normw);
		ss = sin(eta * sigma) / sigma;

		// Setting s (temp) = cs * s (Uw) + ss * y (-g)
		for (jj = 0; jj < n; ++jj) {
			s[jj] = cs * s[jj] + ss * y[jj];
		}

		// U = U + s (temp) * w^T
		for (jj = 0; jj < d; ++jj) {// column
			for (ii = 0; ii < n; ++ii) {// row
				U[jj * n + ii] += s[ii] * w[jj];
			}
		}

		// Avoid leaks
		free(g2);
		free(y);
		free(s);
	}

	// Performs w,y,s update with sampling and performs 3.8
	void larb_sampling(float* pismallU, float* smallU,float* smallv) {
		
		// Extra variables
		int ii, jj;
		float one = 1.0f;
		int oneinc = 1;
		float zero = 0.0f;
		float irho = 1 / rho;

		// Variables that need to be deallocated
		float* s = (float*)calloc(use_number, sizeof(float));
		float* y = (float*)calloc(use_number, sizeof(float));
		float* junk = (float*)malloc(use_number * sizeof(float));

		// Loop to repeat update
		for (ii = 0; ii < maxiterns; ++ii) {

			// 3.3

			// Setting junk = rho * (smallv - s) - y
			for (jj = 0; jj < use_number; ++jj) {
				junk[jj] = rho * (smallv[jj] - s[jj]) - y[jj];
			}

			// Setting w = pismallU * junk
			sgemv("T", &use_number, &d, &irho, pismallU, &use_number, junk, &oneinc, &zero, w, &oneinc);


			// 3.4

			// Setting junk (uw) = smallU * w
			sgemv("N", &use_number, &d, &one, smallU, &use_number, w, &oneinc, &zero, junk, &oneinc);

			// Setting s (temp) = smallv - junk (uw) - y
			for (jj = 0; jj < use_number; ++jj) {
				s[jj] = smallv[jj] - junk[jj] - y[jj];
			}

			// Soft threshold to get our s
			shrink(s, s, 1 / (1 + rho), use_number);

			// 3.5

			// Setting y = y + rho * (s + junk (uw) - smallv)
			for (jj = 0; jj < use_number; ++jj) {
				y[jj] += rho * (s[jj] + junk[jj] - smallv[jj]);
			}
		}

		// 3.8

		// Setting smallv (temp) = y + rho * (s - smallv)
		for (jj = 0; jj < use_number; ++jj) {
			smallv[jj] = y[jj] + rho * (s[jj] - smallv[jj]);
		}

		// Setting smallv (g1) = y + rho * (uw + s - smallv)
		sgemv("N", &use_number, &d, &rho, smallU, &use_number, w, &oneinc, &one, smallv, &oneinc);

		// Avoid leaks
		free(junk);
		free(y);
		free(s);
	}

	// Performs (w,s,y), U, and eta update with sampling
	void grasta_step_sampling() {

		// Extra variables
		float one = 1.0f;
		int oneinc = 1;
		float zero = 0.0f;
		int ii, jj;
		float sigma;
		float normg;
		float normw;
		float cs;
		float ss;
		float scale = 0;

		// Variables that are kept the same
		float* Uw = (float*)malloc(n * sizeof(float));
		float* g2 = (float*)malloc(d * sizeof(float));

		// Creating mask
		createMask();

		// Variables changed or added by sampling
		float* smallv = (float*)malloc(use_number * sizeof(float));
		float* smallU = (float*)malloc(use_number * d * sizeof(float));
		float* pismallU = (float*)malloc(use_number * d * sizeof(float));

		// Sampling v and normilizing to mean of 1
		for (ii = 0; ii < use_number; ++ii) {
			smallv[ii] = v[use_index[ii]];
			scale += fabs(smallv[ii]);
		}

		scale /= (float) use_number;

		for (ii = 0; ii < use_number; ++ii) {
			smallv[ii] /= scale;
		}

		// Sampling U
		for (jj = 0; jj < d; ++jj) {
			for (ii = 0; ii < use_number; ++ii) {
				smallU[jj * use_number + ii] = U[jj * n + use_index[ii]];
			}
		}
		
		// Setting pismallU to pseudo-inverse
		pinv_qr(smallU, pismallU);

		// Update w,y,s and performs 3.8
		larb_sampling(pismallU, smallU, smallv);

		// 3.9

		// Setting g2 = smallU^T * smallv (g1)
		sgemv("T", &use_number, &d, &one, smallU, &use_number, smallv, &oneinc, &zero, g2, &oneinc);

		// 3.10

		// Setting Uw (Ug2) = U * g2
		sgemv("N", &n, &d, &one, U, &n, g2, &oneinc, &zero, Uw, &oneinc);

		// Setting Uw (-g) = Ug2 - Mask_smallv (g1)
		for (jj = 0; jj < use_number; ++jj) {
			Uw[use_index[jj]] -= smallv[jj];
		}

		// Calculate sigma (between 3.11 and 3.12)
		normg = sdot(&n, Uw, &oneinc, Uw, &oneinc);
		normg = sqrt(normg);

		normw = sdot(&d, w, &oneinc, w, &oneinc);
		normw = sqrt(normw);

		sigma = normg * normw;
		
		// Update U

		// Calculate coefficients
		cs = (cos(eta * sigma) - 1) / (normw * normw);
		ss = sin(eta * sigma) / sigma;

		// Setting Uw = cs * U * w - ss * Uw (g)
		sgemv("N", &n, &d, &cs, U, &n, w, &oneinc, &ss, Uw, &oneinc);

		// Scaling w
		for (ii = 0; ii < d; ++ii) {
			w[ii] *= scale;
		}

		// U = U + Uw (temp) * w^T
		for (jj = 0; jj < d; ++jj) {// column
			for (ii = 0; ii < n; ++ii) {// row
				U[jj * n + ii] += Uw[ii] * w[jj];
			}
		}

		// Avoid leaks
		free(g2);
		free(Uw);
		free(pismallU);
		free(smallU);
		free(smallv);
	}

	// Transforms frame from color scale to gray scale
	void toGray(Mat* frame) {

		// Divide by 255
		frame->convertTo(*frame, CV_32F, 0.0039, 0);

		// Merge to 1 channel
		cvtColor(*frame, *frame, CV_BGR2GRAY);

		// Adjust size
		resize(*frame, *frame, Size(m, p), 0, 0, INTER_AREA);

		// Set v to processed frame
		v = frame->ptr<float>(0);
	}

	// Separates v into fg and bg
	void getBGFG(float* fg, float* bg) {

		// Extra variables
		float one = 1.0f;
		int oneinc = 1;
		float zero = 0.0f;

		// Setting bg = U * w
		sgemv("N", &n, &d, &one, U, &n, w, &oneinc, &zero, bg, &oneinc);

		// Setting fg = v - bg
		for (int ii = 0; ii < n; ++ii) {
			fg[ii] = v[ii] - bg[ii];
		}

		// Update turbo
		uptL0(fg);
	}

	// Update turbo with L0 loss
	void uptL0(float* fg) {

		double L0_norm = 0;
		for (int ii = 0; ii < n; ++ii) {
			if (fabs(fg[ii]) > .05) {
				++L0_norm;
			}
		}

		if (L0_norm > n * .6) {
			turbo = 0;
		}
		else {
			++turbo;
		}
	}

	// Performs adaptive GRASTA step
	void GRASTA_step() {
		if (turbo < 50) {
			eta = 5*eta_temp;
			grasta_step_no_sampling();
			eta = eta_temp;
		}
		else {
			grasta_step_sampling();
		}
	}

	// Create upper rectangular mask 
	void rectMask() {
		use_number = 0;
		for (int ii = 0; ii < n/3; ++ii) {
			use_index[use_number++] = ii;
		}
	}

	// Create mask that inmidiately selects changed pixels
	void quickMask(float* fg) {
		use_number = 0;
		for (int ii = 0; ii < 100; ++ii) {
			use_index[use_number++] = ii;
		}
		for (int ii = 100; ii < n; ii++) {
			if (fabs(fg[ii]) > 0.20) {
				use_index[use_number++] = ii;
			}
		}
	}
};
