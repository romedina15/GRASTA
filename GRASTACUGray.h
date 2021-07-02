#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

// CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

// MKL
#include "mkl.h"
#include "mkl_blas.h"

// OpenCV
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/opencv.hpp>

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

	// Host
	float* U_h;
	float* v_h;
	float* w_h;
	int* use_index = 0;

	// CUDA
	cublasHandle_t handle;
	cublasStatus_t status;
	cusolverDnHandle_t handleDn;
	cusolverStatus_t statusDn;
	float* U = 0;
	float* v = 0;
	float* w = 0;

	// Init struct
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

		// Host
		U_h = (float*)malloc(n * d * sizeof(float));
		for (int ii = 0; ii < n * d; ++ii) {
			U_h[ii] = rand();
		}
		use_index = (int*)malloc(n * sizeof(int));

		// CUDA
		status = cublasCreate_v2(&handle);
		statusDn = cusolverDnCreate(&handleDn);
		cudaMalloc(reinterpret_cast<void**>(&U), n * d * sizeof(U[0]));
		cublasSetVector(n * d, sizeof(U[0]), U_h, 1, U, 1);
		cudaMalloc(reinterpret_cast<void**>(&v), n * sizeof(v[0]));
		cudaMalloc(reinterpret_cast<void**>(&w), d * sizeof(w[0]));

	}

	// Destructor
	~GRASTA() {
		cudaFree(v);
		cudaFree(U);
		cudaFree(w);
		status = cublasDestroy_v2(handle);
		statusDn = cusolverDnDestroy(handleDn);
		free(use_index);
		free(U_h);
	}

	//// Create mask by sampling rows of U with sample_percent chance
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
	void shrink(float* s, float rho, int N) {
		
		// Extra variable
		int oneinc = 1;
		float* temp = (float*)malloc(N * sizeof(float));

		// Copying to host
		cublasGetVector(N, sizeof(float), s, oneinc, temp, oneinc);

		// Soft Thresholding
		int ii;
		float t = 0;
		for (ii = 0; ii < N; ++ii) {
			t = temp[ii];
			temp[ii] = (t - rho * ((t > 0) - (t < 0))) * (fabs(t) > rho);
		}
		
		// Copying back to device
		cublasSetVector(N, sizeof(float), temp, oneinc, s, oneinc);

		// Avoid leaks
		free(temp);
	}

	// Do QR decomposition of U and set U = Q
	void getQfromU() {

		// Extra variables
		int oneinc = 1;
		size_t sf = sizeof(float);
		int GEwork = -1;
		int ORwork = -1;

		// Variables that need to be deallocated
		int* info = 0;
		float* tau = 0;
		cudaMalloc(reinterpret_cast<void**>(&tau), n * sf);
		cudaMalloc(reinterpret_cast<void**>(&info), sizeof(int));

		// A very complicated way of making U orthogonal.

		// Calculating memory needed
		cusolverDnSgeqrf_bufferSize(handleDn, n, d, U, n, &GEwork);
		cusolverDnSorgqr_bufferSize(handleDn, n, d, d, U, n, tau, &ORwork);
		int lwork = (GEwork > ORwork)? GEwork : ORwork;
		float* work = 0;
		cudaMalloc(reinterpret_cast<void**>(&work), lwork * sf);

		// Performing QR
		cusolverDnSgeqrf(handleDn, n, d, U, n, tau, work, lwork, info);
		
		// Multiply out the reflectors (U currently contains a mess which was returned by sgeqrf) to get Q
		cusolverDnSorgqr(handleDn, n, d, d, U, n, tau, work, lwork, info);

		// Copy back to host
		cublasGetVector(n * d, sizeof(float), U, oneinc, U_h, oneinc);
		
		// Avoid leaks
		cudaFree(info);
		cudaFree(work);
		cudaFree(tau);
	}

	// Returns transpose of pseudoinverse of smallU in pismallU
	void pinv_qr(float* smallU, float* pismallU) {

		// Extra variables
		int ii, jj;
		int oneinc = 1;
		int GEwork = -1;
		int ORwork = -1;
		float one = 1.0f;
		float zero = 0.0f;
		int infoh = 0;

		// Variables that need to be deallocated
		float* R_h = (float*)calloc(d * d, sizeof(float));
		float* tri_h = (float*)calloc(d * d, sizeof(float));
		float* tempU_h = (float*)malloc(use_number * d * sizeof(float));
		
		// Cuda
		int* info = 0;
		float* tempU = 0;
		float* tau = 0;
		float* R = 0;
		cudaMalloc(reinterpret_cast<void**>(&info), sizeof(int));
		cudaMalloc(reinterpret_cast<void**>(&tempU), use_number * d * sizeof(tempU[0]));
		cudaMalloc(reinterpret_cast<void**>(&tau), use_number * sizeof(tau[0]));
		cudaMalloc(reinterpret_cast<void**>(&R), d * d * sizeof(R[0]));
		cudaMemset(R, 0, d * d * sizeof(R[0]));

		// Making copy of smallU 
		cublasScopy_v2(handle, use_number * d, smallU, oneinc, tempU, oneinc);

		// A very complicated way of getting Q and R from tempU
		cusolverDnSgeqrf_bufferSize(handleDn, use_number, d, tempU, use_number, &GEwork);
		cusolverDnSorgqr_bufferSize(handleDn, use_number, d, d, tempU, use_number, tau, &ORwork);
		int lwork = (GEwork > ORwork) ? GEwork : ORwork;
		float* work = 0;
		cudaMalloc(reinterpret_cast<void**>(&work), lwork * sizeof(work[0]));

		// Performing QR and changing tempU
		cusolverDnSgeqrf(handleDn, use_number, d, tempU, use_number, tau, work, lwork, info);

		// Copying to host
		cublasGetVector(use_number * d, sizeof(float), tempU, oneinc, tempU_h, oneinc);

		// Getting R_h
		for (ii = 0; ii < d; ++ii) {
			for (jj = 0; jj < (ii + 1); ++jj) {
				R_h[ii * d + jj] = tempU_h[ii * use_number + jj];
			}
		}

		// Setting R_h = R_h^-1
		strtri("U", "N", &d, R_h, &d, &infoh);

		// Setting tri_h = (R_h^-1)^T
		for (ii = 0; ii < d; ++ii) {
			for (jj = 0; jj < (ii + 1); ++jj) {
				tri_h[jj * d + ii] = R_h[ii * d + jj];
			}
		}

		// Copying back to device
		cublasSetVector(d * d, sizeof(float), tri_h, oneinc, R, oneinc);

		// Setting tempU = Q
		cusolverDnSorgqr(handleDn, use_number, d, d, tempU, use_number, tau, work, lwork, info);

		// Setting pismallU to transpose of pseudoinverse and destroying tempU
		cublasOperation_t op = CUBLAS_OP_N;
		cublasSgemm_v2(handle, op, op, use_number, d, d, &one, tempU, use_number, R, d, &zero, pismallU, use_number);

		// Avoid leaks
		free(R_h);
		free(tri_h);
		free(tempU_h);
		cudaFree(info);
		cudaFree(work);
		cudaFree(R);
		cudaFree(tau);
		cudaFree(tempU);
	}

	// Performs w,y,s update without sampling
	void larb_no_sampling(float* s, float* y) {
		
		// Extra variables
		size_t sf = sizeof(float);
		cublasOperation_t op;
		int ii;
		float one = 1.0f;
		float mone = -one;
		int oneinc = 1;
		float zero = 0.0f;
		float irho = 1 / rho;
		float nrho = 1 / (1 + rho);

		// Variables that need to be deallocated
		float* junk = 0;
		cudaMalloc(reinterpret_cast<void**>(&junk), n * sf);
		float* temp = (float*)malloc(n * sf);

		// Loop to repeat update
		for (ii = 0; ii < maxiterns; ++ii) {

			// 3.3

			// Setting junk = rho * (v - s) - y
			cublasScopy_v2(handle, n, y, oneinc, junk, oneinc);
			cublasSscal_v2(handle, n, &mone, junk, oneinc);
			cublasSscal_v2(handle, n, &mone, s, oneinc);
			cublasSaxpy_v2(handle, n, &one, v, oneinc, s, oneinc);
			cublasSaxpy_v2(handle, n, &rho, s, oneinc, junk, oneinc);

			// Just multiplying by U^T. We don't need (U'U)^-1 because we assume orthogonal U
			// Setting w = 1/rho * U^T * junk
			op = CUBLAS_OP_T;
			cublasSgemv_v2(handle, op, n, d, &irho, U, n, junk, oneinc, &zero, w, oneinc);

			// 3.4

			// Setting junk (Uw) = U * w
			op = CUBLAS_OP_N;
			cublasSgemv_v2(handle, op, n, d, &one, U, n, w, oneinc, &zero, junk, oneinc);

			// Setting s = v - junk (Uw) - y
			cublasScopy_v2(handle, n, v, oneinc, s, oneinc);
			cublasSaxpy_v2(handle, n, &mone, y, oneinc, s, oneinc);
			cublasSaxpy_v2(handle, n, &mone, junk, oneinc, s, oneinc);

			// Soft threshold to get our s
			shrink(s, nrho, n);

			// 3.5

			// Setting y = y + rho * (s + junk (Uw) - v)
			cublasSaxpy_v2(handle, n, &one, s, oneinc, junk, oneinc);
			cublasSaxpy_v2(handle, n, &mone, v, oneinc, junk, oneinc);
			cublasSaxpy_v2(handle, n, &rho, junk, oneinc, y, oneinc);
		}

		// Avoid leaks
		cudaFree(junk);
		free(temp);

	}

	// Performs (w,y,s), U, and eta update without sampling
	void grasta_step_no_sampling() {

		// Extra variables
		size_t sf = sizeof(float);
		float one = 1.0f;
		float mone = -one;
		int oneinc = 1;
		float zero = 0.0f;
		float sigma;
		float normg;
		float normw;
		float cs;
		float ss;
		cublasOperation_t op;

		// Variables that need to be deallocated
		float* s = 0;
		float* y = 0;
		float* g2 = 0;
		cudaMalloc(reinterpret_cast<void**>(&s), n * sf);
		cudaMalloc(reinterpret_cast<void**>(&y), n * sf);
		cudaMalloc(reinterpret_cast<void**>(&g2), d * sf);
		cudaMemset(s, 0, n * sf);
		cudaMemset(y, 0, n * sf);

		// Update w,s,y and compute Uw = U * w
		larb_no_sampling(s, y);

		// Setting y = y + rho * s
		cublasSaxpy_v2(handle, n, &rho, s, oneinc, y, oneinc);

		// Setting s = U * w
		op = CUBLAS_OP_N;
		cublasSgemv_v2(handle, op, n, d, &one, U, n, w, oneinc, &zero, s, oneinc);

		// 3.8
		
		// Setting y (g1) = y + rho * (Uw + s - v)
		cublasSaxpy_v2(handle, n, &mone, v, oneinc, s, oneinc);
		cublasSaxpy_v2(handle, n, &rho, s, oneinc, y, oneinc);

		// 3.9

		// Setting g2 = U^T * y (g1)
		op = CUBLAS_OP_T;
		cublasSgemv_v2(handle, op, n, d, &one, U, n, y, oneinc, &zero, g2, oneinc);

		// 3.10

		// Setting y (-g) = U * g2 - y (g1)
		op = CUBLAS_OP_N;
		cublasSgemv_v2(handle, op, n, d, &one, U, n, g2, oneinc, &mone, y, oneinc);

		// Calculate sigma (between 3.11 and 3.12)
		cublasSnrm2_v2(handle, n, y, oneinc, &normg);
		normg = sqrt(normg);

		cublasSnrm2_v2(handle, d, w, oneinc, &normw);
		normw = sqrt(normw);

		sigma = normg * normw;

		// Update U

		// Calculate coefficients
		cs = (cos(eta * sigma) - 1) / (normw * normw);
		ss = sin(eta * sigma) / sigma;

		// Setting s (temp) = cs * s (Uw) + ss * y (-g)
		cublasSscal_v2(handle, n, &cs, s, oneinc);
		cublasSaxpy_v2(handle, n, &ss, y, oneinc, s, oneinc);

		// U = U + s (temp) * w^T
		cublasSger_v2(handle, n, d, &one, s, oneinc, w, oneinc, U, n);

		// Copy back to host
		cublasGetVector(n * d, sizeof(float), U, oneinc, U_h, oneinc);

		// Avoid leaks
		cudaFree(g2);
		cudaFree(y);
		cudaFree(s);
	}

	// Performs w,y,s update with sampling and performs 3.8
	void larb_sampling(float* pismallU, float* smallU,float* smallv) {
		
		// Extra variables
		int ii;
		float one = 1.0f;
		float mone = -one;
		int oneinc = 1;
		float zero = 0.0f;
		float irho = 1 / rho;
		float mrho = -rho;
		cublasOperation_t op;

		// Variables that need to be deallocated
		float* s = 0;
		float* y = 0;
		float* junk = 0;
		cudaMalloc(reinterpret_cast<void**>(&s), use_number * sizeof(s[0]));
		cudaMalloc(reinterpret_cast<void**>(&y), use_number * sizeof(y[0]));
		cudaMalloc(reinterpret_cast<void**>(&junk), use_number * sizeof(junk[0]));
		cudaMemset(s, 0, use_number * sizeof(s[0]));
		cudaMemset(y, 0, use_number * sizeof(y[0]));

		// Loop to repeat update
		for (ii = 0; ii < maxiterns; ++ii) {

			// 3.3

			// Setting junk = rho * (smallv - s) - y
			cublasScopy_v2(handle, use_number, y, oneinc, junk, oneinc);
			cublasSscal_v2(handle, use_number, &mone, junk, oneinc);
			cublasSaxpy_v2(handle, use_number, &rho, smallv, oneinc, junk, oneinc);
			cublasSaxpy_v2(handle, use_number, &mrho, s, oneinc, junk, oneinc);

			// Setting w = pismallU * junk
			op = CUBLAS_OP_T;
			cublasSgemv_v2(handle, op, use_number, d, &irho, pismallU, use_number, junk, oneinc, &zero, w, oneinc);

			// 3.4

			// Setting junk (uw) = smallU * w
			op = CUBLAS_OP_N;
			cublasSgemv_v2(handle, op, use_number, d, &one, smallU, use_number, w, oneinc, &zero, junk, oneinc);

			// Setting s (temp) = smallv - junk (uw) - y
			cublasScopy_v2(handle, use_number, smallv, oneinc, s, oneinc);
			cublasSaxpy_v2(handle, use_number, &mone, y, oneinc, s, oneinc);
			cublasSaxpy_v2(handle, use_number, &mone, junk, oneinc, s, oneinc);

			// Soft threshold to get our s
			shrink(s, 1 / (1 + rho), use_number);

			// 3.5

			// Setting y = y + rho * (s + junk (uw) - smallv)
			cublasSaxpy_v2(handle, use_number, &rho, s, oneinc, y, oneinc);
			cublasSaxpy_v2(handle, use_number, &rho, junk, oneinc, y, oneinc);
			cublasSaxpy_v2(handle, use_number, &mrho, smallv, oneinc, y, oneinc);
		}

		// 3.8

		// Setting smallv (temp) = y + rho * (s - smallv)
		cublasSaxpy_v2(handle, use_number, &mone, s, oneinc, smallv, oneinc);
		cublasSscal_v2(handle, use_number, &mrho, smallv, oneinc);
		cublasSaxpy_v2(handle, use_number, &one, y, oneinc, smallv, oneinc);

		// Setting smallv (g1) = y + rho * (uw + s - smallv)
		op = CUBLAS_OP_N;
		cublasSgemv_v2(handle, op, use_number, d, &rho, smallU, use_number, w, oneinc, &one, smallv, oneinc);

		// Avoid leaks
		cudaFree(junk);
		cudaFree(y);
		cudaFree(s);
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
		cublasOperation_t op;

		// Creating mask
		createMask();

		// Host variables
		float* Uw_h = (float*)malloc(n * sizeof(float));
		float* smallv_h = (float*)malloc(use_number * sizeof(float));
		float* smallU_h = (float*)malloc(use_number * d * sizeof(float));

		// Device variables
		float* Uw = 0;
		float* g2 = 0;
		float* smallv = 0;
		float* smallU = 0;
		float* pismallU = 0;
		cudaMalloc(reinterpret_cast<void**>(&Uw), n * sizeof(Uw[0]));
		cudaMalloc(reinterpret_cast<void**>(&g2), d * sizeof(g2[0]));
		cudaMalloc(reinterpret_cast<void**>(&smallv), use_number * sizeof(smallv[0]));
		cudaMalloc(reinterpret_cast<void**>(&smallU), use_number * d * sizeof(smallU[0]));
		cudaMalloc(reinterpret_cast<void**>(&pismallU), use_number * d * sizeof(pismallU[0]));

		// Sampling v and normilizing to mean of 1
		for (ii = 0; ii < use_number; ++ii) {
			smallv_h[ii] = v_h[use_index[ii]];
			scale += fabs(smallv_h[ii]);
		}

		scale /= (float) use_number;

		for (ii = 0; ii < use_number; ++ii) {
			smallv_h[ii] /= scale;
		}

		// Sampling U
		for (jj = 0; jj < d; ++jj) {
			for (ii = 0; ii < use_number; ++ii) {
				smallU_h[jj * use_number + ii] = U_h[jj * n + use_index[ii]];
			}
		}

		// Copying to device
		cublasSetVector(use_number * d, sizeof(float), smallU_h, oneinc, smallU, oneinc);
		cublasSetVector(use_number, sizeof(float), smallv_h, oneinc, smallv, oneinc);

		// Setting pismallU to pseudo-inverse
		pinv_qr(smallU, pismallU);

		// Update w,y,s and performs 3.8
		larb_sampling(pismallU, smallU, smallv);

		// 3.9

		// Setting g2 = smallU^T * smallv (g1)
		op = CUBLAS_OP_T;
		cublasSgemv_v2(handle, op, use_number, d, &one, smallU, use_number, smallv, oneinc, &zero, g2, oneinc);

		// 3.10

		// Setting Uw (Ug2) = U * g2
		op = CUBLAS_OP_N;
		cublasSgemv_v2(handle, op, n, d, &one, U, n, g2, oneinc, &zero, Uw, oneinc);

		// Copying to host
		cublasGetVector(use_number, sizeof(float), smallv, oneinc, smallv_h, oneinc);
		cublasGetVector(n, sizeof(float), Uw, oneinc, Uw_h, oneinc);

		// Setting Uw (-g) = Ug2 - Mask_smallv (g1)
		for (ii = 0; ii < use_number; ++ii) {
			Uw_h[use_index[ii]] -= smallv_h[ii];
		}

		// Copying back to device
		cublasSetVector(n, sizeof(float), Uw_h, oneinc, Uw, oneinc);

		// Calculate sigma (between 3.11 and 3.12)
		cublasSnrm2_v2(handle, n, Uw, oneinc, &normg);
		normg = sqrt(normg);

		cublasSnrm2_v2(handle, d, w, oneinc, &normw);
		normw = sqrt(normw);

		sigma = normg * normw;
		
		// Update U

		// Calculate coefficients
		cs = (cos(eta * sigma) - 1) / (normw * normw);
		ss = sin(eta * sigma) / sigma;

		// Setting Uw = cs * U * w - ss * Uw (g)
		op = CUBLAS_OP_N;
		cublasSgemv_v2(handle, op, n, d, &cs, U, n, w, oneinc, &ss, Uw, oneinc);

		// Scaling w
		cublasSscal_v2(handle, d, &scale, w, oneinc);

		// U = U + Uw (temp) * w^T
		cublasSger_v2(handle, n, d, &one, Uw, oneinc, w, oneinc, U, n);

		// Copy back to host
		cublasGetVector(n * d, sizeof(float), U, oneinc, U_h, oneinc);

		// Avoid leaks
		free(Uw_h);
		free(smallU_h);
		free(smallv_h);
		cudaFree(g2);
		cudaFree(Uw);
		cudaFree(pismallU);
		cudaFree(smallU);
		cudaFree(smallv);
	}

	// Transforms frame from color scale to gray scale
	void toGray(Mat* frame) {

		// Extra variables
		int oneinc = 1;

		// Divide by 255
		frame->convertTo(*frame, CV_32F, 0.0039, 0);

		// Merge to 1 channel
		cvtColor(*frame, *frame, CV_BGR2GRAY);

		// Adjust size
		resize(*frame, *frame, Size(m, p), 0, 0, INTER_AREA);

		// Set v_h to processed frame
		v_h = frame->ptr<float>(0);

		// Copy to device
		cublasSetVector(n, sizeof(float), v_h, oneinc, v, oneinc);
	}

	// Separates v into fg and bg
	void getBGFG(float* fg, float* bg) {

		// Extra variables
		float one = 1.0f;
		int oneinc = 1;
		float zero = 0.0f;

		// Copying to host
		float* bg_d = 0;
		cudaMalloc(reinterpret_cast<void**>(&bg_d), n * sizeof(bg_d[0]));

		// Setting bg = U * w
		cublasOperation_t op = CUBLAS_OP_N;
		cublasSgemv_v2(handle, op, n, d, &one, U, n, w, oneinc, &zero, bg_d, oneinc);

		// Copy to host
		cublasGetVector(n, sizeof(float), bg_d, oneinc, bg, oneinc);

		// Setting fg = v - bg
		for (int ii = 0; ii < n; ++ii) {
			fg[ii] = v_h[ii] - bg[ii];
		}

		// Update turbo
		uptL0(fg);

		// Avoid leaks
		cudaFree(bg_d);
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
			eta = 5 * eta_temp;
			grasta_step_no_sampling();
			eta = eta_temp;
		}
		else {
			grasta_step_sampling();
		}
	}

	// Create upper rectangular mask 
	//void rectMask() {
	//	use_number = 0;
	//	for (int ii = 0; ii < n/3; ++ii) {
	//		use_index[use_number++] = ii;
	//	}
	//}

	// Create mask that inmidiately selects changed pixels
	//void quickMask(float* fg) {
	//	use_number = 0;
	//	for (int ii = 0; ii < 100; ++ii) {
	//		use_index[use_number++] = ii;
	//	}
	//	for (int ii = 100; ii < n; ii++) {
	//		if (fabs(fg[ii]) > 0.20) {
	//			use_index[use_number++] = ii;
	//		}
	//	}
	//}
};
