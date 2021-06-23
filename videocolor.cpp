#include "GRASTAC.h"

struct BGR {
	float blue;
	float green;
	float red;
};

void pr(float* aea, int ne) {
	for (int i = 0; i < ne; ++i) {
		cout << aea[i] << '\n';
	}
	cout << '\n';
}

int main(int argc, char* argv[]) {

	// Init rand seed
	srand(time(0));

	// Dimensions
	int m = 375;
	int p = 480;
	int d = 9;

	// Preparing parameters for Stat init
	float eta = 0.000001;
	float rho = 1;
	int maxiterns = 20;
	int maxiters = 40;
	double sample_percent = 0.1;

	// Extra variables
	float one = 1.0f;
	int oneinc = 1;
	float zero = 0.0f;

	// Stat init
	GRASTA G(m, p, d, eta, rho, maxiterns, maxiters, sample_percent);

	// Vectors to help with OpenCV interface
	float* UwB = (float*)malloc(G.n * sizeof(float));
	float* UwG = (float*)malloc(G.n * sizeof(float));
	float* UwR = (float*)malloc(G.n * sizeof(float));

	// Get Q from the QR decomposition of random U
	G.getQfromU();

	// Create display windows
	namedWindow("capture", 1);
	namedWindow("background", 1);
	namedWindow("foreground", 1);

	// Open camera with error msg
	VideoCapture capture(0);
	if (!capture.isOpened()) {
		cout << "Camera did not open\n";
		return 0;
	}

	// Get frame from camera with error msg
	Mat* frame = new Mat();
	capture >> *frame;
	if (frame->empty()) {
		cout << "Frame not captured\n";
		return 0;
	}

	// Needed to display
	int c;

	// Main loop
	while (1) {

		// Get new frame with error msg
		capture >> *frame;
		if (frame->empty()) {
			cout << "Frame not captured\n";
			break;
		}

		// Resize frame
		resize(*frame, *frame, Size(m, p), 0, 0, INTER_AREA);

		// Display frame
		imshow("capture", *frame);
		
		// Change type
		frame->convertTo(*frame, CV_32FC3);

		// Split frame into channels and link to G
		vector<Mat> bgr(3);
		split(*frame, bgr);
		G.vB = bgr[2].ptr<float>(0);
		G.vG = bgr[1].ptr<float>(0);
		G.vR = bgr[0].ptr<float>(0);

		// Perform GRASTA step
		G.GRASTA_step();

		// Setting Uw = U * w for all channels
		sgemv("N", &G.n, &G.d, &one, G.UB, &G.n, G.wB, &oneinc, &zero, UwB, &oneinc);
		sgemv("N", &G.n, &G.d, &one, G.UG, &G.n, G.wG, &oneinc, &zero, UwG, &oneinc);
		sgemv("N", &G.n, &G.d, &one, G.UR, &G.n, G.wR, &oneinc, &zero, UwR, &oneinc);

		// Merge all Uw into bg
		Mat* bg = new Mat(p, m, CV_32FC3);
		for (int i = 0; i < p; ++i) {
			for (int j = 0; j < m; ++j) {
				BGR& pix = bg->ptr<BGR>(i)[j];
				pix.blue = UwB[i * m + j];
				pix.green = UwG[i * m + j];
				pix.red = UwR[i * m + j];
			}
		}

		// Calculate fg
		Mat* fg = new Mat(p, m, CV_32FC3);

		for (int i = 0; i < p; ++i) {
			for (int j = 0; j < m; ++j) {

				// References
				BGR& rgb1 = frame->ptr<BGR>(i)[j];
				BGR& rgb2 = bg->ptr<BGR>(i)[j];
				BGR& rgb3 = fg->ptr<BGR>(i)[j];

				// Setting bgm to a scaled version of frame
				rgb3.blue = rgb1.blue - rgb2.blue;
				rgb3.green = rgb1.green - rgb2.green;
				rgb3.red = rgb1.red - rgb2.red;

			}
		}

		// Normalize to 0-255
		normalize(*fg, *fg, 255, 0, CV_MINMAX);

		// Convert back to unchar
		bg->convertTo(*bg, CV_8UC3);
		fg->convertTo(*fg, CV_8UC3);

		// Display background
		imshow("background", *bg);

		// Display foreground
		imshow("foreground", *fg);

		// Needed to display
		c = cvWaitKey(10);

		// Interactive keys
		if ((char)c == 27) // ESC
			break;
		switch ((char)c)
		{
		case 'q': // Increase sample percent after an iter
			G.sample_percent = G.sample_percent + .05;
			printf("Sample percent up to %.8f \n", G.sample_percent);
			break;
		case 'w': // Decrease sample percent after an iter
			G.sample_percent = G.sample_percent - .05;
			if (G.sample_percent <= 0) {
				G.sample_percent = 0.01;
			}
			printf("Sample percent down to %.8f \n", G.sample_percent);
			break;
		case 'a': // Increase learning rate after an iter
			G.eta = 3 * G.eta / 2;
			printf("Eta up to %.8f \n", G.eta);
			break;
		case 's': // Decrease learning rate after an iter
			G.eta = 2 * G.eta / 3;
			printf("Eta down to %.8f\n", G.eta);
			break;
		case 'z': // Increase rank d
			G.d = G.d + 1;
			if (G.d > G.max_d) {
				G.d = G.max_d;
			}
			printf("d up to %d\n", G.d);
			break;
		case 'x': // Increase rank d
			G.d = G.d - 1;
			if (G.d == 0) {
				G.d = 1;
			}
			printf("d down to %d\n", G.d);
			break;
		default:
			;
		}

		// Avoid leaks
		delete fg;
		delete bg;
	}

	// Avoid leaks
	delete frame;
}