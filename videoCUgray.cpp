#include "GRASTACUGray.h"

// Specify dimensions m, p, n, and d in GRASTA.h
int main(int argc, char* argv[]) {

	// Init rand seed
	srand(time(0));

	// Dimensions
	int m = 375;
	int p = 480;
	int d = 9;

	// Preparing parameters for Stat init
	float eta = 0.0000001;
	float rho = 1;
	int maxiterns = 20;
	int maxiters = 40;
	double sample_percent = 0.1;

	// Stat init
	GRASTA G(m, p, d, eta, rho, maxiterns, maxiters, sample_percent);

	// Vectors to help with OpenCV interface
	float* bg = (float*)malloc(G.n * sizeof(float));
	float* fg = (float*)malloc(G.n * sizeof(float));

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

		// Make copy if needed and transform to Gray scale
		G.toGray(frame);

		// Display frame
		imshow("capture", *frame);

		// Perform GRASTA step
		G.GRASTA_step();

		// Separate video and update turbo
		G.getBGFG(fg, bg);

		// Matrices to help with OpenCV interface
		Mat* bgm = new Mat(p, m, CV_32F, bg);
		Mat* fgm = new Mat(p, m, CV_32F, fg);

		// Display background
		imshow("background", *bgm);

		// Normalize to 0-1
		normalize(*fgm, *fgm, 1, 0, CV_MINMAX);

		// Display foreground
		imshow("foreground", *fgm);

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
		case 'r': // Reset bg
			G.turbo = 0;
			break;
		case 'f': // Fix bg
			G.turbo = 50;
			break;
		default:
			;
		}

		// Avoid leaks
		delete fgm;
		delete bgm;
	}

	// Avoid leaks
	delete frame;
	free(fg);
	free(bg);
}