#include <opencv2/opencv.hpp>
#include <sys/ipc.h>
#include <sys/shm.h>
#include "shared_memory.hpp"
#include "image_processor.hpp"


int main() {
    const int KEY = 12345;
    const int WIDTH = 2592;
    const int HEIGHT = 1944;
    const int SIZE = WIDTH * HEIGHT * 1.0; // 1.0 - GREY8 byter per pixel

    auto shmem = std::make_share<consumer::SharedMemory>(KEY, SIZE);

    auto imgProcessor = std::make_shared<consumer::ImageProcessor>(shmem);

    imgProcessor->ProcessStream();

    cv::Mat K = (cv::Mat_<double>(3, 3) << 7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02,
                                            0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02,
                                            0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00);

    imgProcessor->MakeVisualOdometry(K);


    return 0;
}
