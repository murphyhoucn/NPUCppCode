// bgr to gray
// canny边缘检测

#include <opencv2/opencv.hpp>

int main()
{
    cv::Mat img_rgb, img_gry, img_cny;

    cv::namedWindow("Example Gray", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Example Canny", cv::WINDOW_AUTOSIZE);
    
    img_rgb = cv::imread("./opencv.png");

    cv::cvtColor(img_rgb, img_gry, cv::COLOR_BGR2GRAY);
    cv::imshow("Example Gray", img_gry);

    cv::Canny(img_gry, img_cny, 10, 100, 3, true);
    cv::imshow("Example Canny", img_cny);

    cv:: waitKey(0);

    return 0;
}