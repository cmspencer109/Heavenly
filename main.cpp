#include <opencv2/opencv.hpp>

cv::Mat blendLighten(const cv::Mat& image1, const cv::Mat& image2) {
    // Make sure the input images have the same size
    CV_Assert(image1.size() == image2.size());
    
    cv::Mat blendedImage;
    cv::max(image1, image2, blendedImage); // Select maximum pixel value for each pixel
    
    return blendedImage;
}

int main() {
    /*
    cd build
    make && ./Heavenly
    */
    
    std::vector<std::string> imagePaths = {
        "../images/_MG_8206.jpg",
        "../images/_MG_8207.jpg",
        "../images/_MG_8208.jpg",
        "../images/_MG_8209.jpg",
        "../images/_MG_8210.jpg",
        "../images/_MG_8211.jpg",
        "../images/_MG_8212.jpg",
        "../images/_MG_8213.jpg",
        "../images/_MG_8214.jpg",
        "../images/_MG_8215.jpg",
    };
    
    cv::Mat resultImage;
    
    // Load the first image as the initial result
    cv::Mat currentImage = cv::imread(imagePaths[0]);
    
    // Iterate over the remaining images
    for (size_t i = 1; i < imagePaths.size(); ++i) {
        cv::Mat nextImage = cv::imread(imagePaths[i]);
        currentImage = blendLighten(currentImage, nextImage);
        std::cout << "Blended images " << i << " and " << i+1 << std::endl;
    }
    
    resultImage = currentImage;

    cv::imshow("Blended Image", resultImage);
    cv::waitKey(0);
    
    // Save the result image to a file
    std::string outputImagePath = "output_image.jpg";
    cv::imwrite(outputImagePath, resultImage);
    std::cout << "Blended image saved to: " << outputImagePath << std::endl;
    

    return 0;
}
