#include <opencv2/opencv.hpp>

cv::Mat blendLighten(const cv::Mat& image1, const cv::Mat& image2) {
    // Make sure the input images have the same size
    CV_Assert(image1.size() == image2.size());
    
    cv::Mat blendedImage;
    cv::max(image1, image2, blendedImage); // Select maximum pixel value for each pixel
    
    return blendedImage;
}

// Creates comet-style star trails
cv::Mat blendAlpha(const cv::Mat& image1, const cv::Mat& image2, double alpha = 0.5) {
    // Create output image of the same size as the input images
    cv::Mat blendedImage(image1.size(), image1.type());
    
    // Perform alpha blending
    cv::addWeighted(image1, alpha, image2, 1.0 - alpha, 0.0, blendedImage);
    
    return blendedImage;
}

cv::Mat stackModeMedian(const std::vector<cv::Mat>& images) {
    // Check if the input vector of images is empty
    if (images.empty()) {
        std::cout << "No images provided." << std::endl;
        return cv::Mat();
    }
    
    // Ensure that all images have the same size and type
    for (const cv::Mat& image : images) {
        if (image.size() != images[0].size() || image.type() != images[0].type()) {
            std::cout << "All images must have the same size and type." << std::endl;
            return cv::Mat();
        }
    }
    
    // Create output image of the same size and type as the input images
    cv::Mat stackedImage(images[0].size(), images[0].type());
    
    // Iterate over each pixel and compute the median
    for (int y = 0; y < images[0].rows; ++y) {
        for (int x = 0; x < images[0].cols; ++x) {
            std::vector<uchar> pixels;
            
            // Collect pixel values from each image in the stack
            for (const cv::Mat& image : images) {
                pixels.push_back(image.at<uchar>(y, x));
            }
            
            // Sort the pixels
            std::sort(pixels.begin(), pixels.end());
            
            // Get the median pixel value
            uchar medianPixel = pixels[pixels.size() / 2];
            
            // Set the median pixel value in the output image
            stackedImage.at<uchar>(y, x) = medianPixel;
        }
    }
    
    return stackedImage;
}

std::vector<cv::Mat> convertToCVMat(const std::vector<std::string>& filepaths) {
    std::vector<cv::Mat> images;
    
    for (const std::string& filepath : filepaths) {
        cv::Mat image = cv::imread(filepath);
        
        // Check if the image was loaded successfully
        if (image.empty()) {
            std::cout << "Failed to load image: " << filepath << std::endl;
            continue;
        }
        
        images.push_back(image);
    }
    
    return images;
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
    
    std::vector<cv::Mat> images = convertToCVMat(imagePaths);
    
    cv::Mat resultImage;
    
    resultImage = stackModeMedian(images);
    
//  // Load the first image as the initial result
//  cv::Mat currentImage = cv::imread(imagePaths[0]);
//  
//  // Iterate over the remaining images
//  for (size_t i = 1; i < imagePaths.size(); ++i) {
//      cv::Mat nextImage = cv::imread(imagePaths[i]);
//      currentImage = blendLighten(currentImage, nextImage);
//      std::cout << "Blended images " << i << " and " << i+1 << std::endl;
//  }
//  
//  resultImage = currentImage;

    cv::imshow("Blended Image", resultImage);
    cv::waitKey(0);
    
    // Save the result image to a file
    std::string outputImagePath = "output_image.jpg";
    cv::imwrite(outputImagePath, resultImage);
    std::cout << "Blended image saved to: " << outputImagePath << std::endl;
    

    return 0;
}
