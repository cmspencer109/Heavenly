#include <iostream>
#include <opencv2/opencv.hpp>


using namespace std;


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
    
    // Iterate over each pixel and compute the median for each color channel
    for (int y = 0; y < images[0].rows; ++y) {
        for (int x = 0; x < images[0].cols; ++x) {
            std::vector<cv::Vec3b> pixels;
            
            // Collect pixel values from each image in the stack
            for (const cv::Mat& image : images) {
                pixels.push_back(image.at<cv::Vec3b>(y, x));
            }
            
            // Sort the pixels for each color channel
            std::sort(pixels.begin(), pixels.end(),
                [](const cv::Vec3b& a, const cv::Vec3b& b) {
                    return a[0] < b[0];  // Sort based on blue channel intensity
                });
            
            // Get the median pixel value for each color channel
            cv::Vec3b medianPixel;
            medianPixel[0] = pixels[pixels.size() / 2][0];  // Blue channel
            medianPixel[1] = pixels[pixels.size() / 2][1];  // Green channel
            medianPixel[2] = pixels[pixels.size() / 2][2];  // Red channel
            
            // Set the median pixel value in the output image for each color channel
            stackedImage.at<cv::Vec3b>(y, x) = medianPixel;
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

std::vector<cv::Mat> resizeImages(const std::vector<cv::Mat>& images, double scaleFactor) {
    std::vector<cv::Mat> resizedImages;
    
    for (const cv::Mat& image : images) {
        cv::Size newSize(image.cols * scaleFactor, image.rows * scaleFactor);
        cv::Mat resizedImage;
        cv::resize(image, resizedImage, newSize);
        resizedImages.push_back(resizedImage);
    }
    
    return resizedImages;
}


cv::Mat differenceMask(const cv::Mat& image1, const cv::Mat& image2) {
    // Check if the input images have the same size
    if (image1.size() != image2.size()) {
        std::cout << "Images must have the same size." << std::endl;
        return cv::Mat();
    }
    
    // Convert the input images to grayscale
    cv::Mat image1Gray, image2Gray;
    cv::cvtColor(image1, image1Gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(image2, image2Gray, cv::COLOR_BGR2GRAY);
    
    // Compute the absolute difference between the grayscale images
    cv::Mat diff;
    cv::absdiff(image1Gray, image2Gray, diff);
    
    // Apply a threshold to obtain a binary mask
    // https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
    cv::Mat mask;
    cv::threshold(diff, mask, 10, 255, cv::THRESH_BINARY);
    
    // Apply Gaussian blur
//  cv::GaussianBlur(mask, mask, cv::Size(0, 0), 2);
    
//  cv::threshold(mask, mask, 1, 255, cv::THRESH_BINARY);
    
//  cv::erode(mask, mask, cv::Mat(), cv::Point(-1, -1), 5);
    
    return mask;
}


cv::Mat starTrails(const std::vector<cv::Mat>& images) {
    // Load the first image as the initial result
    cv::Mat stackedImage = images[0];
    
    // Iterate over the remaining images
    for (size_t i = 1; i < images.size(); ++i) {
        cv::Mat nextImage = images[i];
        stackedImage = blendLighten(stackedImage, nextImage);
        std::cout << "Blended images " << i << " and " << i+1 << std::endl;
    }
    
    return stackedImage;
}

cv::Mat contentAwareFill(const cv::Mat& stackedImage, const cv::Mat& mask, int radius)
{
    cv::Mat filledImage;
    cv::inpaint(stackedImage, mask, filledImage, radius, cv::INPAINT_TELEA);
    return filledImage;
}


std::vector<cv::Mat> maskImages(const std::vector<cv::Mat>& images, const cv::Mat& mask) {
    // Check if the input vector of images is empty
    if (images.empty()) {
        std::cout << "No images provided." << std::endl;
        return {};
    }
    
    // Check if the mask image is empty or not in grayscale
//  if (mask.empty() || mask.channels() != 1) {
//      std::cout << "Invalid mask image." << std::endl;
//      return {};
//  }
    
    // Ensure that all images and the mask have the same size
    cv::Size maskSize = mask.size();
    for (const cv::Mat& image : images) {
        if (image.size() != maskSize) {
            std::cout << "All images must have the same size as the mask." << std::endl;
            return {};
        }
    }
    
    // Convert the mask to binary (threshold it)
    cv::Mat maskBinary;
    cv::threshold(mask, maskBinary, 1, 255, cv::THRESH_BINARY);
    
    // Apply the mask to each image
    std::vector<cv::Mat> maskedImages;
    for (const cv::Mat& image : images) {
        cv::Mat maskedImage;
        image.copyTo(maskedImage, maskBinary);
        maskedImages.push_back(maskedImage);
    }
    
    return maskedImages;
}


cv::Mat makePanorama(const std::vector<cv::Mat>& images) {
    cv::Mat pano;
    cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create(cv::Stitcher::PANORAMA);
    cv::Stitcher::Status status = stitcher->stitch(images, pano);
    
    if (status != cv::Stitcher::OK)
    {
        cout << "Can't stitch images, error code = " << int(status) << endl;
//      return EXIT_FAILURE;
    }
    
    return pano;
}



//int main() {
int main(int argc, char** argv) {
    /*
    cd build
    make && ./Heavenly
    */
    
//  std::vector<std::string> imagePaths = {
//      "../images/_MG_8206.jpg",
//      "../images/_MG_8207.jpg",
//      "../images/_MG_8208.jpg",
//      "../images/_MG_8209.jpg",
//      "../images/_MG_8210.jpg",
//      "../images/_MG_8211.jpg",
//      "../images/_MG_8212.jpg",
//      "../images/_MG_8213.jpg",
//      "../images/_MG_8214.jpg",
//      "../images/_MG_8215.jpg",
//  };
    
//  std::vector<cv::Mat> images = resizeImages(convertToCVMat(imagePaths), 0.5);
//  std::vector<cv::Mat> images = resizeImages(convertToCVMat(imagePaths), 1);
    
    
//  //  Create clean sky
//  cv::Mat cleanSkyImage = stackModeMedian(images);
//  
//  cv::imwrite("clean_sky.jpg", cleanSkyImage);
//  
//  cv::imshow("Blended Image", cleanSkyImage);
//  cv::waitKey(0);
//  
//  //  Create star trails
//  cv::Mat starrySkyImage = starTrails(images);
//  
//  cv::imwrite("starry_sky.jpg", starrySkyImage);
//  
//  cv::imshow("Blended Image", starrySkyImage);
//  cv::waitKey(0);
    
    // Create mask from differences
//  cv::Mat cleanSkyImage = cv::imread("../clean_sky_small.jpg");
//  cv::Mat starrySkyImage = cv::imread("../starry_sky_small.jpg");
//  cv::Mat mask = differenceMask(cleanSkyImage, starrySkyImage);
    
//  cv::imwrite("mask.jpg", mask);
    
//  cv::imshow("Changes Mask", mask);
//  cv::waitKey(0);
//  
    // Fill star trail gaps
//  cv::Mat filledImage = contentAwareFill(starrySkyImage, mask, 3);
//  
//  cv::imwrite("../output.jpg", filledImage);
//  
//  cv::imshow("Star trails", cv::imread("../starry_sky_small.jpg"));
//  cv::waitKey(0);
//  
//  cv::imshow("Filled trails", filledImage);
//  cv::waitKey(0);
    
    
//  ########### LSD DETECTION ###############
//  cv::Mat input = cv::imread("../starry_sky_small.jpg");
//  cv::Mat input = cv::imread("../test.jpg");
//  cv::Mat gray;
//  cv::cvtColor(input, gray, cv::COLOR_BGRA2GRAY);
//  
//  cv::Ptr<cv::LineSegmentDetector> det;
//  det = cv::createLineSegmentDetector();
//  
//  cv::Mat lines;
//  det->detect(gray, lines);
//  
//  det->drawSegments(input, lines);
//  
//  cv::imshow("input", input);
//  cv::waitKey(0);
    
    
    
    
//  cv::Mat inputImage = cv::imread("../test.jpg");
//  cv::Mat inputImageGray;
//  cv::cvtColor(inputImage, inputImageGray, cv::COLOR_BGR2GRAY);
//  
//  cv::imshow("", inputImageGray);
//  cv::waitKey(0);
//  
//  cv::Mat edges;
//  cv::Canny(inputImageGray, edges, 150, 200, 3);
//  
//  int minLineLength = 30;
//  int maxLineGap = 5;
//  
//  std::vector<cv::Vec4i> lines;
//  cv::HoughLinesP(edges, lines, CV_PI / 180, 30, minLineLength, maxLineGap);
//  
//  for (size_t i = 0; i < lines.size(); i++) {
//      cv::Vec4i line = lines[i];
//      cv::line(inputImage, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
//  }
//  
//  cv::putText(inputImage, "Tracks Detected", cv::Point(500, 250), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
//  
//  cv::imshow("result", inputImage);
//  cv::waitKey(0);
//  cv::imshow("edge", edges);
//  cv::waitKey(0);
//  
    
    
    std::vector<cv::Mat> images = {
        cv::imread("../images/milkyway/_MG_3534.jpg"),
        cv::imread("../images/milkyway/_MG_3535.jpg"),
        cv::imread("../images/milkyway/_MG_3536.jpg"),
        cv::imread("../images/milkyway/_MG_3537.jpg"),
        cv::imread("../images/milkyway/_MG_3538.jpg"),
        cv::imread("../images/milkyway/_MG_3539.jpg"),
        cv::imread("../images/milkyway/_MG_3540.jpg"),
        cv::imread("../images/milkyway/_MG_3541.jpg"),
        cv::imread("../images/milkyway/_MG_3542.jpg"),
        cv::imread("../images/milkyway/_MG_3543.jpg"),
    };
    
    std::vector<cv::Mat> masks = {
        cv::imread("../images/milkyway/mask.jpg"),
        cv::imread("../images/milkyway/mask.jpg"),
        cv::imread("../images/milkyway/mask.jpg"),
        cv::imread("../images/milkyway/mask.jpg"),
        cv::imread("../images/milkyway/mask.jpg"),
        cv::imread("../images/milkyway/mask.jpg"),
        cv::imread("../images/milkyway/mask.jpg"),
        cv::imread("../images/milkyway/mask.jpg"),
        cv::imread("../images/milkyway/mask.jpg"),
        cv::imread("../images/milkyway/mask.jpg"),
    };
    
    
    
    
//  std::vector<cv::Mat> p = {
//      cv::imread("../images/p/_MG_8739.jpg"),
//      cv::imread("../images/p/_MG_8740.jpg"),
//      cv::imread("../images/p/_MG_8741.jpg"),
//  };
    
//  cv::Mat pano = makePanorama(p);
//  
//  cv::imwrite("pano.jpg", pano);
//  
//  cv::imshow("", pano);
//  cv::waitKey(0);

    
    // Iterate over the aligned images and display each one
//  for (size_t i = 0; i < alignedImages.size(); ++i)
//  {
//      cv::imshow("Aligned Image " + std::to_string(i+1), alignedImages[i]);
//      cv::waitKey(0);
//  }
    
    
//  cv::Mat res = stackModeMedian(alignedImages);
//  
//  cv::imshow("", res);
//  cv::waitKey(0);
    
    // Save the result image to a file
//  std::string outputImagePath = "output_image.jpg";
//  cv::imwrite(outputImagePath, resultImage);
//  std::cout << "Blended image saved to: " << outputImagePath << std::endl;
    

    return 0;
}
