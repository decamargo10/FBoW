/**

The MIT License

Copyright (c) 2017 Rafael Muñoz-Salinas

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

*/

#include "cmd_line_parser.h"
#include "dir_reader.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#ifdef USE_CONTRIB
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#endif

std::vector<std::string> createVideoList(std::string name_file){
    std::vector<std::string> video_paths;
    try
    {
        std::ifstream file(name_file);
        std::string video_path; 
        while (std::getline(file, video_path))
        {
         video_paths.push_back(video_path);
        }
        return video_paths;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return video_paths;
    }
}

std::vector<cv::Mat> loadFeaturesFromVideo(std::vector<std::string> video_file_paths, const std::string& descriptor = "orb", int every_nth_frame=3) {
    cv::Ptr<cv::Feature2D> feat_detector;
    if (descriptor == "orb") feat_detector = cv::ORB::create(2000);
    else if (descriptor == "brisk") feat_detector = cv::BRISK::create();
    else if (descriptor == "akaze") feat_detector = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, 1e-4);
#ifdef USE_CONTRIB
    else if (descriptor == "surf") feat_detector = cv::xfeatures2d::SURF::create(15, 4, 2);
#endif
    else throw std::runtime_error("invalid descriptor: " + descriptor);

    assert(!descriptor.empty());
    std::vector<cv::Mat> features;

    std::cout << "extracting features ..." << std::endl;
    cv::VideoCapture cap;
    int i = 0;
    cv::Mat frame;
    int frames_since_last_detected = 1;
    
    for (std::string video_file_path : video_file_paths) {
        cap.open(video_file_path);
        if(!cap.isOpened()){
            std::cout << "Error opening video stream or file" << std::endl;
            return features;
        }
        int total_frames= cap.get(cv::CAP_PROP_FRAME_COUNT);
        i = 0;
        while(1){
            std::vector<cv::KeyPoint> keypoints;
            cv::Mat descriptors;
            cap >> frame;
            if (frame.empty()) {
                std::cerr << "could not open frame: #" << i << ". Aborting." << std::endl;
                break;
            }
            // if we pick every frame we create a lot of output features but many of them will contain the same information
            // the clustering might group a lot of them together but they still will have a negative impact
            // as this is rather bad for a robust and general vocabulary we should jump ahead a few frames every time
            if(frames_since_last_detected==every_nth_frame){
                std::cout << "reading frame: " << i <<  "|" << total_frames/every_nth_frame << "-> " << (float(i)/float(total_frames/every_nth_frame))*100 << "%" << std::endl;
                frames_since_last_detected = 1;
                feat_detector->detectAndCompute(frame, cv::Mat(), keypoints, descriptors);
                std::cout << "extracted features: total = " << keypoints.size() << std::endl;
                features.push_back(descriptors);
                i++;
            }else{
                frames_since_last_detected++;
            }
            
        }
    }
    cap.release();
    std::cout << "done detecting features" << std::endl;
    return features;
}

std::vector<cv::Mat> loadFeatures(const std::vector<std::string>& path_to_images, const std::string& descriptor = "orb") {
    cv::Ptr<cv::Feature2D> feat_detector;
    if (descriptor == "orb") feat_detector = cv::ORB::create(2000);
    else if (descriptor == "brisk") feat_detector = cv::BRISK::create();
    else if (descriptor == "akaze") feat_detector = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, 1e-4);
#ifdef USE_CONTRIB
    else if (descriptor == "surf") feat_detector = cv::xfeatures2d::SURF::create(15, 4, 2);
#endif
    else throw std::runtime_error("invalid descriptor: " + descriptor);

    assert(!descriptor.empty());
    std::vector<cv::Mat> features;

    std::cout << "extracting features ..." << std::endl;
    for (const auto& path_to_image : path_to_images) {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        std::cout << "reading image: " << path_to_image << std::endl;
        cv::Mat image = cv::imread(path_to_image, 0);
        if (image.empty()) {
            std::cerr << "could not open image: " << path_to_image << std::endl;
            continue;
        }
        feat_detector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
        std::cout << "extracted features: total = " << keypoints.size() << std::endl;
        features.push_back(descriptors);
        std::cout << "done detecting features" << std::endl;
    }
    return features;
}

void saveToFile(const std::string& filename, const std::vector<cv::Mat>& features, std::string desc_name, bool rewrite = true) {
    if (!rewrite) {
        std::fstream ifile(filename, std::ios::binary);
        if (ifile.is_open()) {
            std::runtime_error("output file " + filename + " already exists");
        }
    }
    std::ofstream ofile(filename, std::ios::binary);
    if (!ofile.is_open()) {
        std::cerr << "could not create an output file: " << filename << std::endl;
        exit(1);
    }

    char _desc_name[20];
    desc_name.resize(std::min(size_t(19), desc_name.size()));
    strcpy(_desc_name, desc_name.c_str());
    ofile.write(_desc_name, 20);

    uint32_t size = features.size();
    ofile.write((char*) &size, sizeof(size));
    for (const auto& f : features) {
        if (!f.isContinuous()) {
            std::cerr << "matrices should be continuous" << std::endl;
            exit(0);
        }
        uint32_t aux = f.cols;
        ofile.write((char*)&aux, sizeof(aux));
        aux = f.rows;
        ofile.write((char*)&aux, sizeof(aux));
        aux = f.type();
        ofile.write((char*)&aux, sizeof(aux));
        ofile.write((char*)f.ptr<uchar>(0), f.total() * f.elemSize());
    }
}


int main(int argc, char** argv) {
    try {
        CmdLineParser cml(argc, argv);
        if (cml["-h"] || argc < 4) {
            std::cerr << "Usage: DESCRIPTOR_NAME (= orb, brisk, akaze, surf(contrib)) FEATURE_OUTPUT IMAGES_DIR USING_VIDEO_BOOLEAN(TRUE,FALSE)-> USE VIDEO_NAME AS THIRD ARGUMENT EVERY_NTH_FRAME" << std::endl;
            std::cerr << std::endl;
            std::cerr << "First step of creating a vocabulary is extracting features from a set of images." << std::endl;
            std::cerr << "We save them to a file for next step." << std::endl;
            std::cerr << std::endl;
            return EXIT_FAILURE;
        }

        const std::string descriptor = argv[1];
        const std::string output = argv[2];

        std::vector<cv::Mat> features;
        if(argc < 2){
            auto images = DirReader::read(argv[3]);
            features = loadFeatures(images, descriptor);
            std::cout << "HERE" << std::endl;
        }else if (argv[4] == std::string("TRUE")){
            std::cout << "ELSEWHERE: " << argv[3] << std::endl;
            std::vector<std::string> video_path_list =createVideoList(argv[3]);
            int nth_frame = atoi(argv[5]);
            features = loadFeaturesFromVideo(video_path_list, descriptor, nth_frame);
            std::cout << features[0] << std::endl;
        }

        std::cout << "saving the features: " << argv[2] << std::endl;

        saveToFile(argv[2], features, descriptor);
    } catch (std::exception& ex) {
        std::cerr << ex.what() << std::endl;
    }

    return EXIT_SUCCESS;
}

// time ./fbow_dump_features orb "/home/decamargo/Documents/output_features" "/home/decamargo/Documents/video_list" TRUE 2

// ./fbow_dump_features orb "/home/decamargo/Documents/features_ice_breaker" "/home/decamargo/Documents/video_list_ev" TRUE 2

