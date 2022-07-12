#include "../extern/pybind11/include/pybind11/pybind11.h"
#include "../extern/pybind11/include/pybind11/stl.h"
#include <opencv2/core.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "../utils/fbow_dump_features.cpp"
#include "../include/fbow/bow_vector.h"
#include "../include/fbow/fbow.h"
#include "../include/fbow/vocabulary.h"
#include "../src/vocabulary_creator.cpp"
#include <fstream>

//PYBIND11_MAKE_OPAQUE(std::map<uint32_t, fbow::_float>);

namespace py = pybind11;

fbow::BoWVector createBoW(std::string path_to_image_folder, std::string path_to_vocabulary_file){
    std::vector<cv::Mat> features;
    std::vector<std::string> strings;
    strings.push_back(path_to_image_folder);
    features = loadFeatures(strings, "orb");
    fbow::Vocabulary bow_vocab_ = fbow::Vocabulary();
    bow_vocab_.readFromFile(path_to_vocabulary_file);
    fbow::BoWVector bow_vec = bow_vocab_.transform(features.front());
    std::ofstream BoW_file("BoW_values.txt");
    for (auto& t : bow_vec){
        BoW_file << t.first << " " << t.second << "\n";
    }
    BoW_file.close();
    return bow_vec;

}

int detectLoops(std::vector<std::vector<int>> tests, std::string feature_path, std::string voc_path, int norm, bool use_tf){
    std::vector<cv::Mat> features;
    features = fbow::readFeaturesFromFile(feature_path, "orb");
    fbow::Vocabulary bow_vocab_ = fbow::Vocabulary();
    bow_vocab_.readFromFile(voc_path);
    int detections = 0;
    for(auto t : tests){
        std::vector<double> results;
        std::vector<fbow::BoWVector> bows;
        for(int j : t){
            bows.push_back(bow_vocab_.transform(features.at(j), norm, use_tf));
        }
        for(fbow::BoWVector c : bows){
            results.push_back(bows.at(0).score(bows.at(0), c));
        }
        bool success = true;
        for(int i=2; i<results.size(); i++){
            if(results[i]>=results[1]){
                success = false;
                break;
            }
        }
        if(success){
            detections++;
        }
    }
    return detections;
}

PYBIND11_MODULE(fbow_pybind, m) {
    m.doc() = "FBoW bind module"; 
    m.def("createBoW", &createBoW, "Function that takes two strings (path to image, path to vocabulary) and returns a BoW Vector of that image.");
    m.def("detect_loops", &detectLoops, "Function which detects loops"); 

    py::class_<std::map<uint32_t, fbow::_float>>(m, "IntFloatMap")
    .def(py::init<>());
    //.def("begin", &std::map<uint32_t, fbow::_float>::first);

    py::class_<fbow::BoWVector, std::map<uint32_t, fbow::_float>>(m, "BoWVector")
        .def(py::init<>())
        .def("toStream", &fbow::BoWVector::toStream)
        .def("fromStream", &fbow::BoWVector::fromStream)
        .def("hash", &fbow::BoWVector::hash)
        .def("score", &fbow::BoWVector::score)
        .def("hash", &fbow::BoWVector::hash);
        //.def("first", &fbow::BoWVector::first);

        /*
        void toStream(std::ostream& str) const;
        void fromStream(std::istream& str);
        //returns a hash identifying this
        uint64_t hash() const;
        //returns the similitude score between to image descriptors using L2 norm
        static double score(const BoWVector& v1, const BoWVector& v2);
        */
} 