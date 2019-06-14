//
// Created by zhenyaf on 6/13/19.
//

#ifndef SSLAM_CONFIG_HPP
#define SSLAM_CONFIG_HPP


#include <string>
#include <opencv2/core/persistence.hpp>
#include <iostream>

namespace sslam {

class Config {
public:
    Config(std::string const& yml_fname) {
        file_ = cv::FileStorage(yml_fname.c_str(), cv::FileStorage::READ);
    }

    bool IsValid() const {return file_.isOpened();}

    template <typename T>
    T get(std::string const& key) const {
        return T(file_[key]);
    }

private:
    cv::FileStorage file_;
};
}


#endif //SSLAM_CONFIG_HPP
