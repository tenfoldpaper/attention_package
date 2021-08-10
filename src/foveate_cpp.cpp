#include "ros/ros.h"
#include "std_msgs/String.h"
#include "attention_package/FoveatedImageCombined.h"
#include "attention_package/FoveatedImageMeta.h"
#include "attention_package/Tuple.h"

#include "sensor_msgs/CompressedImage.h"
#include "yolov5_detector/DetectionMsg.h"

#include <vector>
#include <iostream>
#include <tuple>
#include <numeric>
#include <chrono>
#include <ctime>

#include <opencv2/opencv.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cv_bridge/cv_bridge.h>

class FoveationClass{

private:
    ros::Subscriber detectionSubscriber;
    ros::Publisher foveationPublisher;
    ros::Publisher rgbFoveationPublisher;
    int fovlevel = 3;
    int maxscale = 5;
    bool saveImg;
    bool showImg;
    bool saveRgb;
    cv::Mat scale; // This value is constant for the given maxscale and fovlevel.
    cv::Mat xywh2ogsz(const cv::Mat& x){ // Pass by reference if NULL is not acceptable
        cv::Mat y = cv::Mat(1, 4, CV_32FC1);
        y.at<float>(0, 0) = x.at<float>(0, 1) - (x.at<float>(0, 3) / 2);
        y.at<float>(0, 1) = x.at<float>(0, 0) - (x.at<float>(0, 2) / 2);
        y.at<float>(0, 2) = (x.at<float>(0, 1) + (x.at<float>(0, 3) / 2)) - y.at<float>(0, 0);
        y.at<float>(0, 3) = (x.at<float>(0, 0) + (x.at<float>(0, 2) / 2)) - y.at<float>(0, 1);
        
        return y;
    }

    std::tuple<int, int, cv::Mat> identifyCenterDepthRange(const cv::Mat& img, const cv::Mat& bb_origin, const cv::Mat& bb_size){
        // Image of shape height, width
        // bb origin and size are both 2x1 vectors
        int height = img.rows;
        int width = img.cols;

        int range_height;
        int range_width;

        if(bb_origin.at<int>(0, 0) + bb_size.at<int>(0, 0) < height){
            range_height = bb_origin.at<int>(0, 0) + bb_size.at<int>(0, 0);
        }
        else{
            range_height = height - 1;
        }
        
        if(bb_origin.at<int>(0, 1) + bb_size.at<int>(0, 1) < width){
            range_width = bb_origin.at<int>(0, 1) + bb_size.at<int>(0, 1);
        }
        else{
            range_width = width - 1;
        }
        
        cv::Mat cropped_img = img(cv::Range(bb_origin.at<int>(0, 0), range_height), cv::Range(bb_origin.at<int>(0, 1), range_width));
        int histSize = 20;
        int channels[] = {1}; // Only depth image
        double min, max;
        cv::minMaxLoc(cropped_img, &min, &max);
        float grayRange[] = {0, (float) max + 1};
        const float * ranges[] = {grayRange};
        cv::MatND hist;
        cv::calcHist(&cropped_img, 1, 0, cv::Mat(), hist, 1, &histSize, ranges, true, false);

        // Now we want to get the top 2 max values
        float top2 = 0;
        int top2ind = -1;

        double histMin, histMax;
        cv::Point histMinLoc, histMaxLoc;
        cv::minMaxLoc(hist, &histMin, &histMax, &histMinLoc, &histMaxLoc);

        for(int i = 0; i < histSize; i++){
            float curr_value = hist.at<float>(0, i);
            if(curr_value >= top2){
                if(i != histMaxLoc.y){
                    top2ind = i;
                    top2 = curr_value;
                }
            }
        }

        // Get the list of bin edge values for further down the processing
        cv::Mat histEdges = cv::Mat(1, histSize+1, CV_32F);
        float binSize = (float) max / histSize;
        for(int i = 0; i <= histSize+1; i++){
            histEdges.at<float>(0, i) = i * binSize;
            if(i == histSize+1){
                histEdges.at<float>(0, i) = (float) max;
            }
        }

        int range_ind_low, range_ind_high;
        if(histMaxLoc.y < top2ind){
            range_ind_low = histMaxLoc.y;
            range_ind_high = top2ind;
        }
        else{
            range_ind_high = histMaxLoc.y;
            range_ind_low = top2ind;
        }
        
        return {range_ind_low, range_ind_high, histEdges};
    }

    std::tuple<cv::Mat, cv::Mat> calculateFovlevelBb(const cv::Mat& bb_origin, const cv::Mat& bb_size, int img_width, int img_height, int fovlevel){

        cv::Mat lower_bound = bb_origin.clone();
        cv::Mat upper_bound = cv::Mat(1, 2, CV_32SC1);

        if(bb_origin.at<int>(0, 0) + bb_size.at<int>(0, 0) < img_height){
            upper_bound.at<int>(0, 0) = bb_origin.at<int>(0, 0) + bb_size.at<int>(0, 0);
        }
        else{
            upper_bound.at<int>(0, 0) = img_height - 1;
        }
        
        if(bb_origin.at<int>(0, 1) + bb_size.at<int>(0, 1) < img_width){
            upper_bound.at<int>(0, 1) = bb_origin.at<int>(0, 1) + bb_size.at<int>(0, 1);
        }
        else{
            upper_bound.at<int>(0, 1) = img_width - 1;
        }

        cv::Mat lower_bounds(fovlevel, 2, CV_32SC1);
        cv::Mat upper_bounds(fovlevel, 2, CV_32SC1);
        
        float lb0 = (float) lower_bound.at<int>(0, 0);
        float lb1 = (float) lower_bound.at<int>(0, 1);

        float ub0 = (float) upper_bound.at<int>(0, 0);
        float ub1 = (float) upper_bound.at<int>(0, 1);
        
        float _fovlevel = (float) fovlevel;
        cv::Mat fovlevel_bb(fovlevel, 2, 2);
        
        for(float i = 0; i < fovlevel; i++){

            float lb_scale = (float) (_fovlevel - 1 - i)/(_fovlevel - 1);
            float ub_scale = (float) (i / (_fovlevel-1));
            
            lower_bounds.at<int>(i, 0) = (int) lb0 * lb_scale;
            lower_bounds.at<int>(i, 1) = (int) lb1 * lb_scale;
            upper_bounds.at<int>(i, 0) = (int) ub0 + ((img_height - ub0) * ub_scale);
            upper_bounds.at<int>(i, 1) = (int) ub1 + ((img_width - ub1) * ub_scale);
        }

        return {lower_bounds, upper_bounds};
        
    }

    cv::Mat linspace(float start, float end, int num){
        cv::Mat linVector = cv::Mat(1, num, CV_32F);
        float _num = (float) num;
        float diff = end - start;
        for(int i = 0; i < num; i++){
            linVector.at<float>(0, i) = start + ((i / (_num-1)) * diff);
        }
        return linVector;
    }

    auto calculateFovlevelDepth(int center_low, int center_high, cv::Mat& bins, int fovlevel){

        float _fovlevel = (float) fovlevel;
        float _center_low = (float) center_low;
        float _center_high = (float) center_high;

        cv::Mat fovlevel_depth(fovlevel, 2, CV_32SC1);

        for(float i = 0; i < _fovlevel; i++){

            float lb_scale = (float) (_fovlevel - 1 - i)/(_fovlevel - 1);
            float ub_scale = (float) (i / (_fovlevel-1));
            
            fovlevel_depth.at<int>(i, 0) = (int) _center_low * lb_scale;
            fovlevel_depth.at<int>(i, 1) = (int) _center_high + (ub_scale * ((float) bins.cols - 1 - _center_high));
            
        }
        return fovlevel_depth;
    }

    std::string type2str(int type) {
        std::string r;

        uchar depth = type & CV_MAT_DEPTH_MASK;
        uchar chans = 1 + (type >> CV_CN_SHIFT);

        switch ( depth ) {
            case CV_8U:  r = "8U"; break;
            case CV_8S:  r = "8S"; break;
            case CV_16U: r = "16U"; break;
            case CV_16S: r = "16S"; break;
            case CV_32S: r = "32S"; break;
            case CV_32F: r = "32F"; break;
            case CV_64F: r = "64F"; break;
            default:     r = "User"; break;
        }

        r += "C";
        r += (chans+'0');

        return r;
    }

public:
    FoveationClass( ros::NodeHandle *nh,
                    std::string detectionTopic,
                    std::string publishTopic, 
                    std::string rgbPublishTopic, 
                    std::string savePath,
                    int fovlevel_,
                    int maxscale_,
                    bool saveImg_, 
                    bool saveRgb_, 
                    bool showImg_)
    {
        detectionSubscriber = nh->subscribe(detectionTopic, 100, &FoveationClass::foveationCallback, this);
        foveationPublisher = nh->advertise<attention_package::FoveatedImageMeta>(publishTopic, 10);
        rgbFoveationPublisher = nh->advertise<attention_package::FoveatedImageMeta>(rgbPublishTopic, 10);
        fovlevel = fovlevel_;
        maxscale = maxscale_;
        saveImg = saveImg_;
        saveRgb = saveRgb_;
        showImg = showImg_;
        scale = linspace(1, (float) maxscale, fovlevel);
        std::cout << scale << std::endl;
    }
    void foveationCallback(const yolov5_detector::DetectionMsg::ConstPtr& data){ // const pointer avoids making a copy of the incoming data.

        auto start = std::chrono::system_clock::now();
        cv::Mat recvImg = cv_bridge::toCvCopy(data->depth_image, "mono16")->image;
        cv::Mat recvRgbImg = cv_bridge::toCvCopy(data->rgb_image, "bgr8")->image;
        #define MB 1024*1024

        attention_package::FoveatedImageMeta finalFovMsg = attention_package::FoveatedImageMeta();
        finalFovMsg.height = recvImg.size().height;
        finalFovMsg.width = recvImg.size().width;
        finalFovMsg.detected_objects = data->detection_count;
        finalFovMsg.rgb_image = data->rgb_image;
        finalFovMsg.foveation_level = fovlevel;
        float _imgHeight = (float) recvImg.size().height;
        float _imgWidth = (float) recvImg.size().width;
        
        std::vector<std::tuple<cv::Mat, cv::Mat>> _fovlevel_bbs = std::vector<std::tuple<cv::Mat, cv::Mat>>();
        std::vector<cv::Mat> _bb_origins = std::vector<cv::Mat>();
        std::vector<cv::Mat> _bb_sizes = std::vector<cv::Mat>();
        std::vector<cv::Mat> _fovlevel_depths = std::vector<cv::Mat>();
        std::vector<int> _center_lows = std::vector<int>();
        std::vector<int> _center_highs = std::vector<int>();
        std::vector<int> _det_classes = std::vector<int>();
        std::vector<cv::Mat> _bins = std::vector<cv::Mat>();
        std::vector<cv::Mat> _scales = std::vector<cv::Mat>();

        for(int i = 0; i < data->detection_count; i++){
            
            _det_classes.push_back(data->detection_array[i].detection_info[0]);
            cv::Mat bb_raw(1, 4, CV_32FC1);
            for(int j = 0; j < 4; j++){
                bb_raw.at<float>(0, j) = data->detection_array[i].detection_info[j + 1];
            }
            cv::Mat bb_ogsz = xywh2ogsz(bb_raw); // this is still in ratio 
            cv::Mat bb_origin(1, 2, CV_32SC1);
            cv::Mat bb_size(1, 2, CV_32SC1);
            bb_origin.at<int>(0, 0) = (int) (_imgHeight * bb_ogsz.at<float>(0, 0));
            bb_origin.at<int>(0, 1) = (int) (_imgWidth * bb_ogsz.at<float>(0, 1));
            bb_size.at<int>(0, 0) = (int) (_imgHeight * bb_ogsz.at<float>(0, 2));
            bb_size.at<int>(0, 1) = (int) (_imgWidth * bb_ogsz.at<float>(0, 3));
            
            int center_low, center_high;
            cv::Mat bin;
            std::tie(center_low, center_high, bin) = identifyCenterDepthRange(recvImg, bb_origin, bb_size);
            
            cv::Mat lower_bounds, upper_bounds;
            std::tie(lower_bounds, upper_bounds) = calculateFovlevelBb(bb_origin, bb_size, recvImg.size().width, recvImg.size().height, fovlevel);
            std::tuple<cv::Mat, cv::Mat> fovlevel_bb = {lower_bounds, upper_bounds};
            cv::Mat fovlevel_depth = calculateFovlevelDepth(center_low, center_high, bin, fovlevel);


            _bins.push_back(bin);
            _bb_origins.push_back(bb_origin);
            _bb_sizes.push_back(bb_size);
            _center_lows.push_back(center_low);
            _center_highs.push_back(center_high);
            _fovlevel_depths.push_back(fovlevel_depth);
            _fovlevel_bbs.push_back(fovlevel_bb);

        }

        for(int f = 0; f < fovlevel; f++){
            //cv::Mat imgMask = cv::Mat(recvImg.size(), CV_16UC1, cv::Scalar(65535));
            // copyto wants depth 8UC
            cv::Mat imgMask = cv::Mat(recvImg.size(), CV_8U, cv::Scalar(255));
            cv::Mat imgSend = cv::Mat::zeros(recvImg.size(), CV_16U);
            attention_package::FoveatedImageCombined fovMsg = attention_package::FoveatedImageCombined();
            for(int i = 0; i < finalFovMsg.detected_objects; i++){
                cv::Mat bin = _bins[i];
                cv::Mat bb_origin = std::get<0>(_fovlevel_bbs[i]).row(f);
                cv::Mat bb_end = std::get<1>(_fovlevel_bbs[i]).row(f);
                cv::Mat fovlevel_depth = _fovlevel_depths[i].row(f);
                int depth_lower = fovlevel_depth.at<int>(0, 0);
                int depth_upper = fovlevel_depth.at<int>(0, 1);

                attention_package::Tuple bbo_tpl = attention_package::Tuple();
                attention_package::Tuple bbe_tpl = attention_package::Tuple();
                
                for(int v = 0; v < 2; v++){
                    bbo_tpl.tpl.push_back(bb_origin.at<int>(0, v));
                    bbe_tpl.tpl.push_back(bb_end.at<int>(0, v));
                }

                fovMsg.bounding_box_origins.push_back(bbo_tpl);
                fovMsg.bounding_box_ends.push_back(bbe_tpl);

                cv::Range cropRows(bb_origin.at<int>(0, 0), bb_end.at<int>(0, 0));
                cv::Range cropCols(bb_origin.at<int>(0, 1), bb_end.at<int>(0, 1));
                
                cv::Mat croppedImg = recvImg(cropRows, cropCols);
                cv::Mat depthCroppedImg = cv::Mat(croppedImg > (int) bin.at<float>(0, depth_lower) & croppedImg < (int) bin.at<float>(0, depth_upper));

                double min, max;
                cv::minMaxLoc(croppedImg, &min, &max);
                cv::minMaxLoc(depthCroppedImg, &min, &max);
                imgSend(cropRows, cropCols) = depthCroppedImg;

                cv::Mat croppedMask = cv::Mat(depthCroppedImg == 0);
                //cv::Mat croppedMask16U;
                
                //croppedMask.convertTo(croppedMask16U, CV_16UC1, 257);
                croppedMask.copyTo(imgMask(cv::Rect(bb_origin.at<int>(0, 1), bb_origin.at<int>(0, 0), croppedMask.cols, croppedMask.rows)));

            }
            recvImg.copyTo(imgSend, imgMask);

            std::cout << "Img scaled size int" << std::endl;
            std::cout << (int) (imgSend.size().width / scale.at<float>(0, f)) << std::endl;
            std::cout << (int) (imgSend.size().height / scale.at<float>(0, f)) << std::endl;
            
            cv::resize(imgSend, imgSend, cv::Size(), (double)(1/scale.at<float>(0, f)), (double)(1/scale.at<float>(0, f)), cv::INTER_NEAREST);

            std::cout << imgSend.size() << std::endl;

            std::vector<uchar> buffer;
            buffer.resize(2 * MB);
            fovMsg.foveated_image.header.stamp = ros::Time::now();
            fovMsg.foveated_image.format = "tiff";
            // encode to buffer then assign to data
            cv::imencode(".tiff", imgSend, buffer);
            fovMsg.foveated_image.data = buffer;
            finalFovMsg.foveated_images_groups.push_back(fovMsg);
            
            if(showImg){
                //visualisation part of the code
                //double tmin, tmax;
                
                //cv::minMaxLoc(imgSend, &tmin, &tmax);
                //std::cout << tmin << " " << tmax << std::endl;
                cv::Mat imgSend8U = imgSend / 255;
                //cv::minMaxLoc(imgSend8U, &tmin, &tmax);
                //std::cout << tmin << " " << tmax << std::endl;
                imgSend8U.convertTo(imgSend8U, CV_8U);
                //cv::minMaxLoc(imgSend8U, &tmin, &tmax);
                //std::cout << tmin << " " << tmax << std::endl;
                cv::Mat colorMapImg;
                cv::applyColorMap(imgSend8U, colorMapImg, cv::COLORMAP_JET);
                cv::imshow("imgSendColor", colorMapImg);
                cv::imshow("img Mask", imgMask);
                cv::imshow("imgSendRaw", imgSend);

                cv::waitKey();
            }
            
        }

        try{
            foveationPublisher.publish(finalFovMsg);
        }
        catch(...){
            ROS_ERROR("Failed to publish to the topic!");
        }
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        std::time_t end_time = std::chrono::system_clock::to_time_t(end);

        std::cout << "finished computation at " << std::ctime(&end_time)
                << "elapsed time: " << elapsed_seconds.count() << "s\n";
    }
};

int main(int argc, char **argv){

    ros::init(argc, argv, "foveate_cpp");

    ros::NodeHandle n("~");

    ros::Publisher chatter_pub = n.advertise<std_msgs::String>("chatter", 1000);

    std::string publishTopic, rgbPublishTopic, detectionTopic, savePath;
    int fovlevel;
    int maxscale;
    bool saveImg, saveRgb, showImg;

    n.getParam("publish_topic", publishTopic);
    n.getParam("rgb_publish_topic", rgbPublishTopic);
    n.getParam("detection_topic", detectionTopic);
    n.getParam("save_path", savePath);

    n.getParam("fov_level", fovlevel);
    n.getParam("max_scale", maxscale);
    n.getParam("save_img", saveImg);
    n.getParam("save_rgb", saveRgb);
    n.getParam("show_img", showImg);
    
    ROS_INFO("Started foveation node with the following parameters.");
    ROS_INFO("publish_topic: %s", publishTopic.c_str());
    ROS_INFO("rgb_publish_topic: %s", rgbPublishTopic.c_str());
    ROS_INFO("detection_topic: %s", detectionTopic.c_str());
    ROS_INFO("save_path: %s", savePath.c_str());
    ROS_INFO("fov_level: %i", fovlevel);
    ROS_INFO("max_scale: %i", maxscale);
    ROS_INFO("save_img: %d", saveImg);
    ROS_INFO("save_rgb: %d", saveRgb);
    ROS_INFO("show_img: %d", showImg);

    FoveationClass fc = FoveationClass(&n, detectionTopic, publishTopic, rgbPublishTopic, savePath, fovlevel, maxscale, saveImg, saveRgb, showImg);

    ros::spin();

    return 0;
}