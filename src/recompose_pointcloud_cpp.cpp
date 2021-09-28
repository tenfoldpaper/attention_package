#include "ros/ros.h"
#include "std_msgs/String.h"
#include "std_msgs/Header.h"
#include "attention_package/FoveatedImageCombined.h"
#include "attention_package/FoveatedImageMeta.h"
#include "attention_package/Tuple.h"

// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "sensor_msgs/CompressedImage.h"
#include "sensor_msgs/CameraInfo.h"
#include "sensor_msgs/point_cloud2_iterator.h"


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

class RecompositionClass{

private:
    ros::Subscriber foveationSubscriber;
    ros::Publisher pointcloudPublisher;
    sensor_msgs::CameraInfo::ConstPtr rgbCameraInfo;
    sensor_msgs::CameraInfo::ConstPtr depthCameraInfo;
    bool saveImg;
    bool showImg;
    bool savePath;
    double running_counter;
    double running_total;
    bool saveRgb;

public:
    RecompositionClass( ros::NodeHandle *nh,
                        std::string foveationTopic,
                        std::string publishTopic,
                        std::string rgbCameraTopic,
                        std::string depthCameraTopic,
                        std::string savePath,
                        bool saveImg_,
                        bool saveRgb_,
                        bool showImg_)
    {
        foveationSubscriber = nh->subscribe(foveationTopic, 1, &RecompositionClass::recompositionCallback, this); //freshest message
        pointcloudPublisher = nh->advertise<sensor_msgs::PointCloud2>(publishTopic, 1); //freshest message
        
        rgbCameraInfo = ros::topic::waitForMessage<sensor_msgs::CameraInfo>(rgbCameraTopic, *nh, ros::Duration(5));
        depthCameraInfo = ros::topic::waitForMessage<sensor_msgs::CameraInfo>(depthCameraTopic, *nh, ros::Duration(5));
        
        ROS_INFO("Received the camera infos");
        
        saveImg = saveImg_;
        saveRgb = saveRgb_;
        showImg = showImg_;
        running_counter = 0;
        running_total = 0;
    }

    void recompositionCallback(const attention_package::FoveatedImageMeta::ConstPtr& data){

        auto start = std::chrono::system_clock::now();
        
        //std::vector<float> T = {-0.0254, -0.0013, -0.00218};
        std::vector<float> T = {0, 0, 0}; // is the camera already registered?
        //std::vector<float> depthCameraInfoVec = {576.092756, 0.000000, 316.286974, 0.000000, 575.853472, 239.895662, 0.000000, 0.000000, 1.000000};
        float cx_d = depthCameraInfo->K[2];
        float cy_d = depthCameraInfo->K[5];
        float fx_d = depthCameraInfo->K[0];
        float fy_d = depthCameraInfo->K[4];
        float cx_r = rgbCameraInfo->K[2];
        float cy_r = rgbCameraInfo->K[5];
        float fx_r = rgbCameraInfo->K[0];
        float fy_r = rgbCameraInfo->K[4];
                
        int fovLevel = data->foveation_level;
        int width = data->width;
        int height = data->height;
        
        // initialize the message to be sent 
        sensor_msgs::PointCloud2 cloud;
        std_msgs::Header header;
        header.stamp = ros::Time::now();
        std::cout << header.stamp.nsec << std::endl;
        
        header.frame_id = "camera_rgb_optical_frame";
        cloud.height = 1;
        cloud.is_bigendian = false; // assumption
        cloud.is_dense = false; // chances are not all the available slots will be filled up
        

        sensor_msgs::PointCloud2Modifier modifier(cloud);
        //modifier.setPointCloud2FieldsByString(2,"xyz","rgba");
        modifier.setPointCloud2FieldsByString(2,"xyz","rgb");
        // there can be at most height x width pointclouds. In practice, this should be a lot less.
        modifier.resize(1 * height * width); 

        sensor_msgs::PointCloud2Iterator<float> out_x(cloud, "x");
        sensor_msgs::PointCloud2Iterator<float> out_y(cloud, "y");
        sensor_msgs::PointCloud2Iterator<float> out_z(cloud, "z");
        sensor_msgs::PointCloud2Iterator<uint8_t> out_r(cloud, "r");
        sensor_msgs::PointCloud2Iterator<uint8_t> out_g(cloud, "g");
        sensor_msgs::PointCloud2Iterator<uint8_t> out_b(cloud, "b");
        //sensor_msgs::PointCloud2Iterator<uint8_t> out_a(cloud, "a");
        cv::Mat recvRgbImg = cv_bridge::toCvCopy(data->rgb_image, "bgr8")->image;
        
        int exceptionCounter = 0;
        int pcCounter = 0;   
        
        for(int f = 0; f < fovLevel; f++){
            
            // this has been compressed into tiff byte data from the foveate_cpp node
            cv::Mat depthImg = cv::imdecode(data->foveated_images_groups[f].foveated_image.data, cv::IMREAD_ANYDEPTH);
            std::string filename = std::to_string(f) + "depthimg.png";
            cv::imwrite(filename, depthImg);
            depthImg.convertTo(depthImg, CV_32F);
            depthImg *= 0.001; // scale down to metres
            float scale_factor = (float) depthImg.size().height / (float) height; // computer by depth/rgb
            cv::Mat rgbImg;
            cv::resize(recvRgbImg, rgbImg, cv::Size(), scale_factor, scale_factor);
            float scaled_cx_d = cx_d * scale_factor;
            float scaled_cy_d = cy_d * scale_factor;
            float scaled_fx_d = fx_d * scale_factor;
            float scaled_fy_d = fy_d * scale_factor;
            float scaled_cx_r = cx_r * scale_factor;
            float scaled_cy_r = cy_r * scale_factor;
            float scaled_fx_r = fx_r * scale_factor;
            float scaled_fy_r = fy_r * scale_factor;
            
            for(int i = 0; i < depthImg.size().height; i++){
                for(int j = 0; j < depthImg.size().width; j++){
                    if(depthImg.at<int>(i, j) <= 0.01){
                        exceptionCounter++;
                        continue;
                    }
                    float P3D_x, P3D_y, P3D_z;
                    int P2D_x, P2D_y;
                    P3D_x = (((float) j - scaled_cx_d) * depthImg.at<float>(i, j) / scaled_fx_d) + T[0];
                    P3D_y = (((float) i - scaled_cy_d) * depthImg.at<float>(i, j) / scaled_fy_d) + T[1];
                    P3D_z = ((float) depthImg.at<float>(i, j)) + T[2];

                    P2D_x = (int) ((P3D_x * scaled_fx_r / P3D_z) + scaled_cx_r);
                    P2D_y = (int) ((P3D_y * scaled_fy_r / P3D_z) + scaled_cy_r);
                    //std::cout << i << " " << j << std::endl;
                    
                    if(P2D_x >= rgbImg.size().width || P2D_y >= rgbImg.size().height){
                        exceptionCounter++;
                        continue;
                    }
                    
                    try{
                        cv::Vec3b color = rgbImg.at<cv::Vec3b>(P2D_y,P2D_x);

                        std::uint8_t r = (color[2]);
                        std::uint8_t g = (color[1]);
                        std::uint8_t b = (color[0]);

                        *out_x = P3D_x;
                        *out_y = P3D_y;
                        *out_z = P3D_z;
                        
                        *out_r = r;
                        *out_g = g;
                        *out_b = b;
                        //*out_a = 255;
                        
                        ++out_x;
                        ++out_y;
                        ++out_z;
                        
                        ++out_r;
                        ++out_g;
                        ++out_b;
                        //++out_a;

                        pcCounter++;

                    }
                    catch (...){
                        exceptionCounter++;
                    }
                } // for j
            } // for i
        } // for f
        float ratio = (float) pcCounter / ((float)(height*width));
        ROS_INFO("Ratio of included/original: %f", ratio);
        //cloud.width = height * width;
        cloud.width = pcCounter;
        cloud.header = header;
        modifier.resize(pcCounter);
        
        try{
            pointcloudPublisher.publish(cloud);   
        }
        catch(...){
            ROS_ERROR("Failed to publish to the topic!");
        }
        
        
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        std::time_t end_time = std::chrono::system_clock::to_time_t(end);
        ros::Duration tempTime = ros::Time::now() - data->image_time;        
        double procTime = tempTime.toSec();
        running_total += procTime;
        running_counter++;
        ROS_INFO("Recomposition took %4.3f s, total time took %4.3f s, avg %4.3f over %3.0f samples",elapsed_seconds.count(), procTime, running_total/running_counter, running_counter);
                        
    }

};

union rgba{
    unsigned char colors[4];
    std::uint32_t colorBytes;
};

union xyzrgba{
    struct s{
        float x;
        float y;
        float z;
        std::uint32_t rgba;
    } data;
    uint8_t raw_data [sizeof(struct s)];
};

int main(int argc, char** argv){

   
    /* 
    cv::Mat depthImg = cv::Mat::zeros(480, 640, CV_16U);
    cv::Mat rgbImg = cv::imread("/home/main/Documents/master_thesis_ros/cython/rgb_img.png", cv::IMREAD_COLOR);

    cv::randu(depthImg, cv::Scalar(0), cv::Scalar(10000));


    auto start = std::chrono::system_clock::now();
    depthImg.convertTo(depthImg, CV_32F);
    depthImg *= 0.001;
    std::vector<float> rgbCameraInfo = {520.055928, 0.000000, 312.535255, 0.000000, 520.312173, 242.265554, 0.000000, 0.000000, 1.000000};
    std::vector<float> depthCameraInfo = {576.092756, 0.000000, 316.286974, 0.000000, 575.853472, 239.895662, 0.000000, 0.000000, 1.000000};
    
    std::vector<float> T = {-0.0254, -0.0013, -0.00218};

    float cx_d = depthCameraInfo[2];
    float cy_d = depthCameraInfo[5];
    float fx_d = depthCameraInfo[0];
    float fy_d = depthCameraInfo[4];
    float cx_r = rgbCameraInfo[2];
    float cy_r = rgbCameraInfo[5];
    float fx_r = rgbCameraInfo[0];
    float fy_r = rgbCameraInfo[4];

    float scale_factor = depthImg.size().height / rgbImg.size().height; // computer by depth/rgb
    
    float scaled_cx_d = cx_d * scale_factor;
    float scaled_cy_d = cy_d * scale_factor;
    float scaled_fx_d = fx_d * scale_factor;
    float scaled_fy_d = fy_d * scale_factor;
    float scaled_cx_r = cx_r * scale_factor;
    float scaled_cy_r = cy_r * scale_factor;
    float scaled_fx_r = fx_r * scale_factor;
    float scaled_fy_r = fy_r * scale_factor;
    
    int exceptionCounter = 0;
    int pcCounter = 0;

    sensor_msgs::PointCloud2 cloud;
    cloud.height = 1;
    cloud.is_bigendian = false;
    cloud.is_dense = false;

    sensor_msgs::PointCloud2Modifier modifier(cloud);
    modifier.setPointCloud2FieldsByString(2,"xyz","rgb");
    modifier.resize(1 * rgbImg.size().height * rgbImg.size().width);

    sensor_msgs::PointCloud2Iterator<float> out_x(cloud, "x");
    sensor_msgs::PointCloud2Iterator<float> out_y(cloud, "y");
    sensor_msgs::PointCloud2Iterator<float> out_z(cloud, "z");
    sensor_msgs::PointCloud2Iterator<uint8_t> out_r(cloud, "r");
    sensor_msgs::PointCloud2Iterator<uint8_t> out_g(cloud, "g");
    sensor_msgs::PointCloud2Iterator<uint8_t> out_b(cloud, "b");

    std::cout << depthImg.size().height << " " << depthImg.size().width << std::endl;
    for(int i = 0; i < depthImg.size().height; i++){
        for(int j = 0; j < depthImg.size().width; j++){
            if(depthImg.at<int>(i, j) <= 0.01){
                exceptionCounter++;
                continue;
            }
            float P3D_x, P3D_y, P3D_z;
            int P2D_x, P2D_y;
            P3D_x = (((float) j - scaled_cx_d) * depthImg.at<int>(i, j) / scaled_fx_d) + T[0];
            P3D_y = (((float) i - scaled_cy_d) * depthImg.at<int>(i, j) / scaled_fy_d) + T[1];
            P3D_z = ((float) depthImg.at<int>(i, j)) + T[2];

            P2D_x = (int) ((P3D_x * scaled_fx_r / P3D_z) + scaled_cx_r);
            P2D_y = (int) ((P3D_y * scaled_fy_r / P3D_z) + scaled_cy_r);
            //std::cout << i << " " << j << std::endl;
            //std::cout << P3D_x << " " << P3D_y << " " << P3D_z << std::endl;
            
            if(P2D_x >= rgbImg.size().width || P2D_y >= rgbImg.size().height){
                exceptionCounter++;
                continue;
            }
            
            try{
                cv::Vec3b color = rgbImg.at<cv::Vec3b>(P2D_y,P2D_x);

                std::uint8_t r = color[2];
                std::uint8_t g = color[1];
                std::uint8_t b = color[0];

                *out_x = P3D_x;
                *out_y = P3D_y;
                *out_z = P3D_z;
                *out_r = r;
                *out_g = g;
                *out_b = b;
                
                ++out_x;
                ++out_y;
                ++out_z;
                ++out_r;
                ++out_g;
                ++out_b;

                pcCounter++;

            }
            catch (...){
                exceptionCounter++;
            }
        }
    }
    std::cout<<"Excluded RGBAXYZ points: " << exceptionCounter << std::endl;
    std::cout<<"Included RGBAXYZ points: " << pcCounter << std::endl;

    cloud.width = pcCounter;

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);

    std::cout << "finished computation at " << std::ctime(&end_time)
            << "elapsed time: " << elapsed_seconds.count() << "s\n";
    */
    
    ros::init(argc, argv, "recompose_cpp");

    ros::NodeHandle n("~");
       

    std::string foveationTopic, publishTopic, rgbCameraTopic, depthCameraTopic, savePath;
    bool saveImg, saveRgb, showImg;

    n.getParam("publish_topic", publishTopic);
    n.getParam("foveation_topic", foveationTopic);
    n.getParam("rgb_camera", rgbCameraTopic);
    n.getParam("depth_camera", depthCameraTopic);
    n.getParam("save_path", savePath);

    n.getParam("save_img", saveImg);
    n.getParam("save_rgb", saveRgb);
    n.getParam("show_img", showImg);
    
    ROS_INFO("Started recomposition node with the following parameters.");
    ROS_INFO("publish_topic: %s", publishTopic.c_str());
    ROS_INFO("foveation_topic: %s", foveationTopic.c_str());
    ROS_INFO("rgb_camera: %s", rgbCameraTopic.c_str());
    ROS_INFO("depth_camera: %s", depthCameraTopic.c_str());
    ROS_INFO("save_path: %s", savePath.c_str());
    ROS_INFO("save_img: %d", saveImg);
    ROS_INFO("save_rgb: %d", saveRgb);
    ROS_INFO("show_img: %d", showImg);

    RecompositionClass rc = RecompositionClass(&n, foveationTopic, publishTopic, rgbCameraTopic, depthCameraTopic, savePath, saveImg, saveRgb, showImg);

    ros::spin();
    return 0;
}
