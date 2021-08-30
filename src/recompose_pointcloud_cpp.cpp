#include "ros/ros.h"
#include "std_msgs/String.h"
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
        foveationSubscriber = nh->subscribe(foveationTopic, 100, &RecompositionClass::recompositionCallback, this);
        pointcloudPublisher = nh->advertise<sensor_msgs::PointCloud2>(publishTopic, 10);
        
        rgbCameraInfo = ros::topic::waitForMessage<sensor_msgs::CameraInfo>(rgbCameraTopic, *nh, ros::Duration(0.5));
        depthCameraInfo = ros::topic::waitForMessage<sensor_msgs::CameraInfo>(depthCameraTopic, *nh, ros::Duration(0.5));
        
        
        saveImg = saveImg_;
        saveRgb = saveRgb_;
        showImg = showImg_;
        std::cout << "Initialised the recomposition node" << std::endl;
    }

    void recompositionCallback(const attention_package::FoveatedImageMeta::ConstPtr& data){
        std::cout << "got some message" << std::endl;
    }
// 
/* 
    def __init__(self):
        # args from launch file
        rospy.loginfo('Pointcloud recomposing waiting for camera infos')
        self.save_img = rospy.get_param("~save_img")
        self.show_img = rospy.get_param("~show_img")
        self.save_rgb = rospy.get_param("~save_rgb")
        self.save_path = rospy.get_param("~save_path") + "recompose_imgs"
        publish_topic = rospy.get_param("~publish_topic")
        foveation_topic = rospy.get_param("~foveation_topic")
        rgb_camera = rospy.get_param("~rgb_camera")
        depth_camera = rospy.get_param("~depth_camera")
        #self.rgb_camera_info = rospy.wait_for_message(rgb_camera, CameraInfo)
        #self.depth_camera_info = rospy.wait_for_message(depth_camera, CameraInfo)
        
        # hardcode
        self.rgb_camera_info = {'K':[520.055928, 0.000000, 312.535255, 0.000000, 520.312173, 242.265554, 0.000000, 0.000000, 1.000000]}
        self.depth_camera_info = {'K':[576.092756, 0.000000, 316.286974, 0.000000, 575.853472, 239.895662, 0.000000, 0.000000, 1.000000]}
        
        # Camera parameters, where d = depth and r = rgb
        self.cx_d = self.depth_camera_info['K'][2]
        self.cy_d = self.depth_camera_info['K'][5]
        self.fx_d = self.depth_camera_info['K'][0]
        self.fy_d = self.depth_camera_info['K'][4]
        self.cx_r = self.rgb_camera_info['K'][2]
        self.cy_r = self.rgb_camera_info['K'][5]
        self.fx_r = self.rgb_camera_info['K'][0]
        self.fy_r = self.rgb_camera_info['K'][4]
        self.T = [-0.0254, -0.00013, -0.00218]
        
        # PointCloud2 args
        self.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgb', 12, PointField.UINT32, 1)
        ]
        self.header = std_msgs.msg.Header()
        self.header.frame_id = "camera_rgb_optical_frame"
        
        self.publisher = rospy.Publisher(publish_topic, PointCloud2, queue_size=5)
        self.subscriber = rospy.Subscriber(foveation_topic, FoveatedImageMeta, self.recompose_callback)
        self.bridge = cv_bridge.CvBridge()
        
        self.callback_counter = 0
        if(self.save_img):
            self.save_path = create_dir_and_return_path(self.save_path)
        init_string = "Initialised recompose_node with the following parameters:\n" + \
                    f"foveation_topic: {foveation_topic}\n" + \
                    f"publish_topic: {publish_topic}\n" + \
                    f"rgb_camera: {rgb_camera}\n" + \
                    f"depth_camera: {depth_camera}\n" + \
                    f"save_img: {self.save_img}\n" + \
                    f"save_rgb: {self.save_rgb}\n" + \
                    f"show_img: {self.show_img}\n" + \
                    f"save_path: {self.save_path}"

                        

        rospy.loginfo(init_string)
*/
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
    ros::init(argc, argv, "recompose_pc_cpp");

    ros::NodeHandle n("~");

    std::string publishTopic, foveationTopic, savePath;
    bool saveImg, saveRgb, showImg;

    n.getParam("recompose_topic", publishTopic);
    n.getParam("foveation_topic", foveationTopic);
    n.getParam("save_path", savePath);

    n.getParam("save_img", saveImg);
    n.getParam("save_rgb", saveRgb);
    n.getParam("show_img", showImg);

    ROS_INFO("Started recomposition node with the following parameters.");
    ROS_INFO("recompose_topic: %s", publishTopic.c_str());
    ROS_INFO("foveation_topic: %s", foveationTopic.c_str());
    ROS_INFO("save_path: %s", savePath.c_str());
    ROS_INFO("save_img: %d", saveImg);
    ROS_INFO("save_rgb: %d", saveRgb);
    ROS_INFO("show_img: %d", showImg);

    RecompositionClass rc = RecompositionClass(&n, foveationTopic, publishTopic, savePath, saveImg, saveRgb, showImg);

    ros::spin();
    */
    
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

    return 0;
}