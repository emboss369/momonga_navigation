#include <ros/ros.h>
#include <momonga_navigation/TrafficLightDetect.h>
#include <image_transport/image_transport.h>
#include <opencv/cv.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
// #include <image_geometry/pinhole_camera_model.h>
// #include <tf/transform_listener.h>
// #include <boost/foreach.hpp>
// #include <sensor_msgs/image_encodings.h>

// image_topicを購読して、サーバに渡して結果を受け取るクライアント

class ImageServer
{
    ros::NodeHandle nh_;
ros::ServiceServer service;

  public:
    ImageServer()
    {
        ROS_INFO("init");
        service = nh_.advertiseService("image_server", &ImageServer::detectImage, this);
    }

    bool detectImage(momonga_navigation::TrafficLightDetect::Request &request,
                     momonga_navigation::TrafficLightDetect::Response &response )
    {
        ROS_INFO("detectImage");
        cv_bridge::CvImagePtr cv_ptr;

        cv_ptr = cv_bridge::toCvCopy(request.img, sensor_msgs::image_encodings::BGR8);
        cv::circle(cv_ptr->image, cv::Point(100, 100), 20, CV_RGB(0, 255, 0));

        cv::Mat g_last_image = cv_ptr->image;
        const cv::Mat &image = g_last_image;
        cv::imshow("Original Image",cv_ptr->image);
        cv::waitKey(3);
        response.category = "traffic light is red (test)";
        return true;
    }
};

int
main(int argc, char **argv)
{
    ros::init(argc, argv, "image_server");

    ImageServer server;

    ROS_INFO("image_server ready");

    ros::spin();
    return 0;
}
// bool add(momonga_navigation::TrafficLightDetect::Request  &req,
//          momonga_navigation::TrafficLightDetect::Response &res)
// {
//     ROS_INFO("add");
//   // res.sum = req.a + req.b;
//   // ROS_INFO("request: x=%ld, y=%ld", (long int)req.a, (long int)req.b);
//   // ROS_INFO("sending back response: [%ld]", (long int)res.sum);
//   return true;
// }
//
// int main(int argc, char **argv)
// {
//     ROS_INFO("init");
//   ros::init(argc, argv, "image_server");
//   ros::NodeHandle n;
//
//   ROS_INFO("service");
//   ros::ServiceServer service = n.advertiseService("image_server", add);
//   ROS_INFO("Ready to add two ints.");
//   ros::spin();
//
//   return 0;
// }
