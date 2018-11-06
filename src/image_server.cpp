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


  public:
    ImageServer()
    {

        ros::ServiceServer service = nh_.advertiseService("image_server", &ImageServer::detectImage,this);
    }

    bool detectImage(momonga_navigation::TrafficLightDetect::Request &request,
                     momonga_navigation::TrafficLightDetect::Response &response )
    {
        cv_bridge::CvImagePtr cv_ptr;

        cv_ptr = cv_bridge::toCvCopy(request.img, sensor_msgs::image_encodings::BGR8);
        cv::circle(cv_ptr->image, cv::Point(100, 100), 20, CV_RGB(0, 255, 0));
        cv::imshow("Original Image",cv_ptr->image);

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
