#include <ros/ros.h>
#include <momonga_navigation/ImageDetection.h>
// #include <image_transport/image_transport.h>
// #include <opencv/cv.h>
// #include <cv_bridge/cv_bridge.h>
// #include <image_geometry/pinhole_camera_model.h>
// #include <tf/transform_listener.h>
// #include <boost/foreach.hpp>
// #include <sensor_msgs/image_encodings.h>

// image_topicを購読して、サーバに渡して結果を受け取るクライアント

class ImageServer
{
    ros::NodeHandle nh_;

  public:
    ImageServer(const std::vector<std::string> &frame_ids)
    {
        ros::ServiceServer service = nh_.advertiseService("image_server", &ImageServer::detectImage);
    }

    bool detectImage(momonga_navigation::ImageDetection::Reuest &request,
                     momonga_navigation::ImageDetection::Response &response, )
    {
        request.img 


    }
}

int
main(int argc, char **argv)
{
    ros::init(argc, argv, "image_server");



    ros::spin();
    return 0;
}
