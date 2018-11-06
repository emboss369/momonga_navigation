#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv/cv.h>
#include <cv_bridge/cv_bridge.h>
#include <image_geometry/pinhole_camera_model.h>
#include <tf/transform_listener.h>
#include <sensor_msgs/image_encodings.h>

#include <momonga_navigation/TrafficLightDetect.h>

// image_topicを購読して、サーバに渡して結果を受け取るクライアント

class FrameDrawer
{
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::CameraSubscriber sub_;
    tf::TransformListener tf_listener_;
    image_geometry::PinholeCameraModel cam_model_;
    CvFont font_;

    ros::ServiceClient client_;

  public:
    FrameDrawer()
        : it_(nh_)
    {
        std::string image_topic = nh_.resolveName("image");
        sub_ = it_.subscribeCamera(image_topic, 1, &FrameDrawer::imageCb, this);
        cvInitFont(&font_, CV_FONT_HERSHEY_SIMPLEX, 0.5, 0.5);

        client_ = nh_.serviceClient<momonga_navigation::TrafficLightDetect>("image_server");
    }

    void imageCb(const sensor_msgs::ImageConstPtr &image_msg,
                 const sensor_msgs::CameraInfoConstPtr &info_msg)
    {
        cv::Mat image;
        cv_bridge::CvImagePtr input_bridge;
        try
        {
            input_bridge = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
            image = input_bridge->image;
        }
        catch (cv_bridge::Exception &ex)
        {
            ROS_ERROR("[draw_frames] Failed to convert image");
            return;
        }

        cam_model_.fromCameraInfo(info_msg);

        static const int RADIUS = 3;
        cv::circle(image, cv::Point2d(10.0, 10.0), RADIUS, CV_RGB(255, 0, 0), -1);

        // サーバーを呼び出して結果を受け取る

        momonga_navigation::TrafficLightDetect srv;
        srv.request.img = *(input_bridge->toImageMsg());
        if (client_.call(srv))
        {
            ROS_INFO_STREAM("DETECT RESULT : " << srv.response.category);
        }
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "image_client");
    //std::vector<std::string> frame_ids(argv + 1, argv + argc);
    FrameDrawer drawer();
    ros::spin();
}
