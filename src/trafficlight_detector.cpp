#include <ros/ros.h>
#include <ros/package.h> // パッケージの検索
#include <geometry_msgs/Pose.h>
#include <tf/transform_listener.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h> // RVizに表示するマーカー。配列バージョン
#include <fstream>

#include <boost/tokenizer.hpp>
typedef boost::tokenizer<boost::char_separator<char>> tokenizer;

// 1このWaypointを表すクラス
class TrafficLight
{
  public:
    TrafficLight();
    TrafficLight(geometry_msgs::Pose pose, std::string reach_threshold)
        : pose_(pose), name_(reach_threshold)
    {
    }
    geometry_msgs::Pose pose_;
    std::string name_;
};

class TrafficlightDetector
{
  public:
    // コンストラクタ
    TrafficlightDetector() : rate_(10)
    {
        std::string filename; // TrafficeLights一覧CSVファイル名
        ros::NodeHandle n("~");
        n.param<std::string>("waypointsfile",
                             filename,
                             ros::package::getPath("turtlebot3_momonga") + "/waypoints/trafficelights.csv");

        ROS_INFO("[TraficLights file name] : %s", filename.c_str());

        // Traffic Lightの表示用
        trafficlight_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/traffic_lights", 1);

        readTrafficLights(filename.c_str());
    }


    int readTrafficLights(std::string filename)
    {
        const int rows_num = 7; // x, y, z, Qx,Qy,Qz,Qw
        boost::char_separator<char> sep(",", "", boost::keep_empty_tokens);
        std::ifstream ifs(filename.c_str());
        std::string line;
        while (ifs.good())
        {
            getline(ifs, line);
            if (line.empty())
                break;

            tokenizer tokens(line, sep);
            std::vector<double> data;
            tokenizer::iterator it = tokens.begin();
            for (; it != tokens.end(); ++it)
            {
                std::stringstream ss;
                double d;
                ss << *it;
                ss >> d;
                data.push_back(d);
            }
            if (data.size() != rows_num)
            {
                ROS_ERROR("Row size is mismatch!!");
                return -1;
            }
            else
            {
                geometry_msgs::Pose pose;
                pose.position.x = data[0];
                pose.position.y = data[1];
                pose.position.z = data[2];
                pose.orientation.x = data[3];
                pose.orientation.y = data[4];
                pose.orientation.z = data[5];
                pose.orientation.w = data[6];
                waypoints_.push_back(WayPoint(waypoint, (int)data[7], data[8] / 2.0));
            }
        }
    }

  private:
    ros::Rate rate_;
    ros::NodeHandle nh_;
    ros::Publisher trafficlight_pub_; // 信号位置をパブリッシュする
    std::vector<geometry_msgs::Pose> trafficlights_; // 読み込んだTraficLightの配列
};
