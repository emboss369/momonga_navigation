#include <ros/ros.h>
#include <ros/package.h>
#include <momonga_navigation/TrafficLightDetect.h>
#include <image_transport/image_transport.h>
#include <opencv/cv.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
// #include <image_geometry/pinhole_camera_model.h>
// #include <tf/transform_listener.h>
// #include <boost/foreach.hpp>
// #include <sensor_msgs/image_encodings.h>
#include <fstream>
#include <utility>
#include <vector>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

using tensorflow::Flag;
using tensorflow::int32;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::Tensor;



// Takes a file name, and loads a list of labels from it, one per line, and
// returns a vector of the strings. It pads with empty strings so the length
// of the result is a multiple of 16, because our model expects that.
Status ReadLabelsFile(const string &file_name, std::vector<string> *result,
                      size_t *found_label_count)
{
    std::ifstream file(file_name);
    if (!file)
    {
        return tensorflow::errors::NotFound("Labels file ", file_name,
                                            " not found.");
    }
    result->clear();
    string line;
    while (std::getline(file, line))
    {
        result->push_back(line);
    }
    *found_label_count = result->size();
    const int padding = 16;
    while (result->size() % padding)
    {
        result->emplace_back();
    }
    return Status::OK();
}

static Status ReadEntireFile(tensorflow::Env *env, const string &filename,
                             Tensor *output)
{
    tensorflow::uint64 file_size = 0;
    TF_RETURN_IF_ERROR(env->GetFileSize(filename, &file_size));

    string contents;
    contents.resize(file_size);

    std::unique_ptr<tensorflow::RandomAccessFile> file;
    TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));

    tensorflow::StringPiece data;
    TF_RETURN_IF_ERROR(file->Read(0, file_size, &data, &(contents)[0]));
    if (data.size() != file_size)
    {
        return tensorflow::errors::DataLoss("Truncated read of '", filename,
                                            "' expected ", file_size, " got ",
                                            data.size());
    }
    output->scalar<string>()() = string(data);
    return Status::OK();
}

// Given an image file name, read in the data, try to decode it as an image,
// resize it to the requested size, and then scale the values as desired.
Status ReadTensorFromImageFile(const string &file_name, const int input_height,
                               const int input_width, const float input_mean,
                               const float input_std,
                               std::vector<Tensor> *out_tensors)
{
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops; // NOLINT(build/namespaces)

    string input_name = "file_reader";
    string output_name = "normalized";

    // read file_name into a tensor named input
    Tensor input(tensorflow::DT_STRING, tensorflow::TensorShape());
    TF_RETURN_IF_ERROR(
        ReadEntireFile(tensorflow::Env::Default(), file_name, &input));

    // use a placeholder to read input data
    auto file_reader =
        Placeholder(root.WithOpName("input"), tensorflow::DataType::DT_STRING);

    std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
        {"input", input},
    };

    // Now try to figure out what kind of file it is and decode it.
    const int wanted_channels = 3;
    tensorflow::Output image_reader;
    if (tensorflow::str_util::EndsWith(file_name, ".png"))
    {
        image_reader = DecodePng(root.WithOpName("png_reader"), file_reader,
                                 DecodePng::Channels(wanted_channels));
    }
    else if (tensorflow::str_util::EndsWith(file_name, ".gif"))
    {
        // gif decoder returns 4-D tensor, remove the first dim
        image_reader =
            Squeeze(root.WithOpName("squeeze_first_dim"),
                    DecodeGif(root.WithOpName("gif_reader"), file_reader));
    }
    else if (tensorflow::str_util::EndsWith(file_name, ".bmp"))
    {
        image_reader = DecodeBmp(root.WithOpName("bmp_reader"), file_reader);
    }
    else
    {
        // Assume if it's neither a PNG nor a GIF then it must be a JPEG.
        image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
                                  DecodeJpeg::Channels(wanted_channels));
    }
    // Now cast the image data to float so we can do normal math on it.
    auto float_caster =
        Cast(root.WithOpName("float_caster"), image_reader, tensorflow::DT_FLOAT);
    // The convention for image ops in TensorFlow is that all images are expected
    // to be in batches, so that they're four-dimensional arrays with indices of
    // [batch, height, width, channel]. Because we only have a single image, we
    // have to add a batch dimension of 1 to the start with ExpandDims().
    auto dims_expander = ExpandDims(root, float_caster, 0);
    // Bilinearly resize the image to fit the required dimensions.
    auto resized = ResizeBilinear(
        root, dims_expander,
        Const(root.WithOpName("size"), {input_height, input_width}));
    // Subtract the mean and divide by the scale.
    Div(root.WithOpName(output_name), Sub(root, resized, {input_mean}),
        {input_std});

    // This runs the GraphDef network definition that we've just constructed, and
    // returns the results in the output tensor.
    tensorflow::GraphDef graph;
    TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

    std::unique_ptr<tensorflow::Session> session(
        tensorflow::NewSession(tensorflow::SessionOptions()));
    TF_RETURN_IF_ERROR(session->Create(graph));
    TF_RETURN_IF_ERROR(session->Run({inputs}, {output_name}, {}, out_tensors));
    return Status::OK();
}

static Status ReadEntireMat(tensorflow::Env *env, const cv::Mat &mat,
                            Tensor *output)
{
    vector<uchar> buff; //buffer for coding
    vector<int> param = vector<int>(2);
    param[0] = CV_IMWRITE_JPEG_QUALITY;
    param[1] = 100; //default(95) 0-100
    cv::imencode(".jpg", mat, buff, param)



        tensorflow::uint64 file_size = buff.size();
    //TF_RETURN_IF_ERROR(env->GetFileSize(filename, &file_size));
    

        string contents;
    contents.resize(file_size);

    std::unique_ptr<tensorflow::RandomAccessFile> file;
    TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));

    tensorflow::StringPiece data;
    TF_RETURN_IF_ERROR(file->Read(0, file_size, &data, &(contents)[0]));
    if (data.size() != file_size)
    {
        return tensorflow::errors::DataLoss("Truncated read of '", filename,
                                            "' expected ", file_size, " got ",
                                            data.size());
    }
    output->scalar<string>()() = string(data);
    return Status::OK();
}

Status ReadTensorFromMat(const cv::Mat &mat,
                         const int input_height,
                         const int input_width,
                         const float input_mean,
                         const float input_std, 
                         std::vector<Tensor> *out_tensors)
{
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops; // NOLINT(build/namespaces)

    string input_name = "file_reader";
    string output_name = "normalized";

    // read file_name into a tensor named input
    Tensor input(tensorflow::DT_STRING, tensorflow::TensorShape());
    TF_RETURN_IF_ERROR(
        ReadEntireMat(tensorflow::Env::Default(), mat, &input));

    // use a placeholder to read input data
    auto file_reader =
        Placeholder(root.WithOpName("input"), tensorflow::DataType::DT_STRING);

    std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
        {"input", input},
    };

    // Now try to figure out what kind of file it is and decode it.
    const int wanted_channels = 3;
    tensorflow::Output image_reader;
    if (tensorflow::str_util::EndsWith(file_name, ".png"))
    {
        image_reader = DecodePng(root.WithOpName("png_reader"), file_reader,
                                 DecodePng::Channels(wanted_channels));
    }
    else if (tensorflow::str_util::EndsWith(file_name, ".gif"))
    {
        // gif decoder returns 4-D tensor, remove the first dim
        image_reader =
            Squeeze(root.WithOpName("squeeze_first_dim"),
                    DecodeGif(root.WithOpName("gif_reader"), file_reader));
    }
    else if (tensorflow::str_util::EndsWith(file_name, ".bmp"))
    {
        image_reader = DecodeBmp(root.WithOpName("bmp_reader"), file_reader);
    }
    else
    {
        // Assume if it's neither a PNG nor a GIF then it must be a JPEG.
        image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
                                  DecodeJpeg::Channels(wanted_channels));
    }
    // Now cast the image data to float so we can do normal math on it.
    auto float_caster =
        Cast(root.WithOpName("float_caster"), image_reader, tensorflow::DT_FLOAT);
    // The convention for image ops in TensorFlow is that all images are expected
    // to be in batches, so that they're four-dimensional arrays with indices of
    // [batch, height, width, channel]. Because we only have a single image, we
    // have to add a batch dimension of 1 to the start with ExpandDims().
    auto dims_expander = ExpandDims(root, float_caster, 0);
    // Bilinearly resize the image to fit the required dimensions.
    auto resized = ResizeBilinear(
        root, dims_expander,
        Const(root.WithOpName("size"), {input_height, input_width}));
    // Subtract the mean and divide by the scale.
    Div(root.WithOpName(output_name), Sub(root, resized, {input_mean}),
        {input_std});

    // This runs the GraphDef network definition that we've just constructed, and
    // returns the results in the output tensor.
    tensorflow::GraphDef graph;
    TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

    std::unique_ptr<tensorflow::Session> session(
        tensorflow::NewSession(tensorflow::SessionOptions()));
    TF_RETURN_IF_ERROR(session->Create(graph));
    TF_RETURN_IF_ERROR(session->Run({inputs}, {output_name}, {}, out_tensors));
    return Status::OK();
}

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status LoadGraph(const string &graph_file_name,
                 std::unique_ptr<tensorflow::Session> *session)
{
    tensorflow::GraphDef graph_def;
    Status load_graph_status =
        ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
    if (!load_graph_status.ok())
    {
        return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                            graph_file_name, "'");
    }
    session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
    Status session_create_status = (*session)->Create(graph_def);
    if (!session_create_status.ok())
    {
        return session_create_status;
    }
    return Status::OK();
}

// Analyzes the output of the Inception graph to retrieve the highest scores and
// their positions in the tensor, which correspond to categories.
Status GetTopLabels(const std::vector<Tensor> &outputs, int how_many_labels,
                    Tensor *indices, Tensor *scores)
{
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops; // NOLINT(build/namespaces)

    string output_name = "top_k";
    TopK(root.WithOpName(output_name), outputs[0], how_many_labels);
    // This runs the GraphDef network definition that we've just constructed, and
    // returns the results in the output tensors.
    tensorflow::GraphDef graph;
    TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

    std::unique_ptr<tensorflow::Session> session(
        tensorflow::NewSession(tensorflow::SessionOptions()));
    TF_RETURN_IF_ERROR(session->Create(graph));
    // The TopK node returns two outputs, the scores and their original indices,
    // so we have to append :0 and :1 to specify them both.
    std::vector<Tensor> out_tensors;
    TF_RETURN_IF_ERROR(session->Run({}, {output_name + ":0", output_name + ":1"},
                                    {}, &out_tensors));
    *scores = out_tensors[0];
    *indices = out_tensors[1];
    return Status::OK();
}

// Given the output of a model run, and the name of a file containing the labels
// this prints out the top five highest-scoring values.
Status PrintTopLabels(const std::vector<Tensor> &outputs,
                      const string &labels_file_name)
{
    std::vector<string> labels;
    size_t label_count;
    Status read_labels_status =
        ReadLabelsFile(labels_file_name, &labels, &label_count);
    if (!read_labels_status.ok())
    {
        LOG(ERROR) << read_labels_status;
        return read_labels_status;
    }
    const int how_many_labels = std::min(5, static_cast<int>(label_count));
    Tensor indices;
    Tensor scores;
    TF_RETURN_IF_ERROR(GetTopLabels(outputs, how_many_labels, &indices, &scores));
    tensorflow::TTypes<float>::Flat scores_flat = scores.flat<float>();
    tensorflow::TTypes<int32>::Flat indices_flat = indices.flat<int32>();
    for (int pos = 0; pos < how_many_labels; ++pos)
    {
        const int label_index = indices_flat(pos);
        const float score = scores_flat(pos);
        LOG(INFO) << labels[label_index] << " (" << label_index << "): " << score;
    }
    return Status::OK();
}

// This is a testing function that returns whether the top label index is the
// one that's expected.
Status CheckTopLabel(const std::vector<Tensor> &outputs, int expected,
                     bool *is_expected)
{
    *is_expected = false;
    Tensor indices;
    Tensor scores;
    const int how_many_labels = 1;
    TF_RETURN_IF_ERROR(GetTopLabels(outputs, how_many_labels, &indices, &scores));
    tensorflow::TTypes<int32>::Flat indices_flat = indices.flat<int32>();
    if (indices_flat(0) != expected)
    {
        LOG(ERROR) << "Expected label #" << expected << " but got #"
                   << indices_flat(0);
        *is_expected = false;
    }
    else
    {
        *is_expected = true;
    }
    return Status::OK();
}

class ImageServer
{
    ros::NodeHandle nh_;
    ros::ServiceServer service;

    int argc_;
    char **argv_;

  public:
    ImageServer(int argc, char **argv)
        : argc_(argc), argv_(argv)
    {
        ROS_INFO("init");
        service = nh_.advertiseService("image_server", &ImageServer::detectImage, this);
    }

    bool detectImage(momonga_navigation::TrafficLightDetect::Request &request,
                     momonga_navigation::TrafficLightDetect::Response &response )
    {
        ROS_INFO("detectImage");

        // 受け取った画像をMatに変換する
        cv_bridge::CvImagePtr cv_ptr;
        cv_ptr = cv_bridge::toCvCopy(request.img, sensor_msgs::image_encodings::BGR8);

        // These are the command-line flags the program can understand.
        // They define where the graph and input data is located, and what kind of
        // input the model expects. If you train your own model, or use something
        // other than inception_v3, then you'll need to update these.
        string image = ros::package::getPath("momonga_navigation") + "/tf/img_0.jpg";
        string graph =
            ros::package::getPath("momonga_navigation") + "/tf/trafficlight_graph.pb";
        string labels =
            ros::package::getPath("momonga_navigation") + "/tf/trafficlight_labels.txt";

        int32 input_width = 224;
        int32 input_height = 224;
        float input_mean = 128;
        float input_std = 128;
        string input_layer = "input";
        // string output_layer = "InceptionV3/Predictions/Reshape_1";
        string output_layer = "final_result";
        bool self_test = false;
        string root_dir = "";
        std::vector<Flag> flag_list = {
            Flag("image", &image, "image to be processed"),
            Flag("graph", &graph, "graph to be executed"),
            Flag("labels", &labels, "name of file containing labels"),
            Flag("input_width", &input_width, "resize image to this width in pixels"),
            Flag("input_height", &input_height,
                 "resize image to this height in pixels"),
            Flag("input_mean", &input_mean, "scale pixel values to this mean"),
            Flag("input_std", &input_std, "scale pixel values to this std deviation"),
            Flag("input_layer", &input_layer, "name of input layer"),
            Flag("output_layer", &output_layer, "name of output layer"),
            Flag("self_test", &self_test, "run a self test"),
            Flag("root_dir", &root_dir,
                 "interpret image and graph file names relative to this directory"),
        };
        string usage = tensorflow::Flags::Usage(argv_[0], flag_list);
        const bool parse_result = tensorflow::Flags::Parse(&argc_, argv_, flag_list);
        if (!parse_result)
        {
            LOG(ERROR) << usage;
            return -1;
        }

        // We need to call this to set up global state for TensorFlow.
        tensorflow::port::InitMain(argv_[0], &argc_, &argv_);
        if (argc_ > 1)
        {
            LOG(ERROR) << "Unknown argument " << argv_[1] << "\n"
                       << usage;
            return -1;
        }

        // First we load and initialize the model.
        std::unique_ptr<tensorflow::Session> session;
        string graph_path = tensorflow::io::JoinPath(root_dir, graph);
        Status load_graph_status = LoadGraph(graph_path, &session);
        if (!load_graph_status.ok())
        {
            LOG(ERROR) << load_graph_status;
            return -1;
        }

        // Get the image from disk as a float array of numbers, resized and normalized
        // to the specifications the main graph expects.
        std::vector<Tensor> resized_tensors;
        string image_path = tensorflow::io::JoinPath(root_dir, image);
        // Status read_tensor_status =
        //     ReadTensorFromImageFile(image_path, input_height, input_width, input_mean,
        //                             input_std, &resized_tensors);

        Status read_tensor_status =
            readTensorFromMat(cv_ptr->image, input_height, input_width, input_mean, input_std, &resized_tensors);

        if (!read_tensor_status.ok())
        {
            LOG(ERROR) << read_tensor_status;
            return -1;
        }
        const Tensor &resized_tensor = resized_tensors[0];

        // Actually run the image through the model.
        std::vector<Tensor> outputs;
        Status run_status = session->Run({{input_layer, resized_tensor}},
                                         {output_layer}, {}, &outputs);
        if (!run_status.ok())
        {
            LOG(ERROR) << "Running model failed: " << run_status;
            return -1;
        }

        // This is for automated testing to make sure we get the expected result with
        // the default settings. We know that label 653 (military uniform) should be
        // the top label for the Admiral Hopper image.
        if (self_test)
        {
            bool expected_matches;
            Status check_status = CheckTopLabel(outputs, 653, &expected_matches);
            if (!check_status.ok())
            {
                LOG(ERROR) << "Running check failed: " << check_status;
                return -1;
            }
            if (!expected_matches)
            {
                LOG(ERROR) << "Self-test failed!";
                return -1;
            }
        }

        // Do something interesting with the results we've generated.
        Status print_status = PrintTopLabels(outputs, labels);
        if (!print_status.ok())
        {
            LOG(ERROR) << "Running print failed: " << print_status;
            return -1;
        }





        

        response.category = "traffic light is red (test)";





        // DEBUG SHOW
        if(true){
            cv::circle(cv_ptr->image, cv::Point(100, 100), 20, CV_RGB(0, 255, 0));

            cv::Mat g_last_image = cv_ptr->image;
            const cv::Mat &image = g_last_image;
            cv::imshow("Original Image",cv_ptr->image);
            cv::waitKey(3);
        }

        return true;
    }
};

int
main(int argc, char **argv)
{
    ros::init(argc, argv, "image_server");

    ImageServer server(argc,argv);

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
