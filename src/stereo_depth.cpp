#include <memory>
#include <map>
#include <stdio.h>
#include <sstream>
#include <vector>

#include <ros/ros.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud2.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <sensor_msgs/point_cloud2_iterator.h>

#include <dynamic_reconfigure/server.h>
#include <stereo_depth/DisparityConfig.h>


typedef sensor_msgs::PointCloud2Iterator<float> PCIteratorf;
typedef sensor_msgs::PointCloud2Iterator<uint8_t> PCIteratori;

typedef message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image> ImSync;

typedef dynamic_reconfigure::Server<stereo_depth::DisparityConfig> DisparityConfig;


class StereoDepth
{
public:
        ros::Subscriber info1_, info2_;
        std::unique_ptr<ImSync> sync_;
        std::map<std::string, sensor_msgs::CameraInfo> camera_info_;

        message_filters::Subscriber<sensor_msgs::Image> left_image_sub_, right_image_sub_;

        ros::Publisher cloud_publisher_;

        stereo_depth::DisparityConfig sgbm_config_;
        DisparityConfig server_;
        cv::Mat Q_;
        double baseline_;
        double scale_;

    StereoDepth()
    {
        ros::NodeHandle nh, pnh("~");

        Q_ = cv::Mat::zeros(4, 4, CV_64FC1);
        pnh.param("baseline", baseline_, 0.05);
        pnh.param("scale", scale_, 1.0);

        server_.setCallback(boost::bind(&StereoDepth::DynamicReconfigureCallback, this, _1, _2));

        cloud_publisher_ = nh.advertise<sensor_msgs::PointCloud2>("pointcloud", 1);

        info1_ = nh.subscribe("left/camera_info", 1, &StereoDepth::CameraInfoCallback, this);
        info2_ = nh.subscribe("right/camera_info", 1, &StereoDepth::CameraInfoCallback, this);

        left_image_sub_.subscribe(nh, "left/image", 10);
        right_image_sub_.subscribe(nh, "right/image", 10);

        sync_.reset(new ImSync(left_image_sub_, right_image_sub_, 10));
        sync_->registerCallback(boost::bind(&StereoDepth::Callback, this,  _1, _2));
    }

    cv::Ptr<cv::StereoSGBM> CreateStereoSgbm(int channels)
    {
        int block_size = sgbm_config_.block_size;
        cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
            sgbm_config_.min_disparity,
            sgbm_config_.num_disparities * 16,
            block_size,
            sgbm_config_.p1 * channels * block_size * block_size,
            sgbm_config_.p2 * channels * block_size * block_size,
            sgbm_config_.disp12_max_diff,
            sgbm_config_.pre_filter_cap,
            sgbm_config_.uniqueness_ratio,
            sgbm_config_.speckle_window_size,
            sgbm_config_.speckle_range,
            sgbm_config_.mode);

        return sgbm;
    }

    void DynamicReconfigureCallback(stereo_depth::DisparityConfig &config, uint32_t level)
    {
        ROS_INFO("Setting dyn_config");
        sgbm_config_ = config;
    }

    void CameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& info)
    {
        if (!camera_info_.count(info->header.frame_id))
        {
            camera_info_[info->header.frame_id] = *info;
        }
    }

    void SetQ(const std::string& left_frame, const std::string& right_frame)
    {
        if (camera_info_.size() == 2 && Q_.at<double>(0,0) == 0)
        {
            const auto& K1 = camera_info_[left_frame].K;
            const auto& K2 = camera_info_[right_frame].K;

            Q_.at<double>(0,0) = 1.0;
            Q_.at<double>(0,3) = -K1.at(2) * scale_; //cx
            Q_.at<double>(1,1) = 1.0;
            Q_.at<double>(1,3) = -K1.at(5) * scale_;  //cy
            Q_.at<double>(2,3) = K1.at(0) * scale_;  //Focal
            Q_.at<double>(3,2) = 1.0 / baseline_;    //1.0/BaseLine
            Q_.at<double>(3,3) = (K1.at(2) - K2.at(2)) * scale_ / baseline_;    //cx - cx'
        }
    }

    void PointCloudFromStereo(
        const cv::Mat& left_mat,
        const cv::Mat& right_mat,
        cv::Mat& point_cloud)
    {
        cv::Mat disparity;
        cv::Mat normalized_disparity;

        auto sgbm = CreateStereoSgbm(left_mat.channels());
        sgbm->compute(left_mat, right_mat, disparity);

        disparity.convertTo(normalized_disparity, CV_32F, 1.0/16.0, 0.0);

        //Project disparity map to 3d points
        cv::reprojectImageTo3D(normalized_disparity, point_cloud, Q_, true);
    }

    void Pcl2FromCvMat(
        const cv::Mat& point_cloud,
        sensor_msgs::PointCloud2& cloud_msg,
        const cv::Mat* image = nullptr)
    {
        cloud_msg.width = point_cloud.cols;
        cloud_msg.height = point_cloud.rows;

        cloud_msg.is_dense = false;
        cloud_msg.is_bigendian = false;

        sensor_msgs::PointCloud2Modifier pcd_modifier(cloud_msg);
        pcd_modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");

        PCIteratorf ix(cloud_msg, "x");
        PCIteratorf iy(cloud_msg, "y");
        PCIteratorf iz(cloud_msg, "z");
        PCIteratori ir(cloud_msg, "r");
        PCIteratori ig(cloud_msg, "g");
        PCIteratori ib(cloud_msg, "b");

        const cv::Vec3f *pv;
        const uint8_t *gray;

        for (int v = 0; v < point_cloud.rows; v++)
        {
            pv = point_cloud.ptr<cv::Vec3f>(v);
            gray = image ? image->ptr<uint8_t>(v) : nullptr;
            for (int u = 0; u < point_cloud.cols; u++, ++ix, ++iy, ++iz, ++ir, ++ig, ++ib)
            {
                const cv::Vec3f& value = pv[u];
                *ix = value[0];
                *iy = value[1];
                *iz = value[2];

                if (image)
                {
                    // If image is provided, set color values
                    uint8_t v = gray[u];
                    *ir = v;
                    *ig = v;
                    *ib = v;
                }
            }
        }
    }

    void Callback(
        const sensor_msgs::ImageConstPtr& left_image,
        const sensor_msgs::ImageConstPtr& right_image)
    {
        if (camera_info_.size() != 2)
        {
            ROS_WARN_THROTTLE(2, "Don't have all CameraInfos");
            return;
        }

        cv::Mat point_cloud;
        sensor_msgs::PointCloud2 cloud_msg;

        cv_bridge::CvImagePtr left_cv_ptr = cv_bridge::toCvCopy(left_image, left_image->encoding);
        cv_bridge::CvImagePtr right_cv_ptr = cv_bridge::toCvCopy(right_image, right_image->encoding);
        SetQ(left_image->header.frame_id, right_image->header.frame_id);

        cv::Mat im1, im2;
        if (scale_ != 1.0)
        {
            cv::resize(left_cv_ptr->image, im1, cv::Size(), scale_, scale_, cv::INTER_LINEAR);
            cv::resize(right_cv_ptr->image, im2, cv::Size(), scale_, scale_, cv::INTER_LINEAR);
        } else
        {
            im1 = left_cv_ptr->image;
            im2 = right_cv_ptr->image;
        }

        PointCloudFromStereo(im1, im2, point_cloud);
        Pcl2FromCvMat(point_cloud, cloud_msg, &im1);

        cloud_msg.header = left_image->header;
        cloud_publisher_.publish(cloud_msg);
    }

}; //Class

int main(int argc, char **argv)
{
    ros::init(argc, argv, "stereo_to_depth");
    StereoDepth depth_node;

    ros::spin();

    return 0;
}
