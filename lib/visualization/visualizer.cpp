#include "visualizer.h"



// static member vairables have to be defined outside of the class and some other main function
int VISH::Visualizer::cloud_picked_index_static = 0;
pcl::PointXYZRGB VISH::Visualizer::cloud_picked_point_static = pcl::PointXYZRGB(0, 0, 0);
float VISH::Visualizer::clicked_point_depth_static = 0;

namespace VISH
{
	Visualizer::Visualizer(boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>> cloud)
	{	
		viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer("Point Cloud Viewer"));
		viewer->setCameraPosition(0.0, 0.0, -5.0, 0.0, -1.0, -1.0, 0);
		viewer->setBackgroundColor(0, 0, 0);
		rgbv = pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>(cloud);

		viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgbv, "rgbv");
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
			3, "rgbv");

		cloud_picked_index_static = 0;
		clicked_point_depth_static = 0;
		cloud_val = *cloud;
	}


	void Visualizer::update_visualizer(boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>> cloud)
	{
		rgbv = pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB>(cloud);
		viewer->updatePointCloud<pcl::PointXYZRGB>(cloud, rgbv, "rgbv");
	}



}