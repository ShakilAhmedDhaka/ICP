#pragma once

#ifndef _VISUALIZER_H_
#define _VISUALIZER_H_

#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/point_cloud.h>


namespace VISH
{
	class Visualizer
	{

	public:

		static int cloud_picked_index_static;
		static pcl::PointXYZRGB cloud_picked_point_static;
		static float clicked_point_depth_static;

		pcl::PointCloud<pcl::PointXYZRGB> cloud_val;
		boost::shared_ptr<pcl::visualization::PCLVisualizer>viewer;

		Visualizer(boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>> cloud);

		void update_visualizer(boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>> cloud);


	private:
		pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgbv;



	};
}

#endif