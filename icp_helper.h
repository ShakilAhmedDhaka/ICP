#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/keypoints/sift_keypoint.h>

#include <pcl/features/range_image_border_extractor.h>
#include <pcl/visualization/range_image_visualizer.h>
#include <pcl/range_image/range_image_planar.h>
#include <pcl/visualization/range_image_visualizer.h>

#include <pcl/keypoints/narf_keypoint.h>

#include <pcl/features/fpfh.h>
#include <pcl/features/shot.h>

#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_distance.h>
#include <pcl/registration/correspondence_rejection_features.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/transformation_estimation_svd_scale.h>

#include "math.h"

#include "visualizer.h"

#include <chrono>

#define NEIGHBOUR_SEARCH_RADIUS 6



void removeNanInfPoints(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud2)
{
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud3(new pcl::PointCloud<pcl::PointXYZRGB>());
	for (int i = 0; i < cloud->size(); i++)
	{
		if (cloud->at(i).x == NAN ||
			cloud->at(i).y == NAN ||
			cloud->at(i).z == NAN
			)
		{

		}
		else
		{

			cloud3->points.push_back(cloud->at(i));
		}
	}

	cloud3->height = 1;
	cloud3->width = cloud3->points.size();
	cloud3->is_dense = true;


	cloud2.reset(new pcl::PointCloud<pcl::PointXYZRGB>(*cloud3));
}


void removeNanInfNomals(pcl::PointCloud<pcl::Normal>::Ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr cloud2)
{
	pcl::PointCloud<pcl::Normal>::Ptr cloud3(new pcl::PointCloud<pcl::Normal>());
	for (int i = 0; i < cloud->size(); i++)
	{
		if (cloud->at(i).normal_x == NAN ||
			cloud->at(i).normal_y == NAN ||
			cloud->at(i).normal_z == NAN
			)
		{

		}
		else
		{

			cloud3->points.push_back(cloud->at(i));
		}
	}

	cloud3->height = 1;
	cloud3->width = cloud3->points.size();
	cloud3->is_dense = true;


	cloud2.reset(new pcl::PointCloud<pcl::Normal>(*cloud3));
}



void makePointsNanForNormalNan(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr normals)
{
	for (int i = 0; i < cloud->size(); i++)
	{
		if (std::isinf(normals->at(i).normal_x) ||
			std::isinf(normals->at(i).normal_y) ||
			std::isinf(normals->at(i).normal_z))
		{
			cloud->at(i).x = NAN;
			cloud->at(i).y = NAN;
			cloud->at(i).z = NAN;
		}
	}
}


int handleNanDescriptors(pcl::PointCloud<pcl::SHOT1344>::Ptr shotColorDescriptors)
{
	int count = 0;
	for (int i = 0; i < shotColorDescriptors->size(); i++)
	{
		bool isInf = false;
		for (int j = 0; j < shotColorDescriptors->at(i).descriptorSize(); j++)
		{
			if (i == 874 && j == 0)
			{
				std::cout << shotColorDescriptors->at(i).descriptor[0] << std::endl;
			}
			if (!pcl_isfinite(shotColorDescriptors->at(i).descriptor[j]))
			{
				isInf = true;
				break;
			}
		}
		for (int j = 0; j < 9 && !isInf; j++)
		{
			if (!pcl_isfinite(shotColorDescriptors->at(i).rf[j]))
			{
				isInf = true;
				break;
			}
		}

		if (isInf)
		{
			count++;
			for (int j = 0; j < shotColorDescriptors->at(i).descriptorSize(); j++)
			{
				shotColorDescriptors->at(i).descriptor[j] = 0;
			}
			for (int j = 0; j < 9; j++)
			{
				shotColorDescriptors->at(i).rf[j] = 0;
			}
		}

	}

	return count;
}




void pairError(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud1,
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud2,
	pcl::CorrespondencesPtr correspondences,
	double& error, double& totalError)
{
	error = 0.0f;
	double err = 0.0f, count = 0.0f;

	for (int i = 0; i < correspondences->size(); i++)
	{
		int source_index = correspondences->at(i).index_query;
		int target_index = correspondences->at(i).index_match;

		if (source_index != -1 && target_index != -1)
		{
			pcl::PointXYZRGB p1 = cloud1->at(source_index);
			pcl::PointXYZRGB p2 = cloud2->at(target_index);

			err = (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z);
			if (err >= 0)
			{
				error += sqrt(err);
				count++;
			}
		}


	}

	totalError = error;
	error = error / count;
}



double computeCloudResolution(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr& cloud)
{
	double res = 0.0;
	int n_points = 0;
	int nres;
	std::vector<int> indices(2);
	std::vector<float> sqr_distances(2);
	pcl::search::KdTree<pcl::PointXYZRGB> tree;
	tree.setInputCloud(cloud);

	for (size_t i = 0; i < cloud->size(); ++i)
	{
		if (!pcl_isfinite((*cloud)[i].z))
		{
			continue;
		}
		//Considering the second neighbor since the first is the point itself.
		nres = tree.nearestKSearch(i, 2, indices, sqr_distances);
		if (nres == 2)
		{
			res += sqrt(sqr_distances[1]);
			++n_points;
		}
	}
	if (n_points != 0)
	{
		res /= n_points;
	}
	return res;
}



void getNormalsUnOrganizedCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr normals, double model_resolution)
{
	pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> normalEstimationFromUnOraganizedCloud;
	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZRGB>());
	normalEstimationFromUnOraganizedCloud.setInputCloud(cloud);
	normalEstimationFromUnOraganizedCloud.setSearchMethod(kdtree);
	normalEstimationFromUnOraganizedCloud.setRadiusSearch(NEIGHBOUR_SEARCH_RADIUS * model_resolution);
	normalEstimationFromUnOraganizedCloud.compute(*normals);
}


void getCorrespondences(pcl::PointCloud<pcl::SHOT1344>::Ptr shotColorDescriptors1,
	pcl::PointCloud<pcl::SHOT1344>::Ptr shotColorDescriptors2,
	pcl::CorrespondencesPtr updatedCorrespondences,
	double featureRejectThreshold)
{
	for (int i = 0; i < updatedCorrespondences->size(); i++)
	{
		int source_index = updatedCorrespondences->at(i).index_query;
		int target_index = updatedCorrespondences->at(i).index_match;

		if (source_index != -1 && target_index != -1)
		{
			Eigen::VectorXf sourceVec = Eigen::Map< Eigen::VectorXf>(
				shotColorDescriptors1->at(source_index).descriptor,
				1344);

			Eigen::VectorXf targetVec = Eigen::Map< Eigen::VectorXf>(
				shotColorDescriptors2->at(target_index).descriptor,
				1344);

			double minVecLen = sourceVec.norm() < targetVec.norm() ? sourceVec.norm() : targetVec.norm();
			double diff = (sourceVec - targetVec).norm();
			if (diff > featureRejectThreshold * minVecLen)
			{
				updatedCorrespondences->erase(updatedCorrespondences->begin() + i);
				i--;
			}


		}
	}
}


void getSiftFeatures(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr modelWithColorKeypoints)
{
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudColor(new pcl::PointCloud<pcl::PointXYZRGB>);
	std::vector<int> nindices;
	pcl::removeNaNFromPointCloud(*cloud, *cloudColor, nindices);
	removeNanInfPoints(cloudColor, cloudColor);
	cloudColor->is_dense = false;

	for (int i = 0; i < cloudColor->size(); i++)
	{
		if (std::isinf(cloudColor->at(i).x) ||
			std::isinf(cloudColor->at(i).y) ||
			std::isinf(cloudColor->at(i).z)
			)
		{
			std::cout << "Problem" << std::endl;
		}
	}

	//std::cout << "cloudcolor size: " << cloudColor->size() << std::endl;

	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());

	// Parameters for sift computation
	const float min_scale = 0.1f;
	const int n_octaves = 6;
	const int n_scales_per_octave = 10;
	const float min_contrast = 0.5f;

	pcl::SIFTKeypoint<pcl::PointXYZRGB, pcl::PointXYZRGB> sift;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr result(new pcl::PointCloud<pcl::PointXYZRGB>);
	sift.setSearchMethod(tree);
	sift.setScales(min_scale, n_octaves, n_scales_per_octave);
	sift.setMinimumContrast(min_contrast);
	sift.setInputCloud(cloudColor);
	sift.compute(*result);

	std::cout << result->size() << std::endl;

	modelWithColorKeypoints->resize(result->width * result->height);
	modelWithColorKeypoints->width = result->width;
	modelWithColorKeypoints->height = result->height;
	modelWithColorKeypoints->is_dense = false;

	for (int i = 0; i < result->size(); i++)
	{
		modelWithColorKeypoints->at(i).x = result->at(i).x;
		modelWithColorKeypoints->at(i).y = result->at(i).y;
		modelWithColorKeypoints->at(i).z = result->at(i).z;
		modelWithColorKeypoints->at(i).r = result->at(i).r;
		modelWithColorKeypoints->at(i).g = result->at(i).g;
		modelWithColorKeypoints->at(i).b = result->at(i).b;

	}

	pcl::removeNaNFromPointCloud(*modelWithColorKeypoints, *modelWithColorKeypoints, nindices);
	std::cout << "No of SIFT points in the result are " << modelWithColorKeypoints->size() << std::endl;
}



void getShotColorDescriptors(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr normals,
	pcl::PointCloud<pcl::SHOT1344>::Ptr shotColorDescriptors, pcl::PointCloud<pcl::PointXYZRGB>::Ptr model_keypoints, double model_resolution)
{
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudColor(new pcl::PointCloud<pcl::PointXYZRGB>);
	std::vector<int> nindices;
	pcl::removeNaNFromPointCloud(*cloud, *cloudColor, nindices);


	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZRGB>());
	pcl::SHOTColorEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::SHOT1344> shotEstimator;

	shotEstimator.setInputCloud(model_keypoints);
	shotEstimator.setInputNormals(normals);
	shotEstimator.setRadiusSearch(NEIGHBOUR_SEARCH_RADIUS * model_resolution);
	shotEstimator.setSearchSurface(cloudColor);
	shotEstimator.compute(*shotColorDescriptors);
}



void downSampleCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr sampledCloud(new pcl::PointCloud<pcl::PointXYZRGB>);

	pcl::VoxelGrid<pcl::PointXYZRGB> sor;
	sor.setInputCloud(cloud);
	sor.setLeafSize(10.0f, 10.0f, 10.0f);
	sor.filter(*sampledCloud);

	*cloud = *sampledCloud;
}


void load_matrices(Eigen::Matrix4f& transformMatrix, std::string path)
{
	std::ifstream in_mat(path);
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			in_mat >> transformMatrix(i, j);
		}
	}
}



// concate char arrays
char* concat(char* ch1, const char* ch2)
{
	char* ch3 = (char*)malloc(1 + strlen(ch1) + strlen(ch2));;
	strcpy(ch3, ch1);
	strcat(ch3, ch2);
	return ch3;
}



std::vector<std::string> get_tokens(std::string str, char* delimeter, int& sz)
{
	int i = 0;
	std::vector<std::string> tokens;

	// stringstream class check1 
	std::stringstream str_stream(str);

	std::string intermediate;
	while (getline(str_stream, intermediate, '/'))
	{
		tokens.push_back(intermediate);
	}

	sz = tokens.size() - 1;

	return tokens;
}



// takes input of a set of files: one element of the set =  one point cloud, rgb file, one feature points file
int take_input(int argc, char** argv, std::vector< pcl::PointCloud<pcl::PointXYZRGB>::Ptr >& clouds, std::vector<std::string>& filenames)
{
	std::cout << "Number of files: " << argc - 1 << std::endl;
	clouds.resize(argc - 1);
	filenames.resize(argc - 1);
	int sz = 0;
	for (int i = 1; i < argc; i++)
	{
		// Taking cloud input either as pcd or ply
		std::string fname_str = std::string(argv[i]);
		char* fname = new char[fname_str.length() + 1];
		strcpy(fname, fname_str.c_str());
		filenames[i - 1] = std::string(get_tokens(fname, "/", sz)[sz]);
		if (boost::filesystem::exists(concat(argv[i], ".pcd")))
		{
			clouds[i - 1] = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
			pcl::io::loadPCDFile(concat(argv[i], ".pcd"), *clouds[i - 1]);
			//cout<<"cloud->width "<<cloud->width<<" cloud->height "<<cloud->height<<endl;
		}
		else if (boost::filesystem::exists(concat(argv[i], ".ply"))) pcl::io::loadPLYFile(concat(argv[i], ".ply"), *clouds[i - 1]);
		else
		{
			std::cout << "input just the name of the file without extension.\nname of pointcloud and jpg file must be same" << std::endl;
			std::cout << "example: ./tags_3d ../outputs/0_604" << std::endl;
			std::cout << "given pointcloud file does not exist" << std::endl;
			return -1;
		}
	}

	return 1;
}
