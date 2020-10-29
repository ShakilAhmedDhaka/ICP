#include <iostream>
#include <vector>

#include "icp_helper.h"

using std::cout;


int main(int argc, char* argv[])
{
	std::vector< pcl::PointCloud<pcl::PointXYZRGB>::Ptr > clouds;
	std::vector<std::string> filenames;
	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
	// TAKING INPUT
	// argument 1 is the reference frame
	std::cout << "Taking input:" << std::endl;
	if (take_input(argc, argv, clouds, filenames) == -1) return 0;
	cout << "input taken\n";

	for (int kk = 1; kk < clouds.size(); kk++)
	{
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudColor1(new pcl::PointCloud<pcl::PointXYZRGB>);
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudColor2(new pcl::PointCloud<pcl::PointXYZRGB>);

		cloudColor1.reset(new pcl::PointCloud<pcl::PointXYZRGB>(*clouds[kk - 1]));
		cloudColor2.reset(new pcl::PointCloud<pcl::PointXYZRGB>(*clouds[kk]));

		cloudColor1->is_dense = false;
		cloudColor2->is_dense = false;


		for (int i = 0; i < cloudColor1->height; i++)
		{
			for (int j = 0; j < cloudColor1->width; j++)
			{

				if (cloudColor1->at(j, i).z < 10)
				{
					cloudColor1->at(j, i).x = NAN;
					cloudColor1->at(j, i).y = NAN;
					cloudColor1->at(j, i).z = NAN;

				}

				if (cloudColor2->at(j, i).z < 10)
				{
					cloudColor2->at(j, i).x = NAN;
					cloudColor2->at(j, i).y = NAN;
					cloudColor2->at(j, i).z = NAN;
				}
			}
		}

		std::cout << "points copied\n";

		// creating another cloud without nan values
		std::vector<int> nindices;
		pcl::removeNaNFromPointCloud(*cloudColor1, *cloudColor1, nindices);
		pcl::removeNaNFromPointCloud(*cloudColor2, *cloudColor2, nindices);

		downSampleCloud(cloudColor1);
		downSampleCloud(cloudColor2);
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr viewerCloud10(new pcl::PointCloud<pcl::PointXYZRGB>);
		viewerCloud10.reset(new pcl::PointCloud<pcl::PointXYZRGB>(*cloudColor1));
		*viewerCloud10 += *cloudColor2;
		VISH::Visualizer visualizer10(viewerCloud10);
		visualizer10.viewer->spin();


		// Compute model_resolution (distance from the closest neighbour on avarage)
		double model_resolution1 = 0, model_resolution2 = 0;
		model_resolution1 = std::max(12.0, computeCloudResolution(cloudColor1));
		model_resolution2 = std::max(12.0, computeCloudResolution(cloudColor2));
		std::cout << "model resolution1: " << model_resolution1 << "; model_resolution2: " << model_resolution2 << std::endl;

		std::vector< Eigen::Matrix4f> transformationMatrices;
		Eigen::Matrix4f finalTransform = Eigen::Matrix4f::Identity();
		Eigen::Matrix4f acceptedTransform = Eigen::Matrix4f::Identity();
		double ransacCorrespondenceDistance = 60.0f;
		double pastError = 99999999999.99f;
		double pastTotalError = 99999999999.99f;

		std::chrono::steady_clock::time_point beginICP = std::chrono::steady_clock::now();
		int noIter = 15;
		double ransacCorrespondenceDistanceStep = 5.0f / (noIter * 2);

		std::stringstream sents1(argv[kk]);
		std::stringstream sents2(argv[kk + 1]);
		std::string temp;
		std::vector<std::string> tokens1, tokens2;


		while (getline(sents1, temp, '/'))
		{
			tokens1.push_back(temp);
		}

		while (getline(sents2, temp, '/'))
		{
			tokens2.push_back(temp);
		}


		std::string pathPrefix = "../transformation_matrix_";
		std::string path = pathPrefix + tokens1[tokens1.size() - 1] + "_" + tokens2[tokens2.size() - 1] + ".txt";
		
		std::cout << "path: " << path << std::endl;
		ofstream ofile(path);
		// starting pair-wise icp
		for (int i = 0; i < 10; i++)
		{
			std::cout << "ICP loop: " << i << std::endl;

			// computing normals from unorganized point cloud
			pcl::PointCloud<pcl::Normal>::Ptr normalsFromUnOrganizedCloud1(new pcl::PointCloud<pcl::Normal>);
			pcl::PointCloud<pcl::Normal>::Ptr normalsFromUnOrganizedCloud12(new pcl::PointCloud<pcl::Normal>);
			pcl::PointCloud<pcl::Normal>::Ptr normalsFromUnOrganizedCloud2(new pcl::PointCloud<pcl::Normal>);
			pcl::PointCloud<pcl::Normal>::Ptr normalsFromUnOrganizedCloud22(new pcl::PointCloud<pcl::Normal>);
			getNormalsUnOrganizedCloud(cloudColor1, normalsFromUnOrganizedCloud1, model_resolution1);
			getNormalsUnOrganizedCloud(cloudColor2, normalsFromUnOrganizedCloud2, model_resolution2);

			makePointsNanForNormalNan(cloudColor1, normalsFromUnOrganizedCloud1);
			makePointsNanForNormalNan(cloudColor2, normalsFromUnOrganizedCloud2);

			removeNanInfPoints(cloudColor1, cloudColor1);
			removeNanInfPoints(cloudColor2, cloudColor2);

			removeNanInfNomals(normalsFromUnOrganizedCloud1, normalsFromUnOrganizedCloud1);
			removeNanInfNomals(normalsFromUnOrganizedCloud2, normalsFromUnOrganizedCloud2);


			pcl::PointCloud<pcl::PointXYZRGB>::Ptr modelWithColorKeypoints1(new pcl::PointCloud<pcl::PointXYZRGB>());
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr modelWithColorKeypoints2(new pcl::PointCloud<pcl::PointXYZRGB>());

			// detecting keypoints
			getSiftFeatures(cloudColor1, modelWithColorKeypoints1);
			getSiftFeatures(cloudColor2, modelWithColorKeypoints2);


			// feature correspondence estimation
			pcl::CorrespondencesPtr allCorrespondences(new pcl::Correspondences);
			pcl::CorrespondencesPtr updatedCorrespondences(new pcl::Correspondences);


			// compute shotColor features
			pcl::PointCloud<pcl::SHOT1344>::Ptr shotColorDescriptors1(new pcl::PointCloud<pcl::SHOT1344>());
			pcl::PointCloud<pcl::SHOT1344>::Ptr shotColorDescriptors2(new pcl::PointCloud<pcl::SHOT1344>());
			getShotColorDescriptors(cloudColor1, normalsFromUnOrganizedCloud1, shotColorDescriptors1, modelWithColorKeypoints1, model_resolution1);
			std::cout << "Number of shotColor features computed for cloud1: " << shotColorDescriptors1->size() << std::endl;
			getShotColorDescriptors(cloudColor2, normalsFromUnOrganizedCloud2, shotColorDescriptors2, modelWithColorKeypoints2, model_resolution2);
			std::cout << "Number of shotColor features computed for cloud2: " << shotColorDescriptors2->size() << std::endl;


			int discardedFeatures1 = handleNanDescriptors(shotColorDescriptors1);
			int discardedFeatures2 = handleNanDescriptors(shotColorDescriptors2);

			std::cout << "Discarded Features1: " << discardedFeatures1 << std::endl;
			std::cout << "Discarded Features2: " << discardedFeatures2 << std::endl;

			// feature correspondence estimation
			std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
			pcl::registration::CorrespondenceEstimation<pcl::SHOT1344, pcl::SHOT1344> correspondenceEstimation;
			correspondenceEstimation.setInputSource(shotColorDescriptors1);
			correspondenceEstimation.setInputTarget(shotColorDescriptors2);
			correspondenceEstimation.determineReciprocalCorrespondences(*allCorrespondences);
			std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
			std::cout << "Correspondence between both clouds: " << allCorrespondences->size() << std::endl;
			std::cout << "Correspondence estimation time: " <<
				std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() <<
				"[s]" << std::endl;

			// RANSAC rejection
			if (i > 0)
			{
				begin = std::chrono::steady_clock::now();
				pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZRGB> correspondence_rejector;
				correspondence_rejector.setInputSource(modelWithColorKeypoints1);
				correspondence_rejector.setInputTarget(modelWithColorKeypoints2);
				correspondence_rejector.setInlierThreshold(ransacCorrespondenceDistance);
				correspondence_rejector.setMaximumIterations(100);
				correspondence_rejector.setRefineModel(true);//false
				correspondence_rejector.setInputCorrespondences(allCorrespondences);
				correspondence_rejector.getCorrespondences(*updatedCorrespondences);
				end = std::chrono::steady_clock::now();

				std::cout << "Correspondences after ransac rejector: " << updatedCorrespondences->size() << std::endl;
				std::cout << "RANSAC rejecttion time: " <<
					std::chrono::duration_cast<std::chrono::seconds>(end - begin).count()
					<< "[s]" << std::endl;
			}
			else
			{
				// bad correspondence rejection
				pcl::registration::CorrespondenceRejectorDistance rejDistance;
				rejDistance.setInputSource<pcl::PointXYZRGB>(modelWithColorKeypoints1);
				rejDistance.setInputTarget<pcl::PointXYZRGB>(modelWithColorKeypoints2);
				rejDistance.setMaximumDistance(1500);
				rejDistance.setInputCorrespondences(allCorrespondences);
				rejDistance.getCorrespondences(*updatedCorrespondences);
				std::cout << "Correspondences after distance rejector: " << updatedCorrespondences->size() << std::endl;

				// correspondence rejection based on feture vector
				double featureRejectThreshold = 0.4f; // within 30% of the smaller vector
				getCorrespondences(shotColorDescriptors1, shotColorDescriptors2,
					updatedCorrespondences, featureRejectThreshold);
				std::cout << "Correspondences after feature rejector: " << updatedCorrespondences->size() << std::endl;
			}


			if (updatedCorrespondences->size() > 100)	ransacCorrespondenceDistance -= ransacCorrespondenceDistanceStep;
			if (updatedCorrespondences->size() < 50)	ransacCorrespondenceDistance += ransacCorrespondenceDistanceStep;
			if (updatedCorrespondences->size() < 5)	continue;

			// Obtain the best transformation between the two sets of keypoints given the remaining correspondences
			pcl::registration::TransformationEstimationSVDScale<pcl::PointXYZRGB, pcl::PointXYZRGB> transformation;
			Eigen::Matrix4f transform;

			//transformation.estimateRigidTransformation(*cloud1, *newIndices1, *cloud2, *newIndices2, transform);
			transformation.estimateRigidTransformation(*modelWithColorKeypoints1, *modelWithColorKeypoints2, *updatedCorrespondences, transform);
			std::cout << "Transformation matrix estimated" << std::endl;
			//std::cout << transform << std::endl;


			// transforming point cloud1
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr output(new pcl::PointCloud<pcl::PointXYZRGB>);
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr viewerCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
			pcl::PointCloud<pcl::PointXYZ>::Ptr output2(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::PointCloud<pcl::PointXYZ>::Ptr output3(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::transformPointCloud(*cloudColor1, *output, transform);
			pcl::transformPointCloud(*modelWithColorKeypoints1, *modelWithColorKeypoints1, transform);
			*cloudColor1 = *output;
			double error = 0, totalError = 0;
			pairError(modelWithColorKeypoints1, modelWithColorKeypoints2, updatedCorrespondences, error, totalError);
			std::cout << "Error after icp loop " << i << ": " << error << std::endl;
			ofile << "Error after icp loop " << i << ": " << error << std::endl;

			viewerCloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>(*cloudColor2));
			*viewerCloud += *cloudColor1;
			VISH::Visualizer visualizer1(viewerCloud);
			visualizer1.viewer->spin();

			// if loop breaks on first iter we will still have final transformation matrix
			if (i == 0)	finalTransform = transform;
			transformationMatrices.push_back(transform);
			// updating final transformation matrix
			if (i > 0)
			{
				finalTransform.block(0, 0, 3, 3) = transform.block(0, 0, 3, 3) *
					finalTransform.block(0, 0, 3, 3);
				finalTransform.block(0, 3, 3, 1) = transform.block(0, 0, 3, 3) * finalTransform.block(0, 3, 3, 1) +
					transform.block(0, 3, 3, 1);
			}
			std::cout << finalTransform << std::endl;

			pastError = error;
			pastTotalError = totalError;

			int x;
			std::cout << "Press 0 to break, any other number to continue: ";
			std::cin >> x;
			if (x == 0) break;
		}

		std::chrono::steady_clock::time_point endICP = std::chrono::steady_clock::now();


		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				ofile << finalTransform(i, j) << " ";
			}
			ofile << std::endl;
		}

		std::cout << "Time taken for registration: " <<
			std::chrono::duration_cast<std::chrono::seconds>(endICP - beginICP).count()
			<< "[s]" << std::endl;
		std::cout << "Final Error: " << ": " << pastError << std::endl;
		ofile << "Final average Error: " << ": " << pastError << std::endl;
		ofile << "Final Total Error: " << ": " << pastTotalError << std::endl;

		pcl::PointCloud<pcl::PointXYZRGB>::Ptr viewerCloud2(new pcl::PointCloud<pcl::PointXYZRGB>);
		pcl::transformPointCloud(*clouds[kk - 1], *viewerCloud2, finalTransform);


		*viewerCloud2 += *cloudColor2;
		VISH::Visualizer visualizer2(viewerCloud2);
		visualizer2.viewer->spin();
	}

}