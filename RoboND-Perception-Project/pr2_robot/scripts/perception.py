#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 TODOs:

    # Convert ROS msg to PCL data
    pcl_data = ros_to_pcl(pcl_msg)
    
    # Statistical Outlier Filtering
    outlier_filter = pcl_data.make_statistical_outlier_filter()
    outlier_filter.set_mean_k(50)    # Set threshold scale factor
    outlier_threshold = 0.1
    outlier_filter.set_std_dev_mul_thresh(outlier_threshold)
    pcl_filtered = outlier_filter.filter()

    # Voxel Grid Downsampling
    voxel_filter = pcl_filtered.make_voxel_grid_filter()
    LEAF_SIZE = 0.01
    voxel_filter.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    pcl_filtered = voxel_filter.filter() 

    # PassThrough Filter
    passthrough_filter = pcl_filtered.make_passthrough_filter()
    filter_axis = 'z'
    passthrough_filter.set_filter_field_name(filter_axis)
    axis_min = 0.6
    axis_max = 1.2
    passthrough_filter.set_filter_limits(axis_min, axis_max)
    pcl_filtered = passthrough_filter.filter()

    # filter by Y axis to remove the bin edges
    passthrough_filter = pcl_filtered.make_passthrough_filter()
    filter_axis = 'y'
    passthrough_filter.set_filter_field_name(filter_axis)
    axis_min = -0.5
    axis_max = 0.5
    passthrough_filter.set_filter_limits(axis_min, axis_max)
    pcl_filtered = passthrough_filter.filter()

    # RANSAC Plane Segmentation
    segmenter_filter = pcl_filtered.make_segmenter()
    segmenter_filter.set_model_type(pcl.SACMODEL_PLANE)
    segmenter_filter.set_method_type(pcl.SAC_RANSAC)
    max_distance = 0.02
    segmenter_filter.set_distance_threshold(max_distance)
    inliers, coefficients = segmenter_filter.segment()

    # Extract inliers and outliers
    cloud_table = pcl_filtered.extract(inliers, negative=False)
    cloud_objects = pcl_filtered.extract(inliers, negative=True)

    # Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(cloud_objects)
    tree = white_cloud.make_kdtree()
    ec = white_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.02)
    ec.set_MinClusterSize(10)
    ec.set_MaxClusterSize(20000)
    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)
    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()

    # Create Cluster-Mask Point Cloud to visualize each cluster separately
    cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                             white_cloud[indice][1],
                                             white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])
    #Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    # Convert PCL data to ROS messages
    ros_cloud_objects = pcl_to_ros(cloud_objects)
    ros_cloud_table = pcl_to_ros(cloud_table)
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)

    # Publish ROS messages
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)
    pcl_cluster_pub.publish(ros_cluster_cloud)

# Exercise-3 TODOs:

    # Classify the clusters!
    detected_objects_labels = []
    detected_objects = []

    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster from the extracted outliers (cloud_objects)
        pcl_cluster = cloud_objects.extract(pts_list)
        # convert the cluster from pcl to ROS using helper function
        ros_cluster = pcl_to_ros(pcl_cluster)

        # Extract histogram features
        chists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))

        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))

    # Publish the list of detected objects
    # This is the output you'll need to complete the upcoming project!
    detected_objects_pub.publish(detected_objects)

    # Suggested location for where to invoke your pr2_mover()
    # function within pcl_callback()
    # Could add some logic to determine whether or not your 
    # object detections are robust
    # before calling pr2_mover()

    if len(detected_objects) > 0:
        try:
            pr2_mover(detected_objects)
        except rospy.ROSInterruptException:
            pass


# function to load parameters and request PickPlace service
def pr2_mover(detected_objects_list):

    # TODO: Initialize variables

    # get parameters:
    # 
    # pick list
    object_list_param = rospy.get_param('/object_list')
    object_names = [item['name'] for item in object_list_param]
    object_groups = [item['group'] for item in object_list_param]

    # world; there is no simple way of finding the world_name because it is
    # passed as an argument to the gazebo node; we use the numer of objects
    # to infer the world
    world_dict = {3:1, 5:2, 8:3}
    try:
        test_scene_num_msg = Int32()
        test_scene_num_msg.data = world_dict[len(object_list_param)]
        # also create the name of the yaml output file
        yaml_file = 'output_'+str(world_dict[len(object_list_param)])+'.yaml'
    except KeyError:
        rospy.logwarn('Scene cannot be determined for a pick list with {} objects.'.format(len(object_list_param)))
        return      

    # dropboxes
    dropbox_param = rospy.get_param('/dropbox')
    dropboxes = {}
    for item in dropbox_param:
        dropboxes[item['group']] = item
    
    # prepare the command list for yaml output
    command_list = []
    
    # TODO: Rotate PR2 in place to capture side tables for the collision map

    # Loop through the pick list
    for index, label in enumerate(object_names):

        # Get the PointCloud for a given object and obtain it's centroid
        try:
            detected_object = next((obj for obj in detected_objects_list if obj.label == label))
        except StopIteration:
            # the requested object in the pick list does not exist in
            # the detected objects; issue warning and move to the next in 
            # pick list
            rospy.logwarn('Object: {} from pick list was not detected.'.format(label))
            continue

        # calculate the centroid
        points_arr = ros_to_pcl(detected_object.cloud).to_array()
        centroid = np.mean(points_arr, axis=0)[:3]
        pick_pose = Pose()
        pick_pose.position.x = np.asscalar(centroid[0])
        pick_pose.position.y = np.asscalar(centroid[1])
        pick_pose.position.z = np.asscalar(centroid[2])
        
        # Create 'place_pose' for the object
        place_pose = Pose()
        dropbox_pos = dropboxes[object_groups[index]]['position']
        print('dropbox position: '+str(dropbox_pos))
        place_pose.position.x = dropbox_pos[0]
        place_pose.position.y = dropbox_pos[1]
        place_pose.position.z = dropbox_pos[2]

        # Assign the arm to be used for pick_place
        which_arm_msg = String()
        which_arm_msg.data = dropboxes[object_groups[index]]['name']
        
        # Object name message
        object_name_msg = String()
        object_name_msg.data = label

        # Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
        command_yaml = make_yaml_dict(test_scene_num_msg, 
                                      which_arm_msg,
                                      object_name_msg,
                                      pick_pose,
                                      place_pose)
        command_list.append(command_yaml)

        # Wait for 'pick_place_routine' service to come up

        rospy.wait_for_service('pick_place_routine')

        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)
            print('Calling pick_place_routine with:')
            print('    scenene number: '+str(test_scene_num_msg.data))
            print('    object name   : '+str(object_name_msg.data))
            print('    which arm     : '+str(which_arm_msg.data))
            print('    pick pose     : '+str(pick_pose.position))
            print('    place pose    : '+str(place_pose.position))
            resp = pick_place_routine(test_scene_num_msg,
                                      object_name_msg,
                                      which_arm_msg,
                                      pick_pose,
                                      place_pose)

            print ("Response: ",resp.success)

        except rospy.ServiceException, e:
            print "Service call failed: %s"%e
            
    # Output your request parameters into output yaml file
    send_to_yaml(yaml_file, command_list)
    rospy.loginfo('YAML file saved: {}.'.format(yaml_file))



if __name__ == '__main__':

    # ROS node initialization
    rospy.init_node("perception", anonymous=True)

    # Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", 
                               pc2.PointCloud2, 
                               pcl_callback, 
                               queue_size=1)

    # Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)

    # Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()

