from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    return LaunchDescription([

        #  DBSCAN 
        DeclareLaunchArgument(
            'eps', default_value='0.7',
            description='DBSCAN neighbourhood radius (metres)',
        ),
        DeclareLaunchArgument(
            'min_samples', default_value='5',
            description='DBSCAN minimum points to form a core point',
        ),
        DeclareLaunchArgument(
            'min_cluster_size', default_value='22',
            description='Discard clusters smaller than this',
        ),
        DeclareLaunchArgument(
            'max_cluster_size', default_value='2000',
            description='Discard clusters larger than this (e.g. ground plane bleed)',
        ),

        #  Depth filter
        DeclareLaunchArgument(
            'depth_filter_method', default_value='percentile',
            description='Depth filter: jump | percentile',
        ),
        DeclareLaunchArgument(
            'jump_threshold', default_value='0.8', #1.3
            description='[jump] Depth gap in metres that triggers a cut',
        ),
        DeclareLaunchArgument(
            'depth_percentile', default_value='30.0', #30.0
            description='[percentile] Near-surface percentile to anchor to',
        ),
        DeclareLaunchArgument(
            'depth_tolerance', default_value='1.3',
            description='[percentile] Multiplier on near-percentile depth',
        ),


        DeclareLaunchArgument(
            'publish_cluster_cloud', default_value='true',
            description='Publish coloured cluster cloud for RViz debug',
        ),
        DeclareLaunchArgument(
            'cluster_merge_distance', default_value='1.0',
            description='Max box surface-to-surface gap (m) to merge same-class clusters; 0 = disabled',
        ),
    
        #  DBSCAN node 
        Node(
            package='dbscan_clustering',
            executable='dbscan_node',
            name='dbscan_node',
            output='screen',
            parameters=[{
                'use_sim_time':          True,
                'eps':                   LaunchConfiguration('eps'),
                'min_samples':           LaunchConfiguration('min_samples'),
                'min_cluster_size':      LaunchConfiguration('min_cluster_size'),
                'max_cluster_size':      LaunchConfiguration('max_cluster_size'),
                'depth_filter_method':   LaunchConfiguration('depth_filter_method'),
                'jump_threshold':        LaunchConfiguration('jump_threshold'),
                'depth_percentile':      LaunchConfiguration('depth_percentile'),
                'depth_tolerance':       LaunchConfiguration('depth_tolerance'),
                'publish_cluster_cloud':    LaunchConfiguration('publish_cluster_cloud'),
                'cluster_merge_distance':   LaunchConfiguration('cluster_merge_distance'),
            }],
            remappings=[
                ('/fused/cloud_in_boxes', '/fused/cloud_in_boxes'),
            ],
        ),

    ])