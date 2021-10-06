from __future__ import division, print_function

import rospy
import tf
import message_filters
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from darknet_ros_msgs.msg import BoundingBox, BoundingBoxes
from sensor_msgs.msg import Image, PointCloud2, LaserScan
from sensor_msgs import point_cloud2 as pc2
from geometry_msgs.msg import PoseStamped

from trt_yolo.detector import DarknetTRT
from utils import timeit_ros

import Queue as queue



class YOLORos(object):
    def __init__(self):
        self._bridge = CvBridge()
        self._read_params()
        self.model = DarknetTRT(
            obj_threshold=self.obj_threshold,
            nms_threshold=self.nms_threshold,
            yolo_type=self.yolo_type,
            weights_path=self.weights_path,
            config_path=self.config_path,
            label_filename=self.label_filename,
            postprocessor_cfg=self.postprocessor_cfg,
            cuda_device=self.cuda_device,
            show_image=self.publish_image,
            IMAGE_PATH=self.IMAGE_PATH
        )
        self._init_topics()
        self.msg_queue = queue.Queue(maxsize=5)
        rospy.loginfo("[trt_yolo_ros] loaded and ready")
        rospy.loginfo("init yolo")
        self.lidar_sub_switch=False
        self.count_tmp = 1
        self.corresponding_seq = 0
        

    def _read_params(self):
        """ Reading parameters for YOLORos from launch or yaml files """
        self.publish_image = rospy.get_param("~publish_image", False)
        self.print_outcome = rospy.get_param("~print_outcome", False)
        # default paths to weights from different sources
        self.weights_path = rospy.get_param("~weights_path", "./weights/")
        self.config_path = rospy.get_param("~config_path", "./config/")
        self.label_filename = rospy.get_param("~label_filename", "coco_labels.txt")
        # parameters of yolo detector
        self.yolo_type = rospy.get_param("~yolo_type", "yolov3-416")
        self.postprocessor_cfg = rospy.get_param(
            "~postprocessor_cfg", "yolo_postprocess_config.json"
        )
        self.obj_threshold = rospy.get_param("~obj_threshold", 0.6)
        self.nms_threshold = rospy.get_param("~nms_threshold", 0.3)
        # default cuda device
        self.cuda_device = rospy.get_param("~cuda_device", 0)
        self.num_cameras = rospy.get_param("~num_cam", 1)
        self.IMAGE_PATH = rospy.get_param("~IMAGE_PATH", "./IMAGE/")
        rospy.logdebug("[trt_yolo_rs]: Number of cameras", self.num_cameras)
        rospy.logdebug("[trt_yolo_ros] parameters read")

    @staticmethod
    def _read_subscriber_param(name):
        """ reading subscriber parameters from launch or yaml files """
        topic = rospy.get_param("~subscriber/" + name + "/topic")
        queue_size = rospy.get_param("~subscriber/" + name + "/queue_size", 10)
        return topic, queue_size

    @staticmethod
    def _read_publisher_param(name):
        """ reading publisher parameters from launch or yaml files """
        topic = rospy.get_param("~publisher/" + name + "/topic")
        queue_size = rospy.get_param("~publisher/" + name + "/queue_size", 1)
        latch = rospy.get_param("~publisher/" + name + "/latch", False)
        return topic, queue_size, latch

    def _init_topics(self):
        """ This function is initializing node publisher and subscribers for the node """
        # Publisher
        topic, queue_size, latch = self._read_publisher_param("bounding_boxes")
        self._pub = rospy.Publisher(
            topic, BoundingBoxes, queue_size=queue_size, latch=latch
        )

        topic, queue_size, latch = self._read_publisher_param("image")
        self._pub_viz = rospy.Publisher(
            topic, Image, queue_size=queue_size, latch=latch
        )

        self._pub_object = rospy.Publisher(
            '/detectedObject', PoseStamped, queue_size=10
        )

        self._pub_object_scan = rospy.Publisher(
            '/objectScan', LaserScan, queue_size=10
        )

        # Image Subscriber
        for i in range(self.num_cameras):
            topic, queue_size = self._read_subscriber_param("image")
            # self._image_sub = rospy.Subscriber(
            #     topic,
            #     Image,
            #     self._image_callback,
            #     queue_size=queue_size,
            #     buff_size=2 ** 24,
            # )
            self._image_sub = message_filters.Subscriber(topic,Image)
        
        rospy.logdebug("[trt_yolo_ros] publishers and subsribers initialized")
        # self.pointcloud_sub = rospy.Subscriber("/camera/depth/color/points",PointCloud2,self._pointcloud_callback,queue_size=20)
        self.lidar_sub = message_filters.Subscriber("/scan",LaserScan)
        self.pose_sub = message_filters.Subscriber("/mavros/local_position/pose", PoseStamped)

        self.mf = message_filters.ApproximateTimeSynchronizer([self._image_sub, self.lidar_sub, self.pose_sub], 5,0.1)
        self.mf.registerCallback(self.mf_callback)
        self.lidar_data = LaserScan()
        self._tfpub = tf.TransformBroadcaster()
        self._tfsub = tf.TransformListener()

        # (self._trans, self._rot) = self._tfsub.lookupTransform('/base_link','/camera_link',rospy.Time())
        # self.transMat = self._tfsub.fromTranslationRotation(self._trans,self._rot)
        self.transMat = np.array([[0,1,0,2],[1,0,0,5],[0,0,1,0],[0,0,0,1]]) # Instead of static tf, Use transformation using numpy

    def mf_callback(self,image, pointcloud, pose):
        self._lidar_callback(pointcloud)
        self._pose_callback(pose)
        self._image_callback(image)

    def _pose_callback(self, msg):
        self.corresponding_seq = msg.header.seq

    def _lidar_callback(self,msg):
        self.lidar_data = msg        
        if self.lidar_sub_switch is False:
            self.lidar_sub_switch = True

    def _image_callback(self, msg):
        """ Main callback which is saving the last received image """
        if msg.header is not None:
            self.msg_queue.put(msg)
            rospy.logdebug("[trt_yolo_ros] image recieved")

    def extractingObjectFromLidar(self, lidar_data,left,right):
        start_index = 460-(left//4)
        end_index = 640-(right//4)
        objectRange = lidar_data.ranges[start_index:end_index]
        object_data = LaserScan()
        object_data.header = lidar_data.header
        object_data.angle_min = (start_index*(np.pi/720))-((3/4)*np.pi)
        object_data.angle_max = (end_index*(np.pi/720))-((3/4)*np.pi)
        object_data.angle_increment = lidar_data.angle_increment
        object_data.time_increment = lidar_data.time_increment
        object_data.scan_time = lidar_data.scan_time
        object_data.range_min = lidar_data.range_min
        object_data.range_max = lidar_data.range_max
        object_data.ranges = objectRange
        object_data.intensities = []
        self._pub_object_scan.publish(object_data)

        mid_index = (start_index+end_index)//2
        mid_range = lidar_data.ranges[mid_index]
        mid_angle = (mid_index*(np.pi/720))-((3/4)*np.pi)
        object_x = mid_range*np.cos(mid_angle)
        object_y = mid_range*np.sin(mid_angle)

        return [object_x, object_y]

    def _write_message(self, detection_results, boxes, scores, classes):
        """ populate output message with input header and bounding boxes information """
        if boxes is None:
            return None
        if self.print_outcome:
            rospy.loginfo("pc_list len = " + str(len(boxes)))
        for box, score, category in zip(boxes, scores, classes):
            # Populate darknet message
            left, bottom, right, top = box
            detection_msg = BoundingBox()
            detection_msg.xmin = left
            detection_msg.xmax = right
            detection_msg.ymin = top
            detection_msg.ymax = bottom
            detection_msg.probability = score
            detection_msg.Class = category
            detection_results.bounding_boxes.append(detection_msg)
            y_center = int(bottom - ((bottom - top) / 2))
            x_center = int(right - ((right - left) / 2))       
            if self.print_outcome:
                rospy.loginfo("x_center = " + str(x_center))
                rospy.loginfo("y_center = " + str(y_center))

            if self.lidar_sub_switch is True:  
                rospy.loginfo("received")           
                rospy.loginfo("Extract the object position using LiDAR")                     
                # rospy.loginfo("pointcloud data size  = " + str(len(self.lidar_data.data)))
                # rospy.loginfo("pointcloud row_step  = " + str((self.lidar_data.row_step)))
                # rospy.loginfo("pointcloud point_step  = " + str((self.lidar_data.point_step)))
                # rospy.loginfo("count_tmp  = " + str((self.count_tmp)))
                # if self.count_tmp > self.lidar_data.row_step:
                #     self.count_tmp = 0
                # pc_list = list(pc2.read_points(self.lidar_data,skip_nans=True,field_names=('x', 'y', 'z'),uvs=[(x_center, y_center)]))                
                # self.count_tmp=self.count_tmp+1
                pc_list = self.extractingObjectFromLidar(self.lidar_data,left,right)
                rospy.loginfo("Extraction Complete... ")
                if len(pc_list) > 0:                    
                    obj_pose_x, obj_pose_y= pc_list
                    obj_pose_z = 0
                    rospy.loginfo("x  = " + str(obj_pose_x)+ "  y  = " + str(obj_pose_y)+"  z  = " + str(obj_pose_z))                    
                    object_tf =  [obj_pose_x, obj_pose_y, obj_pose_z] 
                    tf_id = str(category)                    
                    self._tfpub.sendTransform((object_tf),
                                                    tf.transformations.quaternion_from_euler(
                                                        0, 0, 0),
                                                    rospy.Time.now(),
                                                    tf_id,
                                                    'camera_link')
                                                    
                    self.object_position = self.transMat.dot(np.array([obj_pose_x, obj_pose_y, obj_pose_z, 1]))
                    self.object_pose = PoseStamped()
                    self.object_pose.header.seq = self.corresponding_seq
                    self.object_pose.header.timestamp = rospy.Time.now()
                    self.object_pose.header.frame_id = tf_id
                    self.object_pose.pose.position.x = self.object_position[0]
                    self.object_pose.pose.position.y = self.object_position[1]
                    self.object_pose.pose.position.z = self.object_position[2]
                    self.object_pose.pose.orientation.x = 0.0
                    self.object_pose.pose.orientation.y = 0.0
                    self.object_pose.pose.orientation.z = 0.0
                    self.object_pose.pose.orientation.w = 1.0

                    self._pub_object.publish(self.objet_pose)
        return detection_results

    @timeit_ros
    def process_frame(self):
        """ Main function to process the frame and run the infererence """
        # Deque the next image msg
        current_msg = self.msg_queue.get()
        current_image = None
        # Convert to image to OpenCV format
        try:
            current_image = self._bridge.imgmsg_to_cv2(current_msg, "bgr8")
            rospy.logdebug("[trt_yolo_ros] image converted for processing")
        except CvBridgeError as e:
            rospy.logdebug("Failed to convert image %s", str(e))
        # Initialize detection results
        if current_image is not None:
            rospy.logdebug("[trt_yolo_ros] processing frame")
            boxes, classes, scores, visualization = self.model(current_image)
            detection_results = BoundingBoxes()
            detection_results.header = current_msg.header
            detection_results.image_header = current_msg.header
            # construct message
            self._write_message(detection_results, boxes, scores, classes)
            #######################################################TEST
            # if self.depth_sub_switch is True:  
            #     rospy.loginfo("received")                                
            #     rospy.loginfo("pointcloud data size  = " + str(len(self.pointcloud_data.data)))
            #     rospy.loginfo("pointcloud row_step  = " + str((self.pointcloud_data.row_step)))
            #     rospy.loginfo("pointcloud point_step  = " + str((self.pointcloud_data.point_step)))
            #     rospy.loginfo("count_tmp  = " + str((self.count_tmp)))
            #     if self.count_tmp > self.pointcloud_data.row_step:
            #         self.count_tmp = 0
            #     pc_list = list(pc2.read_points(self.pointcloud_data,skip_nans=True,field_names=('x', 'y', 'z'),uvs=[(self.count_tmp, 0)]))                
            #     self.count_tmp=self.count_tmp+1
            #     rospy.loginfo("pc_list len = " + str(len(pc_list)))
            #     if len(pc_list) > 0:                    
            #         obj_pose_x, obj_pose_y, obj_pose_z = pc_list[0]
            #         rospy.loginfo("x  = " + str(obj_pose_z)+ "  y  = " + str(-obj_pose_x)+"  z  = " + str(-obj_pose_y))                    
            #         object_tf =  [obj_pose_z, -obj_pose_x, -obj_pose_y]      
            # #######################################################TEST              
            # send message
            try:
                rospy.logdebug("[trt_yolo_ros] publishing")
                self._pub.publish(detection_results)
                
                if self.publish_image:
                    self._pub_viz.publish(
                        self._bridge.cv2_to_imgmsg(visualization, "bgr8")
                    )
            except CvBridgeError as e:
                rospy.logdebug("[trt_yolo_ros] Failed to convert image %s", str(e))

