#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.


from __future__ import print_function

import glob
import os
import sys
import time

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from carla import ColorConverter as cc

import argparse
import logging
import random
import random
import cv2
import torch

from model import Model
from queue import Queue
from copy import deepcopy
from utils import cls_to_color

try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

NUM_VEHICLE = 30            # Number of vehicles to be spawned
NUM_PEDES = 10              # Number of pedestrians to be spawned
SPAWN_INDEX = 0             # Index in which the simulator determine to spawn ego vehicle
DISTANCE_MAX = 100          # Max distance that showing the following distance
VIEW_FOV = 90               # Camera view FOV


IOU_THRESHOLD = 0.4         
SMOOTH = 1e-6
CARHOOD_HEIGHT = 20     # Used to filter incorrect bounding box
COLOR_DICT = {
    1: (128, 64, 128),      # Street/Lane/Road
    7: (250, 170, 30),       # Traffic Light
    8: (220, 220, 0),       # Traffic Sign
    12: (220, 20, 60),      # Pedestrian
    13: (255, 0, 0),        # Rider
    14: (0, 0, 142),        # Car
    15: (0, 0, 70),         # Truck
    16: (0, 60, 100),       # Bus
    18: (0, 0, 230),        # Motorcycle
    19: (119, 11, 32),      # Bicycle
    24: (157, 234, 50)     # Lane Line / Markings
}
REMAP_TAG = {
    0: 7,   # Traffic Light
    1: 12,  # Pedestrians
    2: 8,   # Traffic Sign
    3: 13,  # Rider
    4: 14,  # Car
    5: 15,  # Truck
    6: 16,  # Bus
    7: 18,  # Motorcycle
    8: 19,  # Bycicle
}
CARLA_TO_MODEL = {
    7: 4,   # Traffic Light
    8: 5,   # Traffic Sign
    12: 0,  # Pedestrian
    13: 1,  # Rider = Rider
    14: 2,  # Car = Big vehicle
    15: 2,  # Truck = Big vehicle
    16: 2,  # Bus = Big vehicle
    18: 1,  # Motorbike = Small vehicle
    19: 1,  # Bicycle = Small vehicle
}


class SyncWorld(object):
    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.senrors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 30)
        self._queues = []
        self._settings = None

    def __enter__(self):
        def make_queue(register_event):
            q = Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.senrors:
            make_queue(sensor.listen)
        return self
    
    def tick(self, timeout):
        self.frame = self.world.tick()  # self.world.wait_for_tick() is also correct
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data
    
    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


class ClientSideBoundingBoxes(object):
    """
    This is a module responsible for creating 3D bounding boxes and drawing them
    client-side on pygame surface.
    """

    @staticmethod
    def get_bounding_boxes(vehicles, distance_to_player, world_2_camera, K, width, height):
        """
        Creates 3D bounding boxes based on carla vehicle list and camera.
        """

        # Transform 3D BBox to 2D BBox
        candidate_bboxes = []
        for i in range(len(vehicles)):
            vehicle = vehicles[i][0]
            cls = vehicles[i][1]
            vertices = [v for v in vehicle.get_world_vertices(carla.Transform())]
            x_max = -10000
            x_min = 10000
            y_max = -10000
            y_min = 10000

            for vert in vertices:
                p = ClientSideBoundingBoxes.get_image_point(vert, K, world_2_camera)
                if p[0] > x_max:
                    x_max = p[0]
                if p[0] < x_min:
                    x_min = p[0]
                if p[1] > y_max:
                    y_max = p[1]
                if p[1] < y_min:
                    y_min = p[1]

            if y_min > CARHOOD_HEIGHT and x_max > 0 and y_max > 0 and x_min < width and \
                y_min < height:   # Filtering based on CARHOOD and BBox's size
                if x_min < x_max and y_min < y_max:
                    if x_min < 0:
                        x_min = 0
                    if y_min < 0:
                        y_min = 0
                    if x_max >= width:
                        x_max = width
                    if y_max >= height:
                        y_max = height
                    w = x_max - x_min
                    h = y_max - y_min
                    if h > 10 and w > 10:
                        candidate_bboxes.append([int(x_min), int(y_min), int(x_max), int(y_max),
                                                cls, distance_to_player[i]])

        # Filtering occluded BBox based on distance
        remove_idx = []
        try:
            for i, bbox_1 in enumerate(candidate_bboxes):
                for j, bbox_2 in enumerate(candidate_bboxes):
                    if i == j:
                        continue
                    if bbox_1[0] < bbox_2[0] and bbox_1[1] < bbox_2[1] and \
                    bbox_1[2] > bbox_2[2] and bbox_1[3] > bbox_2[3] and bbox_1[5] < bbox_2[5] - 5:
                        remove_idx.append(j)
            
            picked_bboxes = []
            for k, bbox in enumerate(candidate_bboxes):
                if k in remove_idx:
                    continue
                picked_bboxes.append(candidate_bboxes[k])
        except Exception as e:
            print(e)
        return picked_bboxes

    @staticmethod
    def draw_bounding_boxes(display, bounding_boxes, font, width, height):
        """
        Draws bounding boxes on pygame display.
        """
        try:
            bb_surface = pygame.Surface((width, height))
            bb_surface.set_colorkey((0, 0, 0))
            for bbox in bounding_boxes:
                w = int(bbox[2] - bbox[0])
                h = int(bbox[3] - bbox[1])
                pygame.draw.rect(bb_surface, (255, 0, 0), pygame.Rect(int(bbox[0]), int(bbox[1]), w, h), 2)
                display.blit(font.render('% 1.d' % (int(bbox[5])), True, (0, 255, 0), (0, 0, 0)), (int(bbox[0]), int(bbox[1] - 15)))

            display.blit(bb_surface, (0, 0))
        except Exception as e:
            print(e)

    @staticmethod
    def get_image_point(loc, K, w2c):
            # Calculate 2D projection of 3D coordinate

            # Format the input coordinate (loc is a carla.Position object)
            point = np.array([loc.x, loc.y, loc.z, 1])
            # transform to camera coordinates
            point_camera = np.dot(w2c, point)

            # New we must change from UE4's coordinate system to an "standard"
            # (x, y ,z) -> (y, -z, x)
            # and we remove the fourth componebonent also
            point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

            # now project 3D->2D using the camera matrix
            point_img = np.dot(K, point_camera)
            # normalize
            point_img[0] /= point_img[2]
            point_img[1] /= point_img[2]

            return point_img[0:2]


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


def convert_labels(image):
    out = np.zeros((384, 640, 3), dtype=np.uint8)
    image = image[:, :, 0]
    for k, v in COLOR_DICT.items():
        out[image == k] = v
    return out

  
def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    k = np.identity(3)
    k[0, 0] = k[1, 1] = focal
    k[0, 2] = w / 2.0
    k[1, 2] = h / 2.0
    return k


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def m_iou(prediction, label):
    intersect = prediction[prediction == label]
    return intersect.shape[0] / (prediction.shape[0]*prediction.shape[1])


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = \
            box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = \
            box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def game_loop(args):
    # Intialize parameters
    pygame.init()
    pygame.font.init()
    actors_list = []
    vehicles_list = []
    walkers_list = []
    all_id = []
    synchronous_master = False
    random.seed(int(time.time()))

    # Intialize the model
    model = Model("model.onnx")
    model.warmup(imgsz=(1, 3, 384, 640))
    print("Finished initializing ONNX model")

    # Initialize metrics
    miou_drive = 0.0
    miou_lane = 0.0
    mean_ap = 0.0
    batch_stats = []
    batch_labels = torch.tensor([], device=torch.device('cuda'))
    count = 0

    # Connect to server
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    sim_world = client.get_world()

    # Apply synchronous settings
    original_settings = sim_world.get_settings()
    settings = sim_world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.1

    # Initialize traffic manager
    traffic_manager = client.get_trafficmanager()   # NOTE: Test with a distinct TM port or None

    # Set traffic_manager to be on synchronous mode
    traffic_manager.set_synchronous_mode(True)
    synchronous_master = True

    # Set hybrid mode for traffic manager
    traffic_manager.set_hybrid_physics_mode(True)
    traffic_manager.set_hybrid_physics_radius(DISTANCE_MAX)
    
    # Apply settings into world and reload keeping the same settings
    sim_world.apply_settings(settings)
    client.reload_world(False)   
    print('Initialized Simulation World')

    display = pygame.display.set_mode((args.width*2, args.height),
                                      pygame.HWSURFACE | pygame.DOUBLEBUF)
    display.fill((0,0,0))
    pygame.display.flip()
    font = get_font()

    clock = pygame.time.Clock()

    map = sim_world.get_map()
    spawn_points = map.get_spawn_points()
    start_pose = spawn_points[SPAWN_INDEX]   # Set first spawn point to belong to player
    del spawn_points[SPAWN_INDEX]
    # waypoint = map.get_waypoint(start_pose.location)

    blueprint_library = sim_world.get_blueprint_library()
    blueprint_vehicles = blueprint_library.filter('vehicle.*')
    blueprint_walkers = blueprint_library.filter('walker.pedestrian.*')

    blueprint_vehicles = sorted(blueprint_vehicles, key=lambda bp: bp.id)

    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    FutureActor = carla.command.FutureActor
    print("Obtained World's blueprint")

    # --------------
    # Spawn vehicles
    # --------------

    batch = []
    # Spawn player car
    blueprint_player = blueprint_library.filter('vehicle.*')[1]
    if blueprint_player.has_attribute('color'):
        color = random.choice(blueprint_player.get_attribute('color').recommended_values)
        blueprint_player.set_attribute('color', color)
    if blueprint_player.has_attribute('drive_id'):
        driver_id = random.choice(blueprint_player.get_attribute('drive_id').recommended_values)
        blueprint_player.set_attribute('driver_id', driver_id)
    blueprint_player.set_attribute('role_name', 'hero')

    batch.append(SpawnActor(blueprint_player, start_pose).then(
        SetAutopilot(FutureActor, True, traffic_manager.get_port())))

    # Spawn NPC car
    for n, transform in enumerate(spawn_points[1:]):
        if n >= NUM_VEHICLE:
            break
        blueprint = random.choice(blueprint_vehicles)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        blueprint.set_attribute('role_name', 'autopilot')

        batch.append(SpawnActor(blueprint, transform).then(
            SetAutopilot(FutureActor, True, traffic_manager.get_port())))

    # Apply batch sync on all NPC vehicles
    for response in client.apply_batch_sync(batch, synchronous_master):
        if response.error:
            logging.error(response.error)
        else:
            vehicles_list.append(response.actor_id)
    
    # Set automatic vehicle lights update if specified
    all_vehicle_actors = sim_world.get_actors(vehicles_list)
    for actor in all_vehicle_actors:
        traffic_manager.update_vehicle_lights(actor, True)
    player = all_vehicle_actors[0]
    
    #------------------
    # Spawn NPC walkers
    #------------------
    percentagePedestriansRunning = 0.0
    percentagePedestriansCrossing = 0.0
    
    # Create spawn_points for walkers
    spawn_points = []
    for i in range(NUM_PEDES):   # 10 is the number of walkers
        spawn_point = carla.Transform()
        loc = sim_world.get_random_location_from_navigation()
        if (loc != None):
            spawn_point.location = loc
            spawn_points.append(spawn_point)

    # Spawn walker objects
    batch = []
    walker_speed = []
    for spawn_point in spawn_points:
        blueprint_walker = random.choice(blueprint_walkers)
        if blueprint_walker.has_attribute('is_invincible'):
            blueprint_walker.set_attribute('is_invincible', 'false')
        if blueprint_walker.has_attribute('speed'):
            if (random.random() > percentagePedestriansRunning):
                walker_speed.append(blueprint_walker.get_attribute('speed').recommended_values[1])
            else:
                walker_speed.append(blueprint_walker.get_attribute('spped').recommended_values[2])
        else:
            print('Walker has no speed')
            walker_speed.append(0.0)
        batch.append(SpawnActor(blueprint_walker, spawn_point))
    results = client.apply_batch_sync(batch, True)
    walker_speed_2 = []
    for i in range(len(results)):
        if results[i].error:
            logging.error(results[i].error)
        else:
            walkers_list.append({"id": results[i].actor_id})
            walker_speed_2.append(walker_speed[i])
    walker_speed = walker_speed_2

    # Spawn walker's AI controller
    batch = []
    walker_controller = blueprint_library.find('controller.ai.walker')
    for i in range(len(walkers_list)):
        batch.append(SpawnActor(walker_controller, carla.Transform(), walkers_list[i]["id"]))
    results = client.apply_batch_sync(batch, True)
    for i in range(len(results)):
        if results[i].error:
            logging.error(results[i].error)
        else:
            walkers_list[i]["con"] = results[i].actor_id
    
    # Put together the walkers and controllers id to get the objects from their id
    for i in range(len(walkers_list)):
        all_id.append(walkers_list[i]["con"])
        all_id.append(walkers_list[i]["id"])
    all_actors = sim_world.get_actors(all_id)

    # Wait for a tick to ensure client receives the last transform of the walkers we 
    # have just created
    sim_world.tick()   # NOTE: Causing the snapshot id to be late 1 frame when we start extracting sensor's data

    # Initialize each controller and set target to walk to
    # Set how many pedestrians can cross the road
    sim_world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
    for i in range(0, len(all_id), 2):
        # Start walker
        all_actors[i].start()
        # Set walker to random point
        all_actors[i].go_to_location(sim_world.get_random_location_from_navigation())
        # Max speed
        all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))
    
    print(f"Spawned {len(vehicles_list)} vehicles and {len(walkers_list)} walkers")
    traffic_manager.global_percentage_speed_difference(30.0)

    # Setting player autopilot
    actors_list.append(player)
    player.set_simulate_physics(True)  # NOTE: experiment
    player.set_autopilot(True)
    player.set_light_state(carla.VehicleLightState.NONE)
    
    # Player's bounding box
    bound_x = 0.5 + player.bounding_box.extent.x
    bound_y = 0.5 + player.bounding_box.extent.y
    bound_z = 0.5 + player.bounding_box.extent.z

    camera_transform = carla.Transform(carla.Location(x=+0.22*bound_x,
                                                        y=+0.0*bound_y,
                                                        z=1.0*bound_z))
    
    instance_transform = carla.Transform(carla.Location(x=+0.24*bound_x,
                                                    y=+0.12*bound_y,
                                                    z=1.0*bound_z))
    # List of sensors
    sensors = [
        ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
        ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)', {}],
        ['sensor.camera.instance_segmentation', cc.Raw, 'Camera Instance Segmentation (Raw)', {}]]

    # Set sensors' resolution and gamma
    for item in sensors:
        bp = blueprint_library.find(item[0])
        if item[0].startswith('sensor.camera'):
            bp.set_attribute('image_size_x', str(args.width))
            bp.set_attribute('image_size_y', str(args.height))
            if bp.has_attribute('gamma'):
                bp.set_attribute('gamma', str(2.2))
            for attr_name, attr_value in item[3].items():
                bp.set_attribute(attr_name, attr_value)

        item.append(bp)

    # Spawn sensor on player
    camera_rgb = sim_world.spawn_actor(sensors[0][-1], camera_transform, attach_to=player,
                                        attachment_type=carla.AttachmentType.Rigid)
    actors_list.append(camera_rgb)

    camera_raw = sim_world.spawn_actor(sensors[1][-1], camera_transform, attach_to=player,
                                        attachment_type=carla.AttachmentType.Rigid) 
    actors_list.append(camera_raw)

    camera_instance = sim_world.spawn_actor(sensors[2][-1], instance_transform, attach_to=player,
                                            attachment_type=carla.AttachmentType.Rigid)
    actors_list.append(camera_instance)        
    print("Finished setting up sensors")

    calibration = np.identity(3)
    calibration[0, 2] = args.width / 2.0
    calibration[1, 2] = args.height / 2.0
    calibration[0, 0] = calibration[1, 1] = args.width / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
    camera_rgb.calibration = calibration
    camera_raw.calibration = calibration
    camera_instance.calibration = calibration
    print("Finished sensors' calibration")

    K = build_projection_matrix(args.width, args.height, VIEW_FOV)
    
    try:
        with SyncWorld(sim_world, camera_rgb, camera_raw, camera_instance, fps=30) as sync_world:
            while True:
                if should_quit():
                    return
                clock.tick_busy_loop(60)
                count += 1
                first = True
                picked_bboxes = []

                # Advance the simulation and wait for the data
                _, image_rgb, image_raw, image_instance = sync_world.tick(timeout=30.0)
                canva = np.zeros((args.height, args.width*2, 3), dtype=np.dtype("uint8"))

                # Extract raw data from sensor class
                im_rgb = np.frombuffer(image_rgb.raw_data, dtype=np.dtype("uint8"))
                im_rgb = np.reshape(im_rgb, (image_rgb.height, image_rgb.width, 4))
                im_rgb = im_rgb[:, :, :3]
                im_rgb = im_rgb[:, :, ::-1]

                instance_mask = np.frombuffer(image_instance.raw_data, dtype=np.dtype("uint8"))
                instance_mask = np.reshape(instance_mask, (image_instance.height, image_instance.width, 4))
                instance_mask = instance_mask[:, :, :3]
                instance_mask = instance_mask[:, :, ::-1]

                # image_raw.convert(cc.CityScapesPalette)
                im_raw = np.frombuffer(image_raw.raw_data, dtype=np.dtype("uint8"))
                im_raw = np.reshape(im_raw, (image_raw.height, image_raw.width, 4))
                im_raw = im_raw[:, :, :3]
                im_raw = im_raw[:, :, ::-1]

                label_drive = deepcopy(im_raw[:, :, 0])   # Red channel
                label_lane = deepcopy(im_raw[:, :, 0])
                label_drive[label_drive == 1] = 255
                label_drive[label_drive != 255] = 0
                label_lane[label_lane == 24] = 255
                label_lane[label_lane != 255] = 0

                # Get and draw predictions to a copy of im_rgb
                lane, drive, det = model(im_rgb)
                drive[drive != 2] = 255
                drive[drive == 2] = 1   # Exclude background class from mIoU
                lane[lane != 0] = 255
                lane[lane == 0] = 1     # Exclude background class form mIoU

                # Calculate miou
                miou_drive += m_iou(drive, label_drive)
                miou_lane += m_iou(lane, label_lane)
                im_result = deepcopy(im_rgb)
                im_result[drive == 255] = [128, 64, 128]
                im_result[lane == 255] = [157, 234, 50]
                
                # Get world level bounding box from CARLA server
                world_2_camera = np.array(camera_rgb.get_transform().get_inverse_matrix())
                world_labels = [carla.CityObjectLabel.TrafficLight, carla.CityObjectLabel.Pedestrians,
                                carla.CityObjectLabel.TrafficSigns, carla.CityObjectLabel.Rider,
                                carla.CityObjectLabel.Car, carla.CityObjectLabel.Truck,
                                carla.CityObjectLabel.Bus, carla.CityObjectLabel.Motorcycle,
                                carla.CityObjectLabel.Bicycle]

                # Get and assign tag to bbox
                world_bbox_set = []
                for i, tag in enumerate(world_labels):
                    class_bbox_set = sim_world.get_level_bbs(tag)
                    for bbox in class_bbox_set:
                        world_bbox_set.append([bbox, REMAP_TAG[i]])
                
                # Filter bbox by location and position with respective to the main sensor
                nearby_bboxes = []
                distance_to_player = []
                for bbox in world_bbox_set:
                    dist = bbox[0].location.distance(player.get_transform().location)
                    if dist < DISTANCE_MAX:
                        if bbox[1] not in [7, 8, 12, 18, 19]:   # Exclude signs, lights and pedestrians
                            forward_vec = player.get_transform().get_forward_vector()
                            ray = bbox[0].location - player.get_transform().location
                            if forward_vec.dot(ray) > 1.75:
                                nearby_bboxes.append(bbox)
                                distance_to_player.append(dist)
                        else:
                            nearby_bboxes.append(bbox)
                            distance_to_player.append(dist)

                world_bboxes = ClientSideBoundingBoxes.get_bounding_boxes(
                    nearby_bboxes, distance_to_player, world_2_camera, K, args.width, args.height)
  
                # Filter occluded bounding box with instance segmentation data
                for i, bbox in enumerate(world_bboxes):
                    cx = bbox[0] + ((bbox[2] - bbox[0]) // 2)
                    cy = bbox[1] + ((bbox[3] - bbox[1]) // 2)
                    world_cls = instance_mask[cy, cx, 0]
                    if world_cls == bbox[4]:
                        bbox[4] = CARLA_TO_MODEL[bbox[4]]
                        if first:
                            picked_bboxes = torch.tensor([bbox], device=torch.device('cuda'))
                            first = False
                        else:
                            picked_bboxes = torch.concat((picked_bboxes, torch.tensor([bbox], device=torch.device('cuda'))), dim=0)

                # Putting all labels into a batch_labels list
                if not first and len(picked_bboxes):
                    if not len(batch_labels):
                        batch_labels = picked_bboxes[:, 4]
                    else:
                        batch_labels = torch.concat((batch_labels, picked_bboxes[:, 4]))

                try:
                    # Calculate Average Precision for object detection per image and add the stats to batch_stats
                    if len(det) != 0:
                        boxes = det[:, :4].int()
                        scores = det[:, 4]
                        classes = det[:, 5].int()
                        true_positives = torch.zeros(boxes.shape[0], dtype=torch.int8, device=det.device)

                        if len(picked_bboxes):
                            detected_boxes = []
                            tg_boxes = picked_bboxes[:, :4]
                            tg_classes = picked_bboxes[:, 4]

                            for pred_i, (pred_box, pred_cls) in enumerate(zip(boxes, classes)):
                                # If groundtruth are found break
                                if len(detected_boxes) == len(picked_bboxes):
                                    break

                                # Ignore if prediction class is not one of the groundtruth classes
                                if pred_cls not in tg_classes:
                                    continue

                                # Filter target boxes by pred_label so that we only match against boxes of our own label
                                filtered_tg_position, filtered_tg = zip(*filter(lambda x: tg_classes[x[0]] == pred_cls, enumerate(tg_boxes)))

                                # Find the best matching target for our predicted box
                                iou, box_filtered_index = bbox_iou(pred_box.unsqueeze(0), torch.stack(filtered_tg)).max(0)

                                # Remap the index in the list of filtered targets for that label to the index in the list with all targets
                                box_index = filtered_tg_position[box_filtered_index]

                                # Check if the iou is above the min threshold and i
                                if iou >= IOU_THRESHOLD and box_index not in detected_boxes:
                                    true_positives[pred_i] = 1
                                    detected_boxes += [box_index]
                        batch_stats.append([true_positives, scores, classes])
                    
                    # Start calculating map for a certain batch amount, avoid many errors
                    if count >= 50 and count % 10 == 0:
                        true_positives, scores, classes = [torch.concat(x, 0) for x in list(zip(*batch_stats))]

                        # Sort by confidence
                        i = torch.argsort(scores, descending=True)
                        true_positives, scores, classes = true_positives[i], scores[i], classes[i]
                        
                        unique_classes = torch.unique(batch_labels)

                        # Create Precision-Recall curve and compute AP for each class
                        ap, p, r = [], [], []
                        for c in unique_classes:
                            i = classes == c
                            n_gt = (batch_labels == c).sum()   # Number of groundtruth objects
                            n_p = i.sum()   # Number of predicted objects

                            if n_p == 0 and n_gt == 0:
                                continue
                            elif n_p == 0 or n_gt == 0:
                                ap.append(0)
                                r.append(0)
                                p.append(0)
                            else:
                                # Accumulate FPs and TPs
                                fpc = (1 - true_positives[i]).cumsum(0)
                                tpc = (true_positives[i]).cumsum(0)

                                # Recall
                                recall_curve = tpc / (n_gt + 1e-16)
                                r.append(recall_curve[-1].cpu().numpy())

                                # Precision
                                precision_curve = tpc / (tpc + fpc)
                                p.append(precision_curve[-1].cpu().numpy())

                                # AP from Precision-Recall curve 
                                mrec = torch.concat((torch.tensor([0.0], device=torch.device('cuda')),
                                                     recall_curve,
                                                     torch.tensor([1.0], device=torch.device('cuda'))))
                                mpre = torch.concat((torch.tensor([0.0], device=torch.device('cuda')),
                                                     precision_curve,
                                                     torch.tensor([0.0], device=torch.device('cuda'))))

                                for i in range(mpre.shape[0] - 1, 0, -1):
                                    mpre[i - 1] = torch.maximum(mpre[i - 1], mpre[i])

                                i = torch.where(mrec[1:] != mrec[:-1])[0]
                                ap.append(torch.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]).detach().cpu().numpy())

                        mean_ap = np.array(ap).mean()

                        # p, r, ap = np.array(p), np.array(r), np.array(ap)
                        # f1 = 2 * p * r / (p + r + 1e-16)
                except Exception as e:
                    print(e)

                # Draw predicted bounding boxes
                det = det.cpu().numpy()
                for bbox in det:
                    cv2.rectangle(im_result, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)

                # Draw the display
                canva[:, :canva.shape[1]//2, :] = im_rgb
                canva[:, canva.shape[1]//2:, :] = im_result

                image_surface = pygame.surfarray.make_surface(canva.swapaxes(0, 1))
                display.blit(image_surface, (0, 0))
                display.blit(
                    font.render('% 5d FPS (real)' % clock.get_fps(), True, (255, 255, 255)),
                    (8, 10))
                display.blit(
                    font.render('% 5.5f mIoU Drive' % (miou_drive/count), True, (255, 255, 255)),
                    (8, 25))
                display.blit(
                    font.render('% 5.5f mAP50' % (mean_ap), True, (255, 255, 255)),
                    (8, 40))
                if not first:
                    ClientSideBoundingBoxes.draw_bounding_boxes(display, picked_bboxes, font, args.width, args.height)
                pygame.display.flip()

    except Exception as e:
        print(e)

    finally:
        
        print("Destroying actors")
        for actor in actors_list:
            actor.destroy()

        print("Destroying NPCs")
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])
        
        for i in range(0, len(all_id), 2):
            all_actors[i].stop()
        client.apply_batch([carla.command.DestroyActor(x) for x in all_id])

        if original_settings:
            sim_world.apply_settings(original_settings)

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='640x384',
        help='window resolution (default: 640x384)')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)
    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
