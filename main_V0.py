"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2
import numpy as np

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
#MQTT_HOST = "ws://localhost:3000"
MQTT_PORT = 3000
MQTT_KEEPALIVE_INTERVAL = 60

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    current_count = 0
    last_count = 0
    total_count = 0
    duration = 0
    avg_duration = 0
    start_time = 0
    active_person = 0
    net_input_shape = []
    frame_count = 0
    old_box = [[0,0], [0,0]]
    box = [[0,0], [0,0]]
    box_dist = 0
    center_old = [[0,0], [0,0]]
    center_new = [[0,0], [0,0]]
    new_detect = 0

    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(model=args.model, device=args.device, cpu_extension=args.cpu_extension)

    ### TODO: Handle the input stream ###
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)

    # get the required shape for the network
    net_input_shape = infer_network.get_input_shape()

    # get the shape of the input image
    width = int(cap.get(3))
    height = int(cap.get(4))

    if net_input_shape != [1, 3, 600, 600]:
        net_input_shape = [1, 3, 600, 600]
        #sometimes gives [1,3] and causes an error, so hard coded shape to match model
        #sys.exit("Input shape error, forced exit. Please run again.")

    ### TODO: Loop until stream is over ###
    while cap.isOpened():

        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        frame_count += 1
        #print("Frame count is " + str(frame_count))
        #print("Getting flag and frame.")
        #print("Frame size is " + str(frame.size))
        #print("Frame shape is " + str(frame.shape))

        if not flag:
            #print("Video stream ended.")
            exit(0)

        key_pressed = cv2.waitKey(45)

        if key_pressed == 27:
            #print("Escape key pressed.")
            break

        ### TODO: Pre-process the image as needed ###
        #print("Trying to reshape input frame.")
        vid_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        #print("Made cv2.resize successfully = " + str(vid_frame.shape))
        #save a copy of the input frame to use on output
        vid_frame_copy = vid_frame
        vid_frame = vid_frame.transpose((2, 0, 1))
        #print("Made transpose successfully = " + str(vid_frame.shape))
        vid_frame = vid_frame.reshape(1, *vid_frame.shape)
        #print("Made reshape successfully = " + str(vid_frame.shape))

        ### TODO: Start asynchronous inference for specified request ###
        infer_network.exec_net(vid_frame)
        #print("Made first infer_network.exec_net successfully.")

        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:

            ### TODO: Get the results of the inference request ###
            #print("Wait is == 0.")
            if frame_count%2 == 0: #check every other frame
                results = infer_network.get_output()
            #print("Finished get_output.")

            # for this model, results should be shape [1, 1, N, 7]
            # N is number of hits, last is a 7 item list [image_id, label, conf, x_min,
            # y_min, x_max, y_max] where label is the predicted class

            ### TODO: Extract any desired stats from the results ###
            #print("The shape of the returned results is: " + str(results.shape))
                #Check for skipped frame
                old_box = box

                out_frame, current_count, box = draw_boxes(vid_frame_copy, results, args, net_input_shape[3], net_input_shape[2])
                out_frame = cv2.putText(out_frame, "Active person detected = "+str(current_count), (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,  0.8, (255, 0, 255), 1, cv2.LINE_AA)
                #print("Old box is " + str(old_box))
                #print("New box is " + str(box))

            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            #Check for new person and check distance from last box to eliminate duplicate counts
                #print("Old box center is " + str(center_old))
                #print("New box center is " + str(center_new))
                #print("center_old[0] is " + str(center_old[0]))
                #print("center_new[0] is " + str(center_new[0]))

                if center_new != [[0, 0], [0, 0]] and new_detect == 1:
                    center_old = get_center(old_box)
                    center_new = get_center(box)
                    box_dist = ((center_old[0]-center_new[0])**2 + (center_old[1]-center_new[1])**2)**0.5
                #print("Box distance is " + str(box_dist))
                out_frame = cv2.putText(out_frame, "box_dist = "+str('% 6.2f' % box_dist), (10, 120), cv2.FONT_HERSHEY_COMPLEX_SMALL,  0.8, (255, 0, 255), 1, cv2.LINE_AA)

                if current_count > last_count:
                    start_time = time.time()
                    total_count = total_count + current_count - last_count
                    if box_dist > 60 and new_detect == 0: #if too much movement from last location, then a frame was dropped
                        total_count -= 1 #take away the false detection of a different person
                    new_detect = 1 #marker for a new detection

                # Calculate duration
                if current_count < last_count and new_detect == 1:
                    duration = int(time.time() - start_time)
                    avg_duration = duration / total_count
                    new_detect = 0 #marker last detection is gone from frame

                out_frame = cv2.putText(out_frame, "Average Duration = "+str('% 6.2f' % avg_duration), (10, 90), cv2.FONT_HERSHEY_COMPLEX_SMALL,  0.8, (255, 0, 255), 1, cv2.LINE_AA)
                out_frame = cv2.putText(out_frame, "Total Count = "+str(total_count), (10, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL,  0.8, (255, 0, 255), 1, cv2.LINE_AA)
                out_frame = cv2.putText(out_frame, "Frame Count = "+str(frame_count), (10, 150), cv2.FONT_HERSHEY_COMPLEX_SMALL,  0.8, (255, 0, 255), 1, cv2.LINE_AA)
                last_count = current_count

                """"#This block of code from Mentor Help question 129845
                if current_count > last_count:
                    start_time = time.time()
                    total_count = total_count + current_count - last_count
                    client.publish("person", json.dumps({"total": total_count}))
                # Person duration in the video is calculated
                if current_count < last_count:
                    duration = int(time.time() - start_time)
                    # Publish messages to the MQTT server
                    client.publish("person/duration",
                                   json.dumps({"duration": duration}))
                client.publish("person", json.dumps({"count": current_count}))
                last_count = current_count
                #End block of code from Mentor Help question 129845
                --never got this to work, used cv2.putText instead"""

        ### TODO: Send the frame to the FFMPEG server ###
                out_frame = out_frame.copy(order='C')
                out_frame = cv2.resize(out_frame, (width, height))
                np.ascontiguousarray(out_frame, dtype=np.float32)
                sys.stdout.buffer.write(out_frame)
                sys.stdout.flush()

        ### TODO: Write an output image if `single_image_mode` ###

    #Release the capture and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

    #Disconnect from MQTT
    client.disconnect()

def draw_boxes(vid_frame, results, args, width, height):
    count = 0
    box = [[0, 0], [0, 0]]
    for i in results[0][0]:
        confidence = i[2]
        is_person = i[1]
        if confidence >= args.prob_threshold and is_person == 1:
            #print("The confidence detected is " + str(confidence))
            count += 1
            xmin = int(i[3] * width)
            ymin = int(i[4] * height)
            xmax = int(i[5] * width)
            ymax = int(i[6] * height)
            cv2.rectangle(vid_frame, (xmin,ymin), (xmax, ymax), (255, 0, 0), 1)
            box = [[xmax, ymax], [xmin, ymin]]
            #print("Result = " + str(i))
    return vid_frame, count, box

def get_center(box):
    cent = [[], []]
    if box != [[], []]:
        cent = (((box[0][0]+box[1][0])/2), ((box[0][1]+box[1][1])/2))
    return cent

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    #print("finished argparser")

    # Connect to the MQTT server
    client = connect_mqtt()
    #print("finished mqtt connect")

    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
