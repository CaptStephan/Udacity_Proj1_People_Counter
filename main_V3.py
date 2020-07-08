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
MQTT_PORT = 3001
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
    count_current = 0
    count_last = 0
    count_last_last = 0
    total_count = 0
    duration = 0
    avg_duration = 0
    total_duration = 0
    start_time = 0
    active_person = 0
    net_input_shape = []
    frame_count = 0

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
        #net_input_shape = [1, 3, 600, 600]
        #sometimes gives [1,3] and causes an error, have not been able to figure out why.  Will work fine at random.
        sys.exit("Input shape error, forced exit. Please run again until this error does not appear.")

    ### TODO: Loop until stream is over ###
    while cap.isOpened():

        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        frame_count += 1

        if not flag:
            #video stream ended, go to end and close out
            break

        ### TODO: Start asynchronous inference for specified request ###
        if frame_count%2 == 0: #check every other frame
            ### TODO: Pre-process the image as needed ###
            vid_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))

            #save a copy of the input frame to use on output
            vid_frame_copy = vid_frame

            vid_frame = vid_frame.transpose((2, 0, 1))
            vid_frame = vid_frame.reshape(1, *vid_frame.shape)

            infer_network.exec_net(vid_frame)

            ### TODO: Wait for the result ###
            if infer_network.wait() == 0:

                ### TODO: Get the results of the inference request ###
                results = infer_network.get_output()

                # for this model, results should be shape [1, 1, N, 7]
                # N is number of hits, last is a 7 item list [image_id, label, conf, x_min,
                # y_min, x_max, y_max] where label is the predicted class

                ### TODO: Extract any desired stats from the results ###
                out_frame, count_current, box = draw_boxes(vid_frame_copy, results, args, net_input_shape[3], net_input_shape[2])
                #out_frame = cv2.putText(out_frame, "Last Frame Analyzed = "+str(frame_count), (10, 420), cv2.FONT_HERSHEY_COMPLEX_SMALL,  1, (255, 0, 0), 1, cv2.LINE_AA)

                ### TODO: Calculate and send relevant information on ###
                ### count_current, total_count and duration to the MQTT server ###
                ### Topic "person": keys of "count" and "total" ###
                ### Topic "person/duration": key of "duration" ###

                # This block of code from Mentor Help question 129845, some modifications by me
                # If both last and last_last are equal, positive ID for two frames.
                if count_current > count_last and count_last_last == count_last:
                    start_time = time.time()
                    total_count = total_count + count_current - count_last

                client.publish("person", json.dumps({"count": count_current}))

                # Person duration in the video is calculated if two frames of no detect to account for skipped frame
                if count_current < count_last_last and count_last < count_last_last:
                    duration = int(time.time() - start_time)
                    total_duration += duration / 12 #frames per second and evaluating only every other frame
                    avg_duration = int(total_duration / total_count)
                    client.publish("person/duration", json.dumps({"duration": avg_duration}))
                #End of modified block of code from Mentor Help question 129845

                # Set a double counter to review two frames at a time
                count_last_last = count_last
                count_last = count_current


        ### TODO: Send the frame to the FFMPEG server ###
                out_frame = out_frame.copy(order='C')
                out_frame = cv2.resize(out_frame, (width, height))
                np.ascontiguousarray(out_frame, dtype=np.float32)
                sys.stdout.buffer.write(out_frame)
                sys.stdout.flush()

        ### TODO: Write an output image if `single_image_mode` ###
        #will implement later

    #Release the capture and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

    #Disconnect from MQTT
    client.disconnect()

    #Print final numbers for reference
#    print("Video stream ended.")
#    print("Final count was " + str(total_count))
#    print("Average Duration was " + str(avg_duration) + " seconds.")

#create box shapes
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

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()

    # Connect to the MQTT server
    client = connect_mqtt()

    # Perform inference on the input stream
    infer_on_stream(args, client)

if __name__ == '__main__':
    main()
