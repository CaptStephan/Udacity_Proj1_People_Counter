#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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
import logging as log
from openvino.inference_engine import IENetwork, IECore
#import faulthandler

class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None

    def load_model(self, model, device="CPU", cpu_extension=None):
        ### TODO: Load the model ###
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        self.plugin = IECore()
        self.network = IENetwork(model=model_xml, weights=model_bin)

        ### TODO: Add any necessary extensions ###
        # moved this from below to in front of supported layers check because adding
        # extensions will change supported layers
        #if cpu_extension != "None":
        #    self.plugin.add_extension(cpu_extension, device)

        ### TODO: Check for supported layers ###
        unsupported_layers = [0]
        #print("Length of unsupported layers before checking is " + str(len(unsupported_layers)))
        supported_layers = self.plugin.query_network(network=self.network, device_name=device)

        for i in self.network.layers.keys():
            if i not in supported_layers:
                unsupported_layers.append(i)
        if len(unsupported_layers) > 1:
            #print("Unsupported Layers Found, list follows:")
            #print(unsupported_layers)
            exit(1)

        ### TODO: Return the loaded inference plugin ###
        ### Note: You may need to update the function parameters. ###
        self.exec_network = self.plugin.load_network(self.network, device)
        #then need to get the input layer
        self.input_blob = next(iter(self.network.inputs))
        #print("Got the input_blob and it is " + str(self.input_blob))
        self.output_blob = next(iter(self.network.outputs))

        return self.exec_network

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        shape = self.network.inputs[self.input_blob].shape
        #print("Retrieved input shape = " + str(shape))

        return shape

    def exec_net(self, image):
        ### TODO: Start an asynchronous request ###
        #print("Got to exec_net def and image shape is " + str(image.shape))
        self.exec_network.start_async(request_id=0, inputs={self.input_blob: image, 'image_info': image.shape[1:]})
        #print("Started an async request.")

        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return

    def wait(self):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        #print("Got to wait function.")
#       #faulthandler.enable()
#       #print("Enabled faulthandler")
        status = self.exec_network.requests[0].wait(-1)
        #print("Got status.")

        ### Note: You may need to update the function parameters. ###
        return status

    def get_output(self):
        ### TODO: Extract and return the output results
        output = self.exec_network.requests[0].outputs[self.output_blob]

        ### Note: You may need to update the function parameters. ###
        return output
