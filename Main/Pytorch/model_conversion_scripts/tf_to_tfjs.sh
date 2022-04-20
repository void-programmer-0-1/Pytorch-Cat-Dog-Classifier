#! /bin/bash

tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model ../weights/cat_vs_dog/ ../weights/tfjs_model/
