from roboflow import Roboflow

rf = Roboflow(api_key="QxGi19KHY22hkvBFiLXs")
project = rf.workspace("vancount-app").project("pl10-qnstk")
dataset = project.version(5).download("yolov11")