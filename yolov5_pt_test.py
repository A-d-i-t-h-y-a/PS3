import subprocess
def run_yolo(image_path):
    # Replace this with the actual path to your YOLO model and configuration files
    yolo_path = 'path_to_your_yolo_files'
    yolo_command = f'yolo task=detect mode=predict model=best.pt show=true conf=0.5 source={image_path} save_crop=True'
    subprocess.run(yolo_command,shell=True)

image_path = r"..\..\big_data.v1i.multiclass\train\labelledimg\aadhardetect.v6i.voc\train\0c0584201ff552c4bdcbe160315aa432_jpg.rf.1bb8240b782857a43940ba0825eb998a.jpg"
run_yolo(image_path)