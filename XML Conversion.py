# NOTE: run this before the dataloading otherwise the dataloader might get those images which don't have YOLO lables so training will be useless even if yolo_loss function is correct

import xml.etree.ElementTree as ET
import os

# using only 7 objects instead of VOC original 20 objects for simplicity
VOC_TO_ID = {
    "person": 0,
    "chair": 1,
    "diningtable": 2,  
    "bottle": 3,
    "sofa": 4,        
    "pottedplant": 5,  
    "tvmonitor": 6
}

# Where to save the new YOLO .txt files
OUTPUT_DIR = r"C:\Users\Parth\Downloads\yolo_labels"

# XML Directories (Train and Test)
XML_DIRS = [
    r"C:\Users\Parth\Downloads\archive(1)\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007\Annotations",
    r"C:\Users\Parth\Downloads\archive(1)\VOCtest_06-Nov-2007\VOCdevkit\VOC2007\Annotations"
]

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x*dw, y*dh, w*dw, h*dh)

def convert_annotation():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    count = 0
    total_objs = 0
    
    for xml_dir in XML_DIRS:
        if not os.path.exists(xml_dir):
            print(f"Skipping (not found): {xml_dir}")
            continue

        print(f"Processing: {xml_dir}")
        
        for filename in os.listdir(xml_dir):
            if not filename.endswith('.xml'): continue

            file_id = filename[:-4]
            in_file = open(os.path.join(xml_dir, filename))
            
            tree = ET.parse(in_file)
            root = tree.getroot()
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)

            # String buffer for labels
            label_data = []

            for obj in root.iter('object'):
                difficult = obj.find('difficult').text
                cls_name = obj.find('name').text
                
                # Check if this object is in our target list
                if cls_name not in VOC_TO_ID or int(difficult) == 1:
                    continue

                cls_id = VOC_TO_ID[cls_name]
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), 
                     float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                
                bb = convert((w,h), b)
                label_data.append(f"{cls_id} {bb[0]} {bb[1]} {bb[2]} {bb[3]}\n")
                total_objs += 1

            in_file.close()

            # Only save file if we found relevant objects
            if label_data:
                with open(os.path.join(OUTPUT_DIR, f"{file_id}.txt"), 'w') as out_file:
                    out_file.writelines(label_data)
                count += 1

    print(f"Success! Processed {count} images containing {total_objs} objects.")
    print(f"Labels saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    convert_annotation()
