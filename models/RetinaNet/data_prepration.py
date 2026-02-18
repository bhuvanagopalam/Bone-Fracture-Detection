import glob
import xml.etree.ElementTree as ET
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def convert_pasccalvoc_to_df(annotation_dir, image_dir,):
    xml_list = []
    for xml_file in glob.glob(annotation_dir + "/*.xml"):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        if root.findall("object"):
            for member in root.findall("object"):
                bbx = member.find("bndbox")
                xmin = int(bbx.find("xmin").text)
                ymin = int(bbx.find("ymin").text)
                xmax = int(bbx.find("xmax").text)
                ymax = int(bbx.find("ymax").text)
                label = member.find("name").text

                value = (
                    root.find("filename").text,
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                    label,
                )
                xml_list.append(value)
        else:
            # Add placeholder entry if no objects are found
            value = (root.find("filename").text, "", "", "", "", "")
            xml_list.append(value)

    column_name = [
        "filename",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "class",
    ]

    xml_df = pd.DataFrame(xml_list, columns=column_name)
    xml_df["filename"] = [
        os.path.join(image_dir, xml_df["filename"][i]) for i in range(len(xml_df))
    ]

    return xml_df



if __name__ == "__main__":

    df = convert_pasccalvoc_to_df('/home/saeid/MP/Project/FracAtlas/Annotations/PASCAL VOC',"/home/saeid/MP/Project/FracAtlas/all_images")
    df.to_csv('data.csv',index=False)

    

    data = pd.read_csv("data.csv")


    # data['class_label'] = data['class'].apply(lambda x: 'fractured' if x == 'fractured' else 'non-fractured')

    # file_groups = data.groupby('filename')['class_label'].apply(lambda x: x.mode()[0]).reset_index()
    # file_groups.columns = ['filename', 'class_label']

    # train_files, temp_files = train_test_split(
    #     file_groups,
    #     test_size=0.3,
    #     stratify=file_groups['class_label'],
    #     random_state=42
    # )


    # train_data = data[data['filename'].isin(train_files['filename'])]
    # temp_data = data[data['filename'].isin(temp_files['filename'])]

    # validation_files, test_files = train_test_split(
    #     temp_files,
    #     test_size=0.5,  
    #     stratify=temp_files['class_label'],
    #     random_state=42
    # )

    # validation_data = data[data['filename'].isin(validation_files['filename'])]
    # test_data = data[data['filename'].isin(test_files['filename'])]


    # train_data = train_data.drop(columns=['class_label'])
    # validation_data = validation_data.drop(columns=['class_label'])
    # test_data = test_data.drop(columns=['class_label'])


    # train_data.to_csv("train_data1.csv", index=False)
    # validation_data.to_csv("validation_data1.csv", index=False)
    # test_data.to_csv("test_data1.csv", index=False)

    # print(f"Train set: {len(train_data)} annotations")
    # print(f"Validation set: {len(validation_data)} annotations")
    # print(f"Test set: {len(test_data)} annotations")


    data_file = "data.csv"
    train_file = "train.csv"
    val_file = "valid.csv"
    test_file = "test.csv"

    train_output = "train_filtered.csv"
    val_output = "val_filtered.csv"
    test_output = "test_filtered.csv"

    
    data_df = pd.read_csv(data_file)
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)


    train_images = set(train_df["image_id"])
    val_images = set(val_df["image_id"])
    test_images = set(test_df["image_id"])

    data_df["image_id"] = data_df["filename"].apply(lambda x: x.split("/")[-1])


    train_filtered = data_df[data_df["image_id"].isin(train_images)]
    val_filtered = data_df[data_df["image_id"].isin(val_images)]
    test_filtered = data_df[data_df["image_id"].isin(test_images)]
    train_filtered.drop('image_id', inplace=True, axis=1)
    val_filtered.drop('image_id', inplace=True, axis=1)
    test_filtered.drop('image_id', inplace=True, axis=1)


    train_filtered.to_csv(train_output, index=False)
    val_filtered.to_csv(val_output, index=False)
    test_filtered.to_csv(test_output, index=False)