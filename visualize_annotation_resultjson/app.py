import cv2
import json, urllib
import numpy as np
import streamlit as st
from abeja.datalake import Client as DatalakeClient
from abeja.datalake import APIClient

def frame_selector_ui(max_number):
    st.sidebar.markdown("# Number of display")

    # The user can select a range for how many of the selected objecgt should be present.
    select_min, select_max = st.sidebar.slider("How many obj to display (select a range)?", 0, max_number, [0, 1])

    return select_min, select_max

def draw_image_with_boxes(image, boxes, header="", description=""):
    # Superpose the semi-transparent object detection boxes.    # Colors for the boxes
    LABEL_COLORS = {
        "car": [255, 0, 0],
        "pedestrian": [0, 255, 0],
        "truck": [0, 0, 255],
        "trafficLight": [255, 255, 0],
        "biker": [255, 0, 255],
    }
    image_with_boxes = image.astype(np.float64)
    for x in boxes:
        image_with_boxes[int(x[1]):int(x[3]),int(x[0]):int(x[2]),:] += LABEL_COLORS["car"]
        image_with_boxes[int(x[1]):int(x[3]),int(x[0]):int(x[2]),:] /= 2

    annotation = ""
    draw_image(image_with_boxes, header, description, annotation)

def draw_image(image, header="", description="", annotation=""):
    # Draw the header and image.
    st.subheader(header)
    st.markdown(annotation)
    st.markdown(description)
    st.image(image.astype(np.uint8), use_column_width=True)

def draw_text(text, header="", description="", annotation=""):
    st.subheader(header)
    st.markdown(annotation)
    st.markdown(description)
    st.code(text, language='text')

@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/abeja-inc/Platform_tool/master/visualize_annotation_resultjson/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")

def get_file_local_content_as_string(file_path):
    with open(file_path) as f:
        content = f.read()
        return content

def run_objdct(json_load):

    max_image_number = len(json_load)
    select_min, select_max = frame_selector_ui(max_image_number)

    for x in json_load[select_min:select_max]:
        task_id = x["task_id"]
        notes = x["notes"]
        review_information = x["review_information"]
        
        information = x["information"]
        task = x["task"]
        metadata = task["metadata"][0]
    
        # Get Platform Channel_id
        try:
            file_name = metadata["information"]["filename"]
            channel_id = metadata["channel_id"]
            channel = datalake_client.get_channel(channel_id)
        except:
            st.error('👈 **Please set your credential**')
            break
        else:
            file_id = metadata["source"]
            datalake_file = channel.get_file(file_id=file_id)
            raw_image = datalake_file.get_content()
            image = np.asarray(bytearray(raw_image), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
            box_list = []
            for y in information:
                box = y["rect"]
                box_list.append(box)
            main_text = f"Task_id: {task_id}, Filename: {file_name}"
            sub_text = f"notes: {notes}, review_information: {review_information}"
            draw_image_with_boxes(image,box_list, main_text, sub_text)

def run_segment(json_load):
    max_image_number = len(json_load)
    select_min, select_max = frame_selector_ui(max_image_number)

    for x in json_load[select_min:select_max]:
        task_id = x["task_id"]
        notes = x["notes"]
        review_information = x["review_information"]
        result_data_lake_channel = x["result_data_lake_channel"]

        # Get combned picture(ouput file)
        for y in  x["information"]:
            if "is_combined" in y:
                out_file_id = y["file_id"]

        # Get ouput channel_id
        out_channel_id = result_data_lake_channel["channel_id"]

        task = x["task"]
        metadata = task["metadata"][0]
        file_name = metadata["information"]["filename"]
        in_channel_id = metadata["channel_id"]

        # Get Platform Channel_id
        try:
            channel_in = datalake_client.get_channel(in_channel_id)
            channel_out = datalake_client.get_channel(out_channel_id)
        except:
            st.error('👈 **Please set your credential**')
            break
        else:
            in_file_id = metadata["source"]

            #in_put_file
            datalake_file_in = channel_in.get_file(file_id=in_file_id)
            raw_image = datalake_file_in.get_content()
            img_in = np.asarray(bytearray(raw_image), dtype="uint8")
            img_in = cv2.imdecode(img_in, cv2.IMREAD_COLOR)

            #annotation_result
            datalake_file_out = channel_out.get_file(file_id=out_file_id)
            raw_image = datalake_file_out.get_content()
            img_out = np.asarray(bytearray(raw_image), dtype="uint8")
            img_out = cv2.imdecode(img_out, cv2.IMREAD_COLOR)

            # アルファブレンディング
            alpha = 0.7

            blended_image = cv2.addWeighted(img_in, alpha, img_out, 1 - alpha, 0)

            main_text = f"Task_id: {task_id}, Filename: {file_name}"
            sub_text = f"notes: {notes}, review_information: {review_information}"
            draw_image(blended_image, main_text, sub_text)

def run_image_classify(json_load):
    max_image_number = len(json_load)
    select_min, select_max = frame_selector_ui(max_image_number)

    for x in json_load[select_min:select_max]:
        task_id = x["task_id"]
        notes = x["notes"]
        review_information = x["review_information"]

        # Get label
        label_list = []
        for y in  x["information"]:
          label_list.append(y["label"])

        task = x["task"]
        metadata = task["metadata"][0]
        file_name = metadata["information"]["filename"]
        channel_id = metadata["channel_id"]

        # Get Platform Channel_id
        try:
            channel = datalake_client.get_channel(channel_id)
        except:
            st.error('👈 **Please set your credential**')
            break
        else:
            file_id = metadata["source"]

            #writing image file
            datalake_file = channel.get_file(file_id=file_id)
            raw_image = datalake_file.get_content()
            img = np.asarray(bytearray(raw_image), dtype="uint8")
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)

            label_text = " ".join(label_list)
            main_text = f"Task_id: {task_id}, Filename: {file_name}"
            annotation  = f"label: {label_text}"
            sub_text = f"notes: {notes}, review_information: {review_information}"
            draw_image(img, main_text, sub_text, annotation)

def run_text_classify(json_load):
     max_text_number = len(json_load)
     select_min, select_max = frame_selector_ui(max_text_number)

     for x in json_load[select_min:select_max]:
         task_id = x["task_id"]
         notes = x["notes"]
         review_information = x["review_information"]

         # Get label
         label_length = ""
         for k,v in  x["information"].items():
             label_length = label_length + str(k) + ":" + str(v) + ", "

         task = x["task"]
         metadata = task["metadata"][0]
         file_name = metadata["information"]["filename"]
         channel_id = metadata["channel_id"]

         # Get Platform Channel_id
         try:
             channel = datalake_client.get_channel(channel_id)
         except:
             st.error('👈 **Please set your credential**')
             break
         else:
             file_id = metadata["source"]

             #writing text file
             datalake_file = channel.get_file(file_id=file_id)
             raw_text = datalake_file.get_content().decode()

             main_text = f"Task_id: {task_id}, Filename: {file_name}"
             sub_text = f"notes: {notes}, review_information: {review_information}"
             draw_text(raw_text, main_text, sub_text, label_length)


def run_the_app():
    uploaded_file = st.file_uploader("Please chose Json file", type=['json'])
    if uploaded_file is not None:
        #with open("project_human-2020-06-18T05_06_15.630Z.json") as f:
        json_load = json.load(uploaded_file)

        #Classify Annotation task
        annotation_task = json_load[0]["project_kind"]
        st.success(annotation_task)

        if annotation_task == "detection":
            run_objdct(json_load)
        elif annotation_task == "segmentation_selectable":
            run_segment(json_load)
        elif annotation_task == "classify":
            run_image_classify(json_load)
        elif annotation_task == "text":
            run_text_classify(json_load)
        else:
            st.error("That task is unsupported yet")

###### Main ######

# Render the readme as markdown using st.markdown.
#readme_text = st.markdown(get_file_local_content_as_string("instructions.md"))
readme_text = st.markdown(get_file_content_as_string("instructions.md"))

# Once we have the dependencies, add a selector for the app mode on the sidebar.
st.sidebar.title("What to do")
app_mode = st.sidebar.selectbox("Choose the app mode",
    ["Show instructions", "Run the app", "Show the source code"])
if app_mode == "Show instructions":
    st.sidebar.success('To continue select "Run the app".')
elif app_mode == "Show the source code":
    readme_text.empty()
    st.code(get_file_content_as_string("app.py"))
elif app_mode == "Run the app":
    readme_text.empty()
    organization_id = st.sidebar.text_input('organization_id:')
    user_id = st.sidebar.text_input('user_id:')
    personal_access_token = st.sidebar.text_input('personal_access_tokena:')

    credential = {
      'user_id': user_id,
      'personal_access_token': personal_access_token
    }


    if st.sidebar.button("login"):
        # Setting Credential(High&Low API)
        api_client = APIClient(credential=credential)
        datalake_client = DatalakeClient(organization_id=organization_id, credential=credential)
        try:
            response = api_client.list_channels(organization_id)
        except:
            st.sidebar.error('Incorrect authentication information. Please check.')
        else:
            st.success('Success authentication!!')
            run_the_app()
    else:
        datalake_client = DatalakeClient(organization_id=organization_id, credential=credential)
        run_the_app()
