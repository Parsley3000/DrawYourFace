from fastapi import FastAPI, Response, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import io
from io import BytesIO
import base64
import torch
from torch import autocast
from diffusers.utils import load_image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler
import numpy as np
import cv2
from PIL import Image
from rembg import remove
import random
import dlib
from matplotlib import pyplot as plt # Good for graphing; install using `pip install matplotlib`
from matplotlib import image as image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "darkstorm2150/Protogen_v2.2_Official_Release", controlnet=controlnet, torch_dtype=torch.float16
    )
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.enable_xformers_memory_efficient_attention()


@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...), option: str = Form(...), gender: str = Form(...)):
    contents = await file.read()
    pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    pil_image.thumbnail((512, 512))
        
    #remove background
    image_no_background = remove(pil_image)
    image_no_background.save("noBackground.png")
    
    image = np.array(image_no_background)

    low_threshold = 100
    high_threshold = 200

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    canny_image.save("cannyImage.png")
    
    prompt = " line art of a" 
    prompt = option + prompt
    
    if gender != 'None':
        prompt = prompt + " young " + gender + " adult, portrait, line drawing, white background"
    
    print(prompt)

    generator = torch.Generator(device="cpu").manual_seed(random.randint(0,1000))
                
    output = pipe(
        prompt,
        canny_image,
        negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality, nsfw",
        num_inference_steps=20,
        guidance_scale=12,
        generator=generator,
        height=512,
        width=512
    ).images[0]

    #image to cv2 image
    output = np.array(output) 
    gray_output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    #image to pil image
    gray_output = Image.fromarray(gray_output)
    gray_output.save("testimage.png")
    
    img_buffer = BytesIO()
    gray_output.save(img_buffer, format="PNG")
    imgstr = base64.b64encode(img_buffer.getvalue())
    
    return Response(content=imgstr, media_type="image/png")


@app.post("/get_data")
async def get_data(file: UploadFile = File(...)):
    images = []
    contents = await file.read()
    image = Image.open(contents)
    
    #image to cv2 image
    image = np.array(image) 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #image to pil image
    image = Image.fromarray(image)

    original = image.copy()

    predictor_path = 'shape_predictor_81_face_landmarks.dat'

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    a4_square_size = 210 #210mm

    jaw = [0,16]
    left_eyebrow = [17,21]
    right_eyebrow = [22,26]
    nose = [27,35]
    left_eye = [36,41]
    right_eye = [42,47]
    lips_outer = [48,60]
    lips_inner = [61,67]
    hair_line = [68,80]


    #blank white image 512x512
    white_image = np.zeros([512,512,1],dtype=np.uint8)
    white_image.fill(255)

    # Define grid parameters
    rows, cols, _ = white_image.shape
    grid_size = 32
    color = (200, 200, 200) # light gray color
    gridThickness = 1

    # Draw horizontal lines
    for i in range(0, rows, grid_size):
        cv2.line(white_image, (0, i), (cols, i), color, thickness=gridThickness)

    # Draw vertical lines
    for j in range(0, cols, grid_size):
        cv2.line(white_image, (j, 0), (j, rows), color, thickness=gridThickness)

    dets = detector(image, 0)
    for k, d in enumerate(dets):
        
        shape = predictor(image, d)
        landmarks = np.matrix([[p.x, p.y] for p in shape.parts()])
        lineThickness = 1
        
        #white_image = image
        
    #    for x in range(jaw[0], jaw[1]+1):
    #        cv2.circle(image, (landmarks[x, 0],landmarks[x, 1]), 3, (0,255,0), -1)

        #crossline 
        cl_point1 = [landmarks[0,0]-20,landmarks[0,1]]
        cl_point2 = [landmarks[16,0]+20,landmarks[0,1]]
        cv2.line(white_image, cl_point1, cl_point2, (0, 0, 0), thickness=lineThickness)
        
        buffered = BytesIO()
        white_image.save(buffered, format="PNG")
        encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
        images.append({"filename": "Instruction1", "data": encoded_string})
        
        #------------------------------------------------------------
        
        #Vertical line
        vl_point1 = [landmarks[8,0],landmarks[8,1]+20]
        vl_point2 = [landmarks[8,0], landmarks[0,1]-int((landmarks[16,0] - landmarks[0,0])/2)-20]
        cv2.line(white_image, vl_point1, vl_point2, (0, 0, 0), thickness=lineThickness)
        
        buffered = BytesIO()
        white_image.save(buffered, format="PNG")
        encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
        images.append({"filename": "Instruction2", "data": encoded_string})
        
        #------------------------------------------------------------
        
        #Circle
        circle_point = [landmarks[0,0]+int((landmarks[16,0] - landmarks[0,0])/2), landmarks[0,1]]
        cv2.circle(white_image, (circle_point), int((landmarks[16,0] - landmarks[0,0])/2), (0,0,0), thickness=lineThickness)
        
        buffered = BytesIO()
        white_image.save(buffered, format="PNG")
        encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
        images.append({"filename": "Instruction3", "data": encoded_string})
        
        #------------------------------------------------------------
        
        #jaw
        jaw_points = landmarks[jaw[0]:jaw[1]+1]
        cv2.polylines(white_image, [jaw_points], False, (0, 0, 0), thickness=lineThickness)
        
        buffered = BytesIO()
        white_image.save(buffered, format="PNG")
        encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
        images.append({"filename": "Instruction4", "data": encoded_string})
        
        #------------------------------------------------------------
        
        #right Eye
        right_eye_points = landmarks[right_eye[0]:right_eye[1]+1]
        right_eye_points = np.concatenate([right_eye_points, landmarks[right_eye[0]]])
        cv2.polylines(white_image, [right_eye_points], False, (0, 0, 0), thickness=lineThickness)
        
        buffered = BytesIO()
        white_image.save(buffered, format="PNG")
        encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
        images.append({"filename": "Instruction5", "data": encoded_string})
        
        #------------------------------------------------------------
        
        #right Eyebrow
        right_eyebrow_points = landmarks[right_eyebrow[0]:right_eyebrow[1]+1]
        for x in range(right_eyebrow[1],right_eyebrow[0]+1, -1 ):
            new_point = landmarks[x-1]
            new_point[0,1] = new_point[0,1] + 10
            right_eyebrow_points = np.concatenate([right_eyebrow_points, new_point])
        
        right_eyebrow_points[3,1] = right_eyebrow_points[3,1] - 10
        right_eyebrow_points = np.concatenate([right_eyebrow_points, landmarks[right_eyebrow[0]]])
        cv2.polylines(white_image, [right_eyebrow_points], False, (0, 0, 0), thickness=lineThickness)
        
        buffered = BytesIO()
        white_image.save(buffered, format="PNG")
        encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
        images.append({"filename": "Instruction6", "data": encoded_string})
        
        #------------------------------------------------------------
        
        #Left eye
        left_eye_points = landmarks[left_eye[0]:left_eye[1]+1]
        left_eye_points = np.concatenate([left_eye_points, landmarks[left_eye[0]]])
        cv2.polylines(white_image, [left_eye_points], False, (0, 0, 0), thickness=lineThickness)
        #ellipse = cv2.fitEllipse(left_eye_points)
        #cv2.ellipse(image, ellipse, (0, 255, 255), thickness=2)
        
        buffered = BytesIO()
        white_image.save(buffered, format="PNG")
        encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
        images.append({"filename": "Instruction7", "data": encoded_string})
        
        #------------------------------------------------------------
        
        #left Eyebrow
        left_eyebrow_points = landmarks[left_eyebrow[0]:left_eyebrow[1]+1]
        for x in range(left_eyebrow[1],left_eyebrow[0]+1, -1 ):
            new_point = landmarks[x-1]
            new_point[0,1] = new_point[0,1] + 10
            left_eyebrow_points = np.concatenate([left_eyebrow_points, new_point])
        
        left_eyebrow_points[3,1] = left_eyebrow_points[3,1] - 10
        left_eyebrow_points = np.concatenate([left_eyebrow_points, landmarks[left_eyebrow[0]]])
        cv2.polylines(white_image, [left_eyebrow_points], False, (0, 0, 0), thickness=lineThickness)
        
        buffered = BytesIO()
        white_image.save(buffered, format="PNG")
        encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
        images.append({"filename": "Instruction8", "data": encoded_string})
        
        #------------------------------------------------------------
        
        #nose
        nose_bottom_points = landmarks[31:35+1]
        nose_join_bottom = [[landmarks[31,0]-5,landmarks[33,1]], [landmarks[35,0]+5,landmarks[33,1]]]
        cv2.line(white_image, nose_join_bottom[0], nose_join_bottom[1], (0, 0, 0), thickness=lineThickness)
        
        nose_join_top = [[landmarks[32,0]-2,landmarks[30,1]+5], [landmarks[34,0]+2,landmarks[30,1]+5]]
        cv2.line(white_image, nose_join_top[0], nose_join_top[1], (0, 0, 0), thickness=lineThickness)
        
        nose_join_left = np.array([[landmarks[31,0]-5, landmarks[33,1]], [landmarks[32,0]-2, landmarks[30,1]+5], [landmarks[27,0]-10, landmarks[27,1]+5], [landmarks[21,0], landmarks[21,1]+15], [landmarks[27,0]-10, landmarks[27,1]+5], [landmarks[39,0]+9, landmarks[39,1]], [landmarks[39,0]-1, landmarks[39,1]-10], [landmarks[39,0]+9, landmarks[39,1]], [landmarks[31,0]-5, landmarks[33,1]]])
        cv2.polylines(white_image, [nose_join_left], False, (0, 0, 0), thickness=lineThickness)
        
        nose_join_right = np.array([[landmarks[35,0]+5, landmarks[33,1]], [landmarks[34,0]+2, landmarks[30,1]+5], [landmarks[27,0]+10, landmarks[27,1]+5], [landmarks[22,0], landmarks[21,1]+15], [landmarks[27,0]+10, landmarks[27,1]+5], [landmarks[42,0]-9, landmarks[39,1]], [landmarks[42,0]+1, landmarks[39,1]-10], [landmarks[42,0]-9, landmarks[39,1]], [landmarks[35,0]+5, landmarks[33,1]]])
        cv2.polylines(white_image, [nose_join_right], False, (0, 0, 0), thickness=lineThickness)
        
        buffered = BytesIO()
        white_image.save(buffered, format="PNG")
        encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
        images.append({"filename": "Instruction9", "data": encoded_string})
        
        #------------------------------------------------------------
        
        #lips outer
        lips_outer_points = landmarks[lips_outer[0]:lips_outer[1]]
        lips_outer_points = np.concatenate([lips_outer_points, landmarks[lips_outer[0]]])
        cv2.polylines(white_image, [lips_outer_points], False, (0, 0, 0), thickness=lineThickness)
        
        #lips hozizontal line
        lips_hl_points = [[landmarks[48,0], landmarks[48,1]], [landmarks[54,0], landmarks[54,1]]]
        cv2.line(white_image, lips_hl_points[0], lips_hl_points[1], (0, 0, 0), thickness=lineThickness)
        
        buffered = BytesIO()
        white_image.save(buffered, format="PNG")
        encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
        images.append({"filename": "Instruction10", "data": encoded_string})
        
        
        #nose_brim_points = landmarks[27:30+1]
        #nose_brim_points = np.concatenate([nose_brim_points, landmarks[33]])
        #cv2.polylines(white_image, [nose_brim_points], False, (0, 0, 0), thickness=lineThickness)
        
        # Convert to grayscale
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

        # Compute median pixel value
        median = np.median(gray)

        # Compute threshold values using median
        low = int(max(0, 0.66 * median))
        high = int(min(255, 1.33 * median))

        # Apply Canny edge detection
        edges = cv2.Canny(gray, low, high)

        # Invert image
        inverted_img = 255 - edges

        # Resize images to same size (if necessary)
        #img2 = cv2.resize(img2, img1.shape[1::-1])

        # Choose weight factors for blending
        alpha = 0.5
        beta = 1 - alpha

        # Combine the images using the cv2.addWeighted() function
        image = cv2.addWeighted(white_image, alpha, inverted_img, beta, 0)
        
        buffered = BytesIO()
        white_image.save(buffered, format="PNG")
        encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
        images.append({"filename": "Instruction 1", "data": encoded_string})
            
    text = ["Draw horzonital line", "Vertical line", "Draw circle", "Draw jaw", "Draw right eye", "Draw right eye brow", "Draw left eye", "Draw left eye", "Draw nose", "Draw lips", "final drawing"]
    return {"images": images, "text": text}