from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
import base64
from pydantic import BaseModel
from typing import List

import torch
from torch import autocast
from diffusers.utils import load_image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler
import numpy as np
import cv2
import dlib
from PIL import Image
from rembg import remove
import random

app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageData(BaseModel):
    image: str
    gender: int
    age: int
    style: str
    
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "darkstorm2150/Protogen_v2.2_Official_Release", controlnet=controlnet, torch_dtype=torch.float16
    )
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.enable_xformers_memory_efficient_attention()

@app.post("/rotate-image/")
async def rotate_image(image_data: ImageData):
    base64_img = image_data.image.split(',')[1]
    image_bytes = base64.b64decode(base64_img)
    pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    pil_image.thumbnail((512, 512))

    # Print the received number
    gender_strings = [" Male ", " Slightly Male ", " ", " Slightly Female ", " Female "]
    print(gender_strings[image_data.gender])
    
    age_strings = [" Very Young ", " Young ", " Middle Aged ", " Old ", " Very Old "]
    print(age_strings[image_data.age])
    
    print(image_data.style)
    
      
    #-----------------------------  
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
    prompt = image_data.style + prompt
    
    
    prompt = prompt + age_strings[image_data.age] + gender_strings[image_data.gender] + " , portrait, line drawing, white background"
    
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
    #
    
    output_buffer = BytesIO()
    gray_output.save(output_buffer, format='PNG')
    base64_output = base64.b64encode(output_buffer.getvalue()).decode('utf-8')

    return {"image": base64_output}

class InstructionsData(BaseModel):
    image: str
    
def add_instructions(white_image, instructions, output_buffer):
    pil_image = Image.fromarray(cv2.cvtColor(white_image, cv2.COLOR_BGR2RGB))
    pil_image.save(output_buffer, format='PNG')
    base64_output = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
    instructions.append(base64_output)

@app.post("/instructions/")
async def generate_instructions(image_data: InstructionsData):
    base64_img = image_data.image.split(',')[1]
    image_bytes = base64.b64decode(base64_img)
    image = Image.open(BytesIO(image_bytes))
    
    #convert to cv2
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    original = image.copy()

    predictor_path = 'shape_predictor_81_face_landmarks.dat'

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    
    dets = detector(image, 0)
    
    #If no face is detected
    if not dets: 
        no_face = []
        failure_image = cv2.imread("noFace.png")
        
        output_buffer = BytesIO()
        pil_image = Image.fromarray(cv2.cvtColor(failure_image, cv2.COLOR_BGR2RGB))
        pil_image.save(output_buffer, format='PNG')
        base64_output = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
        no_face.append(base64_output)
        
        text = ["No face found: please try generating a new image or replace uploaded image"]
        
        return {"images": no_face, "texts": text}
    
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
        
    instructions = []
        
    for k, d in enumerate(dets):
        
        shape = predictor(image, d)
        landmarks = np.matrix([[p.x, p.y] for p in shape.parts()])
        lineThickness = 1
        
        #white_image = image
        
        #    for x in range(jaw[0], jaw[1]+1):
        #        cv2.circle(image, (landmarks[x, 0],landmarks[x, 1]), 3, (0,255,0), -1)
        
        #blankgrid
        output_buffer = BytesIO()
        pil_image = Image.fromarray(cv2.cvtColor(white_image, cv2.COLOR_BGR2RGB))
        pil_image.save(output_buffer, format='PNG')
        base64_output = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
        instructions.append(base64_output)

        #crossline 
        cl_point1 = [landmarks[0,0]-20,landmarks[0,1]]
        cl_point2 = [landmarks[16,0]+20,landmarks[0,1]]
        cv2.line(white_image, cl_point1, cl_point2, (0, 0, 0), thickness=lineThickness)
        
        #add image to array - need to add function to remove repeat code
        output_buffer = BytesIO()
        pil_image = Image.fromarray(cv2.cvtColor(white_image, cv2.COLOR_BGR2RGB))
        pil_image.save(output_buffer, format='PNG')
        base64_output = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
        instructions.append(base64_output)
        
        #------------------------------------------------------------
        
        #Vertical line
        vl_point1 = [landmarks[8,0],landmarks[8,1]+20]
        vl_point2 = [landmarks[8,0], landmarks[0,1]-int((landmarks[16,0] - landmarks[0,0])/2)-20]
        cv2.line(white_image, vl_point1, vl_point2, (0, 0, 0), thickness=lineThickness)
        
        output_buffer = BytesIO()
        pil_image = Image.fromarray(cv2.cvtColor(white_image, cv2.COLOR_BGR2RGB))
        pil_image.save(output_buffer, format='PNG')
        base64_output = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
        instructions.append(base64_output)
        
        #------------------------------------------------------------
        
        #Circle
        circle_point = [landmarks[0,0]+int((landmarks[16,0] - landmarks[0,0])/2), landmarks[0,1]]
        cv2.circle(white_image, (circle_point), int((landmarks[16,0] - landmarks[0,0])/2), (0,0,0), thickness=lineThickness)
        
        output_buffer = BytesIO()
        pil_image = Image.fromarray(cv2.cvtColor(white_image, cv2.COLOR_BGR2RGB))
        pil_image.save(output_buffer, format='PNG')
        base64_output = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
        instructions.append(base64_output)
        
        #------------------------------------------------------------
        
        #jaw
        jaw_points = landmarks[jaw[0]:jaw[1]+1]
        cv2.polylines(white_image, [jaw_points], False, (0, 0, 0), thickness=lineThickness)
        
        output_buffer = BytesIO()
        pil_image = Image.fromarray(cv2.cvtColor(white_image, cv2.COLOR_BGR2RGB))
        pil_image.save(output_buffer, format='PNG')
        base64_output = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
        instructions.append(base64_output)
        
        #------------------------------------------------------------
        
        #right Eye
        right_eye_points = landmarks[right_eye[0]:right_eye[1]+1]
        right_eye_points = np.concatenate([right_eye_points, landmarks[right_eye[0]]])
        cv2.polylines(white_image, [right_eye_points], False, (0, 0, 0), thickness=lineThickness)
        
        output_buffer = BytesIO()
        pil_image = Image.fromarray(cv2.cvtColor(white_image, cv2.COLOR_BGR2RGB))
        pil_image.save(output_buffer, format='PNG')
        base64_output = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
        instructions.append(base64_output)
        
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
        
        output_buffer = BytesIO()
        pil_image = Image.fromarray(cv2.cvtColor(white_image, cv2.COLOR_BGR2RGB))
        pil_image.save(output_buffer, format='PNG')
        base64_output = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
        instructions.append(base64_output)
        
        #------------------------------------------------------------
        
        #Left eye
        left_eye_points = landmarks[left_eye[0]:left_eye[1]+1]
        left_eye_points = np.concatenate([left_eye_points, landmarks[left_eye[0]]])
        cv2.polylines(white_image, [left_eye_points], False, (0, 0, 0), thickness=lineThickness)
        #ellipse = cv2.fitEllipse(left_eye_points)
        #cv2.ellipse(image, ellipse, (0, 255, 255), thickness=2)
        
        output_buffer = BytesIO()
        pil_image = Image.fromarray(cv2.cvtColor(white_image, cv2.COLOR_BGR2RGB))
        pil_image.save(output_buffer, format='PNG')
        base64_output = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
        instructions.append(base64_output)
        
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
        
        output_buffer = BytesIO()
        pil_image = Image.fromarray(cv2.cvtColor(white_image, cv2.COLOR_BGR2RGB))
        pil_image.save(output_buffer, format='PNG')
        base64_output = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
        instructions.append(base64_output)
        
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
        
        output_buffer = BytesIO()
        pil_image = Image.fromarray(cv2.cvtColor(white_image, cv2.COLOR_BGR2RGB))
        pil_image.save(output_buffer, format='PNG')
        base64_output = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
        instructions.append(base64_output)
        
        #------------------------------------------------------------
        
        #lips outer
        lips_outer_points = landmarks[lips_outer[0]:lips_outer[1]]
        lips_outer_points = np.concatenate([lips_outer_points, landmarks[lips_outer[0]]])
        cv2.polylines(white_image, [lips_outer_points], False, (0, 0, 0), thickness=lineThickness)
        
        #lips hozizontal line
        lips_hl_points = [[landmarks[48,0], landmarks[48,1]], [landmarks[54,0], landmarks[54,1]]]
        cv2.line(white_image, lips_hl_points[0], lips_hl_points[1], (0, 0, 0), thickness=lineThickness)
        
        output_buffer = BytesIO()
        pil_image = Image.fromarray(cv2.cvtColor(white_image, cv2.COLOR_BGR2RGB))
        pil_image.save(output_buffer, format='PNG')
        base64_output = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
        instructions.append(base64_output)
        
        
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
        combined = cv2.addWeighted(white_image, alpha, inverted_img, beta, 0)
        
        output_buffer = BytesIO()
        pil_image = Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
        pil_image.save(output_buffer, format='PNG')
        base64_output = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
        instructions.append(base64_output)
        
        #Add orginal image
        output_buffer = BytesIO()
        pil_image = Image.fromarray(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        pil_image.save(output_buffer, format='PNG')
        base64_output = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
        instructions.append(base64_output)


    #######################################################
    texts = ["Blank Grid page can be printed from the first page of the pdf, download in bottom left",
             "Step 1: Using the grid as a reference start by draw a line horizontally about 2/3 width across near the middle of the page",
             "Step 2: Then from the centre of the horizontal, draw a vertical line about 3/4 the lenght of the page. It should be slightly longer on the bottom",
             "Step 3: From the intersection of the 2 lines draw a draw a circle with a radius just shorter then half the lenght of horizontal line",
             "Step 4: Where the circle intersects the horizontal line, draw a arc downwards that slowly increases in gradient as it nears the vertical line. This can continue back up to the other side of the circle intersect and will be our Jaw line",
             "Step 5: On the right side of the horizontal line on the face, draw a small oval crossing over the center of the line. This will be the right eye",
             "Step 6: Just above this oval draw the outline of the right eyebrow, which should be a downward bending arc that is slightly wider then the eye",
             "Step 7: Copy the oval from step 5 over onto the left side of the face. This will be the left eye",
             "Step 8: Similarly, invert and copy over the eyebrow from step 6 onto the left side of the face",
             "Step 9: To outline the nose, roughly draw a open top rectangle just inbetween the eyes that goes half way down towards the chin very slightly widening at the bottom. Inside the draw a smaller rectangle and use lines to connect the bottom corners of each retangle. At the top of each side of these rectangles draw a small square U shape pointing inwards towards the eyes",
             "Step 10: The lips can be finally added by drawing another vertical line roughly just above the bottom intersect of the circle, then draw a rough oval shape around this line with a small indent at the top",
             "Step 11: Now that rough proportions of the face have been lined out, further details can be added over each feature and other pieces such as the neck and hair can be added",
             "Step 12: Finally by add things such as shadowing and blending the final image can be reached"]
    

    return {"images": instructions, "texts": texts}