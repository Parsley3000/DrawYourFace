from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from io import BytesIO
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

device = "cuda"
model_id = "darkstorm2150/Protogen_v2.2_Official_Release"
pipe = StableDiffusionPipeline.from_pretrained(model_id ,torch_dtype=torch.float16)
pipe.to(device)

negitive_prompt_faces = "(((duplicate))), ((morbid)), ((mutilated)), out of frame, extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), out of frame, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck))), blurred lines, robot eyes"

@app.get("/")
def generate(prompt: str):
    with autocast(device): 
        image = pipe(prompt, negative_prompt=negitive_prompt_faces ,guidance_scale=8.5).images[0]
        
    image.save("testimage.png")
    img_buffer = BytesIO()
    image.save(img_buffer, format="PNG")
    imgstr = base64.b64encode(img_buffer.getvalue())
    
    return Response(content=imgstr, media_type="image/png")