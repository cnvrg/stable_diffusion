# Stable Diffusion optimized for CPU

This blueprint can be used for image generation. Blueprint will deploy a stable diffusion based model as an endpoint that can accessed directly or integrated with other applications. User will have three different ways to use the endpoint. 

- Text -> Image. 
- Text + Image -> Image (The image provided along with input text serves as reference image.)
- Text + Image + Mask Image -> Image (The mask image provided is a black and white image where the model will only make changes in the white portion of the image.)

In order to send the images along with the text, the user will need to conver the image to base64 string.

**Note**: Two other arguments namely steps and samples will need to be provided. Higher step count will result in better quality images. A minimum step size of 32 is recommended. Samples will determine the number of images being generated and sent back as a response.
The images received from the endpoint will be in base64 string format and will need to be decoded to get the final images.

## Example Usage

### Example Text-To-Image
```
curl -X POST \
    https://inference-869-1.arszihujmasy4t5o7ynxjvf.cloud.cnvrg.io/api/v1/endpoints/hbblvje4regt6weh6cds \
-H 'Cnvrg-Api-Key: YAbwiRZWB7H7EmgLFWojYrxo' \
-H 'Content-Type: application/json' \
-d '{"input_params": {"prompt": text, "staps":32, "samples":1}}'
```

### Example Image-To-Image
```
curl -X POST \
    https://inference-869-1.arszihujmasy4t5o7ynxjvf.cloud.cnvrg.io/api/v1/endpoints/hbblvje4regt6weh6cds \
-H 'Cnvrg-Api-Key: YAbwiRZWB7H7EmgLFWojYrxo' \
-H 'Content-Type: application/json' \
-d '{"input_params": {"prompt": text, "staps":32, "samples":1,"init_image": <base64string> }}'
```

### Example Inpainting
```
curl -X POST \
    https://inference-869-1.arszihujmasy4t5o7ynxjvf.cloud.cnvrg.io/api/v1/endpoints/hbblvje4regt6weh6cds \
-H 'Cnvrg-Api-Key: YAbwiRZWB7H7EmgLFWojYrxo' \
-H 'Content-Type: application/json' \
-d '{"input_params": {"prompt": text, "staps":32, "samples":1,"init_image": <base64string>, "mask_image": <base64string>}}'
```

### Sample response:

```
{
    "prediction": "images" :
                            [
                            <base64string 1>,
                            <base64string 2> 
                            ..,
                            ]
}

```

## References
https://github.com/bes-dev/stable_diffusion.openvino