You can deploy the Stable Diffusion model optimized to run on CPUs to use it via API calls. Once deployed the model will take as input raw text as prompt, along with the number of iterations and number of example images you want to create. This blueprint supports one click deployment. Follow the below steps to get started.

1. Click on `Use Blueprint` button
2. In the pop up, choose the relevant compute you want to use to deploy your API endpoint
3. You will be redirected to your endpoint
4. You can now use the `Try it Live` section with any text or link. 
5. You can now integrate your API with your code using the integration panel at the bottom of the page
6. You will now have a functioning API endpoint that returns base64 strings that you can decode to get get the images.

## Example Input
Text:   
```
prompt: Horse dancing in the rain
steps: 32
samples: 1
```  
Labels: 
```
<Base64 string >
```
