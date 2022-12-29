You can use this blueprint to generate an image based on a prompt. In order to train this model with your data, you would need to provide one folder located in s3:

stable_diffusion: the folder where the image file (which can be used to create a higher quality version of the same image) stored.
Click on Use Blueprint button

You will be redirected to your blueprint flow page

In the flow, edit the following tasks to provide your data:

In the S3 Connector task:

Under the bucketname parameter provide the bucket name of the data
Under the prefix parameter provide the main path to where the image file is located
In the Batch-Predict task:

Under the input_path parameter provide the path to the input file including the prefix you provided in the S3 Connector, it should look like: /input/s3_connector/<prefix>/sample_image.png
NOTE: You can use prebuilt data examples paths that are already provided

Click on the 'Run Flow' button
In a few minutes you will deploy a blueprint that will generate a custom image.
Go to the 'Serving' tab in the project and look for your endpoint
You can also integrate your API with your code using the integration panel at the bottom of the page
Congrats! You have deployed a blueprint that generates images via prompts and improved versions of custom images!