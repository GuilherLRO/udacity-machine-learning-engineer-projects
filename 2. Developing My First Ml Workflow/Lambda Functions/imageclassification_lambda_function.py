# p2_imageclassification lambda_function.py
import json
import boto3
import base64

endpoint = 'image-classification-endpoint'

def lambda_handler(event,contex):
    # Decode the image data
    image = base64.b64decode(event["body"]["image_data"])

    # Instantiate a runtime client
    runtime = boto3.client("runtime.sagemaker")

    # Make a prediction request
    response = runtime.invoke_endpoint(
        EndpointName=endpoint,
        ContentType="image/png",
        Body=image,
    )

    # Parse the response
    inferences = response["Body"].read().decode()
    
    # Add inferences to event
    event["body"]["inferences"] = inferences

    # Return response to Step Function
    return event

    