# p2_inferencefilter lambda_function.py
import json
def lambda_handler(event, context = None):
    
    # Grab the inferences from the event
    inferences = json.loads(event["body"]["inferences"])
    treshold = 0.93
    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = 1 if inferences[0] > treshold else 0
    
    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    
    if meets_threshold:
        pass
    else:
        raise("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }