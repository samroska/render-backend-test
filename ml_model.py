from skin_lesion_classifier import SkinLesionClassifier

def inference_function(image):
    """
    Legacy inference function that uses the static classifier class.
    
    Args:
        image: PIL Image object or path to image file
        
    Returns:
        Dict[str, float]: Prediction results
    """
    try:
        predictions = SkinLesionClassifier.predict(image)
        
        # Print results in the original format
        print('\nProbabilities:')
        print('ACK: ' + str(predictions['ACK']))
        print('BCC: ' + str(predictions['BCC']))
        print('MEL: ' + str(predictions['MEL']))
        print('NEV: ' + str(predictions['NEV']))
        print('SCC: ' + str(predictions['SCC']))
        print('SEK: ' + str(predictions['SEK']))
        
        return predictions
        
    except Exception as e:
        print(f"Error in inference: {e}")
        raise