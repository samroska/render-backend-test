from skin_lesion_classifier import SkinLesionClassifier


classifier = SkinLesionClassifier('PAD-UFES-20.keras')

def inference_function(image):
 
    try:
        predictions = classifier.predict(image)
        
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