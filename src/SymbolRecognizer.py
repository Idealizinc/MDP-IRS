# SymbolRecognizer - Recognizes symbols from input images using a weighted YOLOv5 Model
# @author Lim Rui An, Ryan
# @version 1.0
# @since 2022-02-10
# @modified 2022-02-10

# Imported dependencies
import torch # For the inference model

class SymbolRecognizer:
    ModelStatus = "nil"
    Model = None

    # Class constuctor to setup weights
    def __init__(self, weightPath):
        self.LoadWeights(weightPath)

    # Import the selected weights file into YOLOv5
    def LoadWeights(self, weightPath):
        # Set the current weight to be the status
        self.ModelStatus = weightPath
        # Initialize the model
        self.Model = torch.hub.load('ultralytics/yolov5', 'custom', path = weightPath)
        # Print status
        print("YOLOv5 Model initiallized with weight: " + weightPath)

    # Run inference on the source images in the path with the model
    def ProcessSourceImages(self, srcPath):
        # Conduct Inference
        results = self.Model(srcPath)
        #results.save()  # or .show()
        results.show()
        #labels, cord_thres = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
        outputMessage = self.SetupResultString(self.ProcessInferenceResults(results))
        return 0

    # Process results of model inference to determine which symbol is most likely the result
    def ProcessInferenceResults(self, results):
        print("Processing Inference Results")
        print(results.xyxy[0])  # predictions (tensor)
        print(results.pandas().xyxy[0])  # predictions (pandas)
        return 0

    # Make use of processed results to create an output string to be sent back to RPi
    def SetupResultString(self, results):
        print("Setup Result Message")
        return "test"
