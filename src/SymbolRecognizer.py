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
    Classes = None
    ClassCount = -1

    # Class constuctor to setup weights
    def __init__(self, weightPath, classes, numclasses):
        self.LoadWeights(weightPath)
        self.Classes = classes
        self.ClassCount = numclasses

    # Import the selected weights file into YOLOv5
    def LoadWeights(self, weightPath):
        # Set the current weight to be the status
        self.ModelStatus = weightPath
        # Initialize the model
        self.Model = torch.hub.load('ultralytics/yolov5', 'custom', path = weightPath)
        # Print status
        print("\nYOLOv5 Model initiallized with weight: " + weightPath)

    # Run inference on the source images in the path with the model
    def ProcessSourceImages(self, srcPath):
        # Conduct Inference
        results = self.Model(srcPath)
        #results.save()  # or .show()
        results.show()
        outputMessage = self.SetupResultString(self.ProcessInferenceResults(results))
        return outputMessage

    # Process results of model inference to determine which symbol is most likely the result
    def ProcessInferenceResults(self, results):
        print("\nProcessing Inference Results")
        #print(results.xyxy[0])  # predictions (tensor)
        print(results.pandas().xyxy[0])  # predictions (pandas)
        labels, coord = results.xyxyn[0][:, -1].to('cpu').numpy(), results.xyxyn[0][:, :-1].to('cpu').numpy()
        #print(labels) # Yoink labels
        #print(coord) # Coords for each detected label
        # Area calculations
        if False: # Skipping this section first
            areaList = []
            #print(coord)
            for i in range(len(coord)): # Finding area is bugged
                y = coord[i][2] - coord[i][1] # Accessing coord values is a problem
                print(coord[i][2], coord[i][1])
                x = coord[i][3] - coord[i][0]
                areaList.append(x * y)
            print(areaList)
        return labels

    # Make use of processed results to create an output string to be sent back to RPi
    def SetupResultString(self, results):
        print("\nSetting Up Result Message")
        # Do stuff with passed labels
        label = "No Symbol Detected"
        for idx in results:
            i = int(idx)
            if (label[i] != "bullseye"):
                label = self.Classes[i]
        return label
