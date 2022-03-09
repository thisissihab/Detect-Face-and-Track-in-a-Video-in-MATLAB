% Create a cascade detector object.
faceDetector = vision.CascadeObjectDetector();

% Read a video frame and run the face detector.
videoReader = VideoReader('up2.mp4');
videoFrame   = readFrame(videoReader);
faceLocation = step(faceDetector, videoFrame);

% Draw the returned bounding box around the detected face.
detectedFrame = insertShape(videoFrame, 'Rectangle', faceLocation);
figure; imshow(detectedFrame); title('Detected face');

%%
faceLocationPoints = bbox2points(faceLocation(1, :));
featurePoints = detectMinEigenFeatures(rgb2gray(detectedFrame), 'ROI', faceLocation);

pointTracker = vision.PointTracker('MaxBidirectionalError', 2);

% Initialize the tracker with the initial point locations and the initial
% video frame.
featurePoints = featurePoints.Location;
initialize(pointTracker, featurePoints, detectedFrame);

%Define the video player object property
videoPlayer  = vision.VideoPlayer('Position',...
    [100 100 [size(detectedFrame, 2), size(detectedFrame, 1)]+30]);

oldPoints = featurePoints;

while hasFrame(videoReader)
    % get the next frame
    videoFrame = readFrame(videoReader);

    % Track the points. Note that some points may be lost.
    [featurePoints, isFound] = step(pointTracker, videoFrame);
    newPoints = featurePoints(isFound, :);
    old_points = oldPoints(isFound, :);
    
    if size(newPoints, 1) >= 2 % need at least 2 points
        
        % Estimate the geometric transformation between the old points
        % and the new points and eliminate outliers
        [transformed_Rectangle, old_points, newPoints] = estimateGeometricTransform(...
            old_points, newPoints, 'similarity', 'MaxDistance', 4);
        
        % Apply the transformation to the bounding box points
        faceLocationPoints = transformPointsForward(transformed_Rectangle, faceLocationPoints);
                
        % Insert a bounding box around the object being tracked
        reshaped_Rectangle = reshape(faceLocationPoints', 1, []);
        detectedFrame = insertShape(videoFrame, 'Polygon', reshaped_Rectangle, ...
            'LineWidth', 2);
                
        % Display tracked points
        detectedFrame = insertMarker(detectedFrame, newPoints, '+', ...
            'Color', 'white');       
        
        % Reset the points
        oldPoints = newPoints;
        setPoints(pointTracker, oldPoints);        
    end
    
    % Display the annotated video frame using the video player object
    step(videoPlayer, detectedFrame);
end

% Clean up
release(videoPlayer);