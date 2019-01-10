# DEPS:
# SIFT detector
# YAML reader
# PLY reader

# Loop over all objects
    # Read object model from file

    # Loop over all images
        # Read camera calibration etc. from yaml
        # Read pose annotations from yaml

        # Apply SIFT detector
        # Normalize pixel coordinates using calibration
        # Determine viewing ray from 2D point, transform from camera coordinate system to object coordinate system using inverse annotated pose
        # Find all vertices within some distance from line (radius depends on SIFT scale)
        # Filter vertices based on depth along viewing ray (project to viewing ray, obtaining the coordinate lambda along this axis). Select only vertices for which lambda is close enough to lambda_min (mm threshold).

        # Each vertex gets a vote depending on:
            # Detector score
            # Distance from line - gaussian kernel weighting according to SIFT scale?

    # Pick top-n vertices for object
