from mp3dpose import PoseEstimator

if __name__ == "__main__":
    pose_estimator = PoseEstimator()
    #pose_estimator.process_image('media/test_image.jpg')
    pose_estimator.process_video('media/test_video.mp4')
    #pose_estimator.process_webcam()