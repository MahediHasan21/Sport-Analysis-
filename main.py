from utils import read_video, save_video
from trackers.tracker import Traker
from team_assigner import TeamAssigner
from camera_movement_estimator.camera_movement_estimator import CameraMovementEstimator
import cv2
import numpy as np
from player_ball_assigner import PlayerBallAssigner
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator

def main():
    # Read Video
    video_frames = read_video('/Users/apple/Desktop/Final_Project_Thesis copy/Football game analytics and player detections/input_videos/08fd33_4.mp4')

    # ✅ Check if frames were extracted
    if not video_frames:
        print("❌ ERROR: No frames were extracted from the video. Exiting.")
        exit()  # ✅ Exit early to prevent further errors

    # Initialize Tracker
    tracker = Traker('models/best.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
    # Get object positions 
    tracker.add_position_to_tracks(tracks)

    # Camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames, read_from_stub=True, stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # View Transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Assign Ball Acquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    
    team_ball_control = np.array(team_ball_control)

   # ...existing code...

    # Draw output
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    print(f"After draw_annotations: type={type(output_video_frames)}, length={len(output_video_frames)}")
    if len(output_video_frames) > 0:
        print(f"First frame type: {type(output_video_frames[0])}, shape: {np.shape(output_video_frames[0])}")

    ## Draw Camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
    print(f"After draw_camera_movement: type={type(output_video_frames)}, length={len(output_video_frames)}")
    if len(output_video_frames) > 0:
        print(f"First frame type: {type(output_video_frames[0])}, shape: {np.shape(output_video_frames[0])}")

    ## Draw Speed and Distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)
    print(f"After draw_speed_and_distance: type={type(output_video_frames)}, length={len(output_video_frames)}")
    if len(output_video_frames) > 0:
        print(f"First frame type: {type(output_video_frames[0])}, shape: {np.shape(output_video_frames[0])}")

    # Save video
    save_video(output_video_frames, 'output_videos/output_video.avi')
    print("✅ Video saved successfully.")

if __name__ == '__main__':
    main()

# Compare this snippet from utils/__init__.py:clear
