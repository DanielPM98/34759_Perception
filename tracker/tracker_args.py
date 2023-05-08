from dataclasses import dataclass

@dataclass(frozen=True)
class BYTETrackerArgs:
    track_low_thresh = 0.2
    track_high_thresh = 0.6
    new_track_thresh: float = 0.15
    track_buffer: int = 300
    match_thresh: float = 0.9
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False