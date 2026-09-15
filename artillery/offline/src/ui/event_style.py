"""Stable event identity shared by video, map and coordinate cards (BGR)."""

def event_label(event_id):
    return f"S{event_id:02d}"


def event_color(event_id):
    palette = ((70, 170, 255), (220, 190, 70), (180, 110, 240),
               (110, 210, 100), (220, 140, 100), (100, 120, 255))
    return palette[int(event_id) % len(palette)]
